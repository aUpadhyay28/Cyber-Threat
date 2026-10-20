"""The client your login backend should use to call this service.

Read this before wiring the engine into a real login flow, because the default failure mode of a
naive integration is an outage.

The problem
-----------
Your login endpoint now depends on a second service. If that service is slow, restarting, or
unreachable, a naive client blocks on it and your users cannot sign in at all. A risk engine that
takes the whole site down has done far more damage than the credential stuffing it was meant to
stop. So the integration needs three decisions made deliberately, up front:

1. A TIMEOUT, set above the service's p99 and below what your login page can tolerate. Measured on
   a single worker: p99 was 24 ms at low concurrency and 511 ms at 32 concurrent requests. 1000 ms
   is a sane starting point; re-measure with scripts/loadtest.py on your own hardware.
2. A DEFAULT DECISION for when the call fails. This client defaults to FAIL OPEN - on error the
   login proceeds as if Low risk.
3. A RECORD of every failure, because a service that is failing open silently is indistinguishable
   from a service that is working, and you will not notice it has been down for a week.

Why fail open, and when not to
------------------------------
Fail open is right for the overwhelming majority of deployments: a password check already happened,
so failing open returns you to the security posture you had before this engine existed. Failing
closed means a Redis blip locks out every user on the site.

Fail CLOSED is defensible only for narrow, high-value flows - changing a payout account, an admin
login - where "come back in five minutes" beats "maybe that was an attacker". Decide per flow, not
globally, and if you do fail closed anywhere, make sure your support team can get a human in.

A circuit breaker is included: after `breaker_threshold` consecutive failures the client stops
calling for `breaker_reset_seconds` and returns the default immediately. Without it, a dead service
turns every login into a full timeout wait, and your login page gets as slow as the timeout.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("riskengine.client")

# What the caller sees when the service could not be reached. `degraded` is the flag your login
# code should branch on if it wants to treat "we don't know" differently from "we know it's fine".
FAIL_OPEN_RESULT = {"risk_score": 0.0, "tier": "Low", "action": "allow", "ip_risk": None,
                    "takeover_risk": None, "reasons": [], "event_id": None, "degraded": True}
FAIL_CLOSED_RESULT = {"risk_score": 100.0, "tier": "High", "action": "block_and_review",
                      "ip_risk": None, "takeover_risk": None, "reasons": [], "event_id": None,
                      "degraded": True}


@dataclass
class ClientStats:
    calls: int = 0
    failures: int = 0
    timeouts: int = 0
    degraded: int = 0
    breaker_open_count: int = 0

    def as_dict(self) -> dict:
        return {"calls": self.calls, "failures": self.failures, "timeouts": self.timeouts,
                "degraded": self.degraded, "breaker_open_count": self.breaker_open_count,
                "failure_rate": round(self.failures / self.calls, 4) if self.calls else 0.0}


@dataclass
class RiskClient:
    """A small, dependency-light client with a timeout, a default decision and a circuit breaker.

    base_url: where the service listens. Keep it internal - this API must not be reachable from
        the internet, and nothing it returns should ever be shown to an end user.
    api_key: the shared secret, sent as X-API-Key.
    timeout_seconds: hard ceiling on the call.
    fail_open: the decision used when the call fails. True (default) lets the login proceed.
    on_degraded: called with the exception whenever the default is used. Wire this to your metrics
        and alerting - this is the signal that the engine has stopped protecting you.
    """

    base_url: str
    api_key: str
    timeout_seconds: float = 1.0
    fail_open: bool = True
    breaker_threshold: int = 5
    breaker_reset_seconds: float = 30.0
    on_degraded: Optional[Callable[[BaseException | None], None]] = None
    stats: ClientStats = field(default_factory=ClientStats)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _consecutive_failures: int = field(default=0, repr=False)
    _breaker_until: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------ breaker
    def _breaker_is_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._breaker_until

    def _note_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def _note_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.breaker_threshold and time.monotonic() >= self._breaker_until:
                self._breaker_until = time.monotonic() + self.breaker_reset_seconds
                self.stats.breaker_open_count += 1
                log.error("risk service failed %d times in a row; not calling it for %.0fs. "
                          "Logins are proceeding on the default decision (fail_open=%s).",
                          self._consecutive_failures, self.breaker_reset_seconds, self.fail_open)

    def _default(self, exc: BaseException | None) -> dict:
        self.stats.degraded += 1
        if self.on_degraded is not None:
            try:
                self.on_degraded(exc)
            except Exception:                           # noqa: BLE001 - never let a callback break a login
                log.exception("on_degraded callback raised")
        return dict(FAIL_OPEN_RESULT if self.fail_open else FAIL_CLOSED_RESULT)

    # ------------------------------------------------------------------ the call
    def score(self, *, user_id: str, ip: str, success: bool = True, browser: str | None = None,
              os: str | None = None, device: str | None = None, rtt_ms: float | None = None,
              country: str | None = None, asn: Any = None, timestamp: str | None = None) -> dict:
        """Score one login. NEVER raises: on any failure it returns the default decision.

        `ip` must be the address your edge actually saw, not a client-supplied X-Forwarded-For that
        the user can set. `country` and `asn` are ignored when the service resolves geo itself,
        which is the configuration you want.
        """
        self.stats.calls += 1
        if self._breaker_is_open():
            return self._default(None)

        payload = {"user_id": user_id, "ip": ip, "success": success}
        for k, v in (("browser", browser), ("os", os), ("device", device), ("rtt_ms", rtt_ms),
                     ("country", country), ("asn", asn), ("timestamp", timestamp)):
            if v is not None:
                payload[k] = v
        try:
            body = self._post("/v1/score", payload)
            self._note_success()
            body.setdefault("degraded", False)
            return body
        except TimeoutError as exc:
            self.stats.timeouts += 1
            self.stats.failures += 1
            self._note_failure()
            log.warning("risk service timed out after %.0f ms", self.timeout_seconds * 1000)
            return self._default(exc)
        except Exception as exc:                        # noqa: BLE001 - a login must not die here
            self.stats.failures += 1
            self._note_failure()
            log.warning("risk service call failed: %s", exc)
            return self._default(exc)

    def confirm(self, event_id: str) -> bool:
        """Call after the user passes step-up. Returns False if it could not be recorded.

        Safe to lose: a missed confirmation only means the login does not join the user's normal
        profile, so they may be challenged again. Never block a successful login on this.
        """
        try:
            self._post(f"/v1/confirm/{event_id}", {})
            return True
        except Exception as exc:                        # noqa: BLE001
            log.warning("could not confirm %s: %s", event_id, exc)
            return False

    # ------------------------------------------------------------------ transport
    def _post(self, path: str, payload: dict) -> dict:
        """Uses urllib so this client adds no dependency to your web application."""
        import json
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            # 4xx means we sent something wrong - worth surfacing loudly in your tests, but at
            # runtime it still must not block a login.
            raise RuntimeError(f"risk service returned {exc.code}: {exc.read()[:200]!r}") from exc
        except TimeoutError:
            raise
        except OSError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise TimeoutError(str(exc)) from exc
            raise