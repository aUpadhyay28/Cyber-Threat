"""Loading the service's secrets: where they come from, how many there are, who may use which.

Three keys, three roles
-----------------------
One shared key used to unlock everything: the login backend that only needs to score logins could
also read every queued login (raw IPs and user agents), record analyst verdicts that retrain the
model, erase accounts and swap the model. A leak of that one key - from a web server's environment,
a log line, a crash dump - handed all of it over.

  RISK_API_KEY      the login backend: /v1/score and /v1/confirm only
  RISK_ANALYST_KEY  the review console: /v1/queue, /v1/event, /v1/verdict, /v1/stats
  RISK_ADMIN_KEY    operators: /v1/admin/* (model reload, erasure, retention) - and analyst routes

In development an unset analyst or admin key falls back to RISK_API_KEY so the demo still runs with
one secret. RISK_ENV=production refuses to start unless all three are set and all different.

Where a key comes from
----------------------
``RISK_API_KEY=...`` in the environment works, but environment variables leak: they are inherited
by child processes, shown by ``docker inspect`` and often dumped by error reporters. Each key can
instead be read from a file with ``RISK_API_KEY_FILE=/run/secrets/risk_api_key`` - the way Docker
and Kubernetes secrets are mounted. The file wins if both are set.

Rotation
--------
Each variable may hold several keys separated by commas. All of them are accepted, so a key can be
rotated with no downtime: add the new one, move callers over, remove the old one.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

MIN_KEY_LEN = 24
ROLES = ("service", "analyst", "admin")
ENV_FOR_ROLE = {"service": "RISK_API_KEY", "analyst": "RISK_ANALYST_KEY", "admin": "RISK_ADMIN_KEY"}


def read_secret(name: str) -> str:
    """Value of ``name``, preferring the file named by ``name_FILE`` when that is set."""
    file_var = os.getenv(f"{name}_FILE")
    if file_var:
        p = Path(file_var)
        if not p.is_file():
            raise RuntimeError(f"{name}_FILE points at {file_var!r}, which is not a readable file")
        return p.read_text(encoding="utf-8").strip()
    return os.getenv(name, "").strip()


def parse_keys(value: str | None, name: str, allow_short: bool = False) -> list[str]:
    keys = [k.strip() for k in (value or "").split(",") if k.strip()]
    for k in keys:
        if len(k) < MIN_KEY_LEN and not allow_short:
            raise RuntimeError(
                f"{name} contains a key of {len(k)} characters; use at least {MIN_KEY_LEN} random ones "
                f"(e.g. `python -c \"import secrets;print(secrets.token_urlsafe(32))\"`). "
                f"Set RISK_ALLOW_SHORT_KEY=1 only for tests.")
    return keys


class KeyRing:
    """Which keys each role accepts. Comparison is constant-time and checks every key, so the time
    taken does not reveal which key, or which role, a guess came close to."""

    def __init__(self, service: list[str], analyst: list[str], admin: list[str],
                 analyst_fallback: bool = False, admin_fallback: bool = False):
        self.keys = {"service": list(service), "analyst": list(analyst), "admin": list(admin)}
        self.analyst_fallback, self.admin_fallback = analyst_fallback, admin_fallback

    @classmethod
    def from_env(cls, api_key: str | None = None, analyst_key: str | None = None,
                 admin_key: str | None = None) -> "KeyRing":
        short_ok = bool(os.getenv("RISK_ALLOW_SHORT_KEY"))
        service = parse_keys(api_key if api_key is not None else read_secret("RISK_API_KEY"),
                             "RISK_API_KEY", short_ok)
        if not service:
            raise RuntimeError("Set RISK_API_KEY (or RISK_API_KEY_FILE): the secret your login backend "
                               "sends in the X-API-Key header")
        analyst = parse_keys(analyst_key if analyst_key is not None else read_secret("RISK_ANALYST_KEY"),
                             "RISK_ANALYST_KEY", short_ok)
        admin = parse_keys(admin_key if admin_key is not None else read_secret("RISK_ADMIN_KEY"),
                           "RISK_ADMIN_KEY", short_ok)
        return cls(service, analyst or service, admin or service,
                   analyst_fallback=not analyst, admin_fallback=not admin)

    def role_of(self, presented: str) -> set[str]:
        """Every role the presented key grants (empty set if none). Checks all keys, always."""
        granted: set[str] = set()
        presented_b = (presented or "").encode()
        for role, keys in self.keys.items():
            for k in keys:
                if hmac.compare_digest(presented_b, k.encode()):
                    granted.add(role)
        return granted

    def production_problems(self) -> list[str]:
        out = []
        if self.analyst_fallback:
            out.append("RISK_ANALYST_KEY is not set: the login backend's key would also open the review "
                       "queue and record verdicts")
        if self.admin_fallback:
            out.append("RISK_ADMIN_KEY is not set: the login backend's key would also erase accounts and "
                       "reload the model")
        if not (self.analyst_fallback or self.admin_fallback):
            s, a, d = (set(self.keys[r]) for r in ROLES)
            if (s & a) or (s & d) or (a & d):
                out.append("RISK_API_KEY, RISK_ANALYST_KEY and RISK_ADMIN_KEY must all be different")
        return out
