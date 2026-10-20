"""Metrics and structured logging.

A risk engine that is quietly broken looks exactly like one that is working: requests return 200,
nothing throws, and every login is scored Low. The only way to tell the difference from outside is
to watch the *distribution* of what it decides, so these are the things worth measuring:

  * the tier mix - if Medium+ drifts away from its 2% budget, either you are under attack or the
    thresholds no longer fit your traffic
  * how often the home-country policy fires, separately from the model, so you can see what the
    rule costs in OTPs
  * latency percentiles, because your caller's timeout has to sit above p99
  * warm-up state, since a service stuck in warm-up is enforcing nothing at all
  * geo source, because `client-supplied` in production means the country is attacker-chosen

prometheus_client is optional. Without it everything here becomes a no-op and /metrics returns a
short note instead of failing, so the service runs the same in a dev shell as in a cluster.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Optional

log = logging.getLogger("riskengine")

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
    HAVE_PROMETHEUS = True
except ImportError:                                     # pragma: no cover - exercised by absence
    HAVE_PROMETHEUS = False
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"


class _NoopMetric:
    def labels(self, *a, **k): return self
    def inc(self, *a, **k): return None
    def observe(self, *a, **k): return None
    def set(self, *a, **k): return None


class Metrics:
    """Created once per process. Safe to construct twice in tests (duplicate registration is caught)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and HAVE_PROMETHEUS
        if not self.enabled:
            self.scored = self.policy = self.degraded = self.rejected = _NoopMetric()
            self.latency = self.warming = self.state_logins = self.state_users = _NoopMetric()
            return
        try:
            self.scored = Counter("risk_logins_scored_total", "Logins scored", ["tier", "model_tier", "warming"])
            self.policy = Counter("risk_policy_applied_total", "Policy rules applied", ["rule"])
            self.rejected = Counter("risk_requests_rejected_total", "Requests refused", ["reason"])
            self.degraded = Counter("risk_write_queue_inline_total", "Writes done inline because the queue was full")
            self.latency = Histogram("risk_score_latency_seconds", "POST /v1/score latency",
                                     buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))
            self.warming = Gauge("risk_warming_up", "1 while the service is log-only")
            self.state_logins = Gauge("risk_state_logins", "Logins recorded in the live state")
            self.state_users = Gauge("risk_state_users", "Accounts with a profile in the live state")
        except ValueError:
            # Already registered in this process (a second create_app in the same test session).
            self.enabled = False
            self.scored = self.policy = self.degraded = self.rejected = _NoopMetric()
            self.latency = self.warming = self.state_logins = self.state_users = _NoopMetric()

    def render(self) -> tuple[bytes, str]:
        if not self.enabled:
            return (b"# prometheus_client is not installed; `pip install prometheus_client` to export metrics\n",
                    "text/plain; charset=utf-8")
        return generate_latest(), CONTENT_TYPE_LATEST


class JsonFormatter(logging.Formatter):
    """One JSON object per line, so a log aggregator can filter on fields instead of regexes.

    Deliberately omits anything that identifies a person: no IPs, no user ids, no user agents. The
    risk log is the sensitive one - it is a record of where people log in from - so identifying
    details belong in the audit database, behind the retention and erasure endpoints, not in a log
    stream that gets shipped to a third party and kept forever.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(json_logs: Optional[bool] = None, level: Optional[str] = None) -> None:
    """Install the JSON formatter when RISK_LOG_JSON is set. Call once, at startup."""
    if json_logs is None:
        json_logs = os.getenv("RISK_LOG_JSON", "0") not in ("0", "", "false", "False")
    level = (level or os.getenv("RISK_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(level)
    if not json_logs:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)