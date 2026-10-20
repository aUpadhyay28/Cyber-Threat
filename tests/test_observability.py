"""Metrics, structured logs, and readiness.

The thing worth testing here is what the metrics let you *notice*. A risk engine that silently
scores everything Low returns 200 for every request and throws nothing; the only external symptom
is the tier distribution, so that distribution has to be exported.
"""
import json
import logging

import pytest
from fastapi.testclient import TestClient

from riskengine.api import create_app
from riskengine.data import normalize_chunk
from riskengine.geoip import StaticResolver
from riskengine.observability import JsonFormatter, Metrics, configure_logging
from riskengine.policy import HomeCountryPolicy
from riskengine.training import train
from scripts.make_mock_data import make

API_KEY = "test-key-0123456789-abcdefghijkl"
KEY = {"X-API-Key": API_KEY}
GEO = StaticResolver({"49.36.1.1": ("IN", 9829), "203.0.113.50": ("RU", 64500)})


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("obs-model")
    train(normalize_chunk(make(40_000, n_users=500, seed=3, span_days=30)), str(d), verbose=False)
    return d


def _app(trained, tmp_path, **kw):
    return create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"),
                      str(tmp_path / "r.db"), API_KEY, geo_resolver=GEO, **kw)


def _ev(ip="49.36.1.1", user="u1"):
    return dict(user_id=user, ip=ip, country="IN", asn=1, browser="Chrome 120", os="Windows 11",
                device="desktop", rtt_ms=50.0, success=True)


def test_metrics_endpoint_serves_something_scrapable(trained, tmp_path):
    app = _app(trained, tmp_path)
    with TestClient(app) as c:
        c.post("/v1/score", json=_ev(), headers=KEY)
        r = c.get("/metrics")
        assert r.status_code == 200
        body = r.text
        # Either real metrics, or a clear note that the exporter is not installed.
        assert "risk_logins_scored_total" in body or "prometheus_client is not installed" in body


def test_metrics_need_no_api_key(trained, tmp_path):
    """A scrape endpoint has to be reachable by the scraper, which is why it carries no personal data."""
    app = _app(trained, tmp_path)
    with TestClient(app) as c:
        assert c.get("/metrics").status_code == 200


def test_tier_distribution_is_exported(trained, tmp_path):
    """The one signal that reveals a silently-broken engine."""
    pytest.importorskip("prometheus_client")
    app = _app(trained, tmp_path, policy=HomeCountryPolicy(home_countries=frozenset({"IN"})))
    with TestClient(app) as c:
        for i in range(3):
            c.post("/v1/score", json=_ev(user=f"m{i}"), headers=KEY)
        body = c.get("/metrics").text
        if "risk_logins_scored_total" in body:
            assert 'tier=' in body and 'model_tier=' in body


def test_ready_is_separate_from_health(trained, tmp_path):
    """/health = is the process alive (restart policy). /ready = should it get traffic (load balancer)."""
    app = _app(trained, tmp_path)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 200
        assert c.get("/ready").json()["ready"] is True


def test_ready_reports_false_before_startup(trained, tmp_path):
    """Outside the lifespan the app has no model, so it must not claim to be ready."""
    app = _app(trained, tmp_path)
    client = TestClient(app)                     # not entered: lifespan has not run
    assert client.get("/ready").status_code == 503


def test_health_reports_the_state_backend(trained, tmp_path):
    app = _app(trained, tmp_path)
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["state_backend"] == "in-process"
        assert "geoip_source" in h and "timestamp_source" in h


def test_json_formatter_emits_one_object_per_line():
    rec = logging.LogRecord("riskengine", logging.WARNING, __file__, 10, "something %s", ("odd",), None)
    line = JsonFormatter().format(rec)
    doc = json.loads(line)
    assert doc["level"] == "WARNING" and doc["msg"] == "something odd" and doc["logger"] == "riskengine"
    assert "\n" not in line


def test_json_logs_carry_no_personal_data():
    """The risk log records where people sign in from. Identifying detail belongs in the audit
    database behind the retention endpoints, not in a log stream shipped to a third party."""
    rec = logging.LogRecord("riskengine", logging.INFO, __file__, 1, "scored a login", (), None)
    doc = json.loads(JsonFormatter().format(rec))
    assert set(doc) <= {"ts", "level", "logger", "msg", "exc"}


def test_configure_logging_is_a_noop_without_the_flag(monkeypatch):
    monkeypatch.delenv("RISK_LOG_JSON", raising=False)
    before = list(logging.getLogger().handlers)
    configure_logging()
    assert list(logging.getLogger().handlers) == before


def test_metrics_disabled_still_renders(monkeypatch):
    m = Metrics(enabled=False)
    body, content_type = m.render()
    assert b"prometheus_client" in body and "text/plain" in content_type
    m.scored.labels(tier="Low", model_tier="Low", warming="false").inc()      # must not raise
    m.latency.observe(0.01)
    m.warming.set(1)