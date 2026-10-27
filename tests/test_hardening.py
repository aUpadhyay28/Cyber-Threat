"""Access control and hardening of the scoring service.

Each test pins down a way the service could be abused by someone who got hold of *one* secret, or
who can reach it on the network without any: a leaked login-backend key used to erase accounts, a
key typed into a log, interactive docs mapping the API, a multi-megabyte body, a console that would
run injected script.
"""
import logging
import os
import stat
import sys

import pytest
from fastapi.testclient import TestClient

from riskengine.api import create_app
from riskengine.client import RiskClient
from riskengine.data import normalize_chunk
from riskengine.geoip import StaticResolver
from riskengine.persist import write_digest
from riskengine.training import train
from scripts.make_mock_data import make

SERVICE = "svc-key-0123456789-abcdefghijklmn"
ANALYST = "analyst-key-0123456789-abcdefghij"
ADMIN = "admin-key-0123456789-abcdefghijklm"
HASH = "hash-key-0123456789-abcdefghijklmn"
GEO = StaticResolver({"49.36.10.2": ("IN", 55836)})
LOGIN = {"user_id": "alice@example.com", "ip": "49.36.10.2", "browser": "Chrome 120",
         "os": "Android 14", "device": "mobile", "success": True}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("hard-model")
    train(normalize_chunk(make(30_000, n_users=400, seed=5, span_days=30)), str(d), verbose=False)
    write_digest(d / "latest.joblib")
    return d


@pytest.fixture
def env(monkeypatch):
    for v in ("RISK_ENV", "RISK_API_KEY", "RISK_ANALYST_KEY", "RISK_ADMIN_KEY", "RISK_HASH_KEY",
              "RISK_METRICS_TOKEN", "RISK_ALLOW_SHORT_KEY", "RISK_REDIS_URL", "RISK_API_KEY_FILE"):
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def _app(trained, tmp_path, **kw):
    kw.setdefault("geo_resolver", GEO)
    return create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), **kw)


def _roles(trained, tmp_path):
    return _app(trained, tmp_path, api_key=SERVICE, analyst_key=ANALYST, admin_key=ADMIN)


def _h(key):
    return {"X-API-Key": key}


# --------------------------------------------------------------------------- roles
def test_login_backend_key_cannot_review_or_administer(env, trained, tmp_path):
    with TestClient(_roles(trained, tmp_path)) as c:
        ev = c.post("/v1/score", headers=_h(SERVICE), json=LOGIN)
        assert ev.status_code == 200
        eid = ev.json()["event_id"]
        assert c.get("/v1/queue", headers=_h(SERVICE)).status_code == 403
        assert c.get(f"/v1/event/{eid}", headers=_h(SERVICE)).status_code == 403
        assert c.post("/v1/verdict", headers=_h(SERVICE),
                      json={"event_id": eid, "verdict": "benign"}).status_code == 403
        assert c.post("/v1/admin/forget-user", headers=_h(SERVICE), json={"user_id": "x"}).status_code == 403
        assert c.post("/v1/admin/reload-model", headers=_h(SERVICE)).status_code == 403


def test_analyst_key_reviews_but_cannot_score_or_administer(env, trained, tmp_path):
    with TestClient(_roles(trained, tmp_path)) as c:
        eid = c.post("/v1/score", headers=_h(SERVICE), json=LOGIN).json()["event_id"]
        assert c.get("/v1/queue", headers=_h(ANALYST)).status_code == 200
        assert c.get(f"/v1/event/{eid}", headers=_h(ANALYST)).status_code == 200
        assert c.get("/v1/stats", headers=_h(ANALYST)).status_code == 200
        assert c.post("/v1/score", headers=_h(ANALYST), json=LOGIN).status_code == 403
        assert c.post("/v1/admin/purge-raw", headers=_h(ANALYST)).status_code == 403


def test_admin_key_administers_and_may_review(env, trained, tmp_path):
    with TestClient(_roles(trained, tmp_path)) as c:
        assert c.post("/v1/admin/forget-user", headers=_h(ADMIN), json={"user_id": "nobody"}).status_code == 200
        assert c.get("/v1/queue", headers=_h(ADMIN)).status_code == 200
        assert c.post("/v1/score", headers=_h(ADMIN), json=LOGIN).status_code == 403


def test_unknown_key_is_401(env, trained, tmp_path):
    with TestClient(_roles(trained, tmp_path)) as c:
        assert c.post("/v1/score", headers=_h("x" * 40), json=LOGIN).status_code == 401
        assert c.get("/v1/queue").status_code == 401


def test_single_key_still_works_in_development(env, trained, tmp_path):
    """Backwards compatible: with only RISK_API_KEY set, that key opens every route (dev only)."""
    with TestClient(_app(trained, tmp_path, api_key=SERVICE)) as c:
        assert c.post("/v1/score", headers=_h(SERVICE), json=LOGIN).status_code == 200
        assert c.get("/v1/queue", headers=_h(SERVICE)).status_code == 200
        assert c.post("/v1/admin/prune-state", headers=_h(SERVICE)).status_code == 200


def test_keys_rotate_without_downtime(env, trained, tmp_path):
    old, new = SERVICE, "svc-key-NEW-0123456789-abcdefghijk"
    with TestClient(_app(trained, tmp_path, api_key=f"{old}, {new}", analyst_key=ANALYST,
                         admin_key=ADMIN)) as c:
        assert c.post("/v1/score", headers=_h(old), json=LOGIN).status_code == 200
        assert c.post("/v1/score", headers=_h(new), json=LOGIN).status_code == 200


def test_key_can_come_from_a_file(env, trained, tmp_path):
    f = tmp_path / "api_key"
    f.write_text(SERVICE + "\n")
    env.setenv("RISK_API_KEY_FILE", str(f))
    with TestClient(_app(trained, tmp_path)) as c:
        assert c.post("/v1/score", headers=_h(SERVICE), json=LOGIN).status_code == 200


def test_short_key_in_a_rotation_list_is_refused(env, trained, tmp_path):
    with pytest.raises(RuntimeError, match="at least"):
        _app(trained, tmp_path, api_key=f"{SERVICE},short")


# --------------------------------------------------------------------------- production mode
def _prod(env):
    env.setenv("RISK_ENV", "production")
    env.setenv("RISK_HASH_KEY", HASH)


def test_production_requires_three_distinct_keys(env, trained, tmp_path):
    _prod(env)
    with pytest.raises(RuntimeError, match="RISK_ANALYST_KEY is not set"):
        _app(trained, tmp_path, api_key=SERVICE)
    with pytest.raises(RuntimeError, match="must all be different"):
        _app(trained, tmp_path, api_key=SERVICE, analyst_key=SERVICE, admin_key=ADMIN)


def test_production_requires_a_redis_password(env, trained, tmp_path):
    _prod(env)
    env.setenv("RISK_REDIS_URL", "redis://redis:6379/0")
    with pytest.raises(RuntimeError, match="no password"):
        _roles(trained, tmp_path)


def test_production_hides_docs_and_detailed_health(env, trained, tmp_path):
    _prod(env)
    with TestClient(_roles(trained, tmp_path)) as c:
        assert c.get("/docs").status_code == 404
        assert c.get("/openapi.json").status_code == 404
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/health", headers=_h(SERVICE)).json() == {"status": "ok"}
        assert "model_sha256" in c.get("/health", headers=_h(ANALYST)).json()


def test_development_keeps_docs(env, trained, tmp_path):
    with TestClient(_roles(trained, tmp_path)) as c:
        assert c.get("/openapi.json").status_code == 200


# --------------------------------------------------------------------------- transport
def test_oversized_body_is_refused_before_parsing(env, trained, tmp_path):
    with TestClient(_roles(trained, tmp_path)) as c:
        big = {**LOGIN, "browser": "x" * 200, "pad": "y" * 100_000}
        assert c.post("/v1/score", headers=_h(SERVICE), json=big).status_code == 413


def test_security_headers_and_console_csp(env, trained, tmp_path):
    with TestClient(_roles(trained, tmp_path)) as c:
        api = c.get("/health").headers
        assert api["content-security-policy"].startswith("default-src 'none'")
        assert api["x-frame-options"] == "DENY" and api["x-content-type-options"] == "nosniff"
        assert "server" not in {k.lower() for k in api.keys()}
        csp = c.get("/console").headers["content-security-policy"]
        script_src = [d for d in csp.split(";") if d.strip().startswith("script-src")][0]
        assert "'sha256-" in script_src and "unsafe-inline" not in script_src
        assert "connect-src 'self'" in csp and "frame-ancestors 'none'" in csp


def test_metrics_token(env, trained, tmp_path):
    env.setenv("RISK_METRICS_TOKEN", "metrics-token-0123456789-abcdef")
    with TestClient(_roles(trained, tmp_path)) as c:
        assert c.get("/metrics").status_code == 401
        ok = c.get("/metrics", headers={"Authorization": "Bearer metrics-token-0123456789-abcdef"})
        assert ok.status_code == 200


# --------------------------------------------------------------------------- data at rest
def test_logs_never_carry_the_raw_user_id(env, trained, tmp_path, caplog):
    with TestClient(_roles(trained, tmp_path)) as c, caplog.at_level(logging.INFO, logger="riskengine"):
        c.post("/v1/score", headers=_h(SERVICE), json=LOGIN)
        c.post("/v1/admin/forget-user", headers=_h(ADMIN), json={"user_id": LOGIN["user_id"]})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "erased account" in text
    assert LOGIN["user_id"] not in text and LOGIN["ip"] not in text


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX permissions")
def test_audit_db_and_state_are_owner_only(env, trained, tmp_path):
    with TestClient(_roles(trained, tmp_path)) as c:
        c.post("/v1/score", headers=_h(SERVICE), json=LOGIN)
    for f in ("r.db", "s.json.gz"):
        mode = stat.S_IMODE(os.stat(tmp_path / f).st_mode)
        assert mode & 0o077 == 0, f"{f} is {oct(mode)}"


# --------------------------------------------------------------------------- client
def test_client_refuses_non_http_urls_and_hides_its_key():
    with pytest.raises(ValueError):
        RiskClient("file:///etc/passwd", SERVICE)
    c = RiskClient("http://127.0.0.1:8000", SERVICE)
    assert SERVICE not in repr(c) and SERVICE not in str(c)
