"""Security properties of the scoring service and of the model itself.

Each test here corresponds to a way the system could be made to produce the WRONG decision, rather
than to crash. A risk engine that can be talked into scoring an attack as Low has failed completely
while looking perfectly healthy, so these are the tests that matter most.
"""
import gzip
import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from riskengine.api import create_app
from riskengine.data import normalize_chunk, event_to_row
from riskengine.features import FeatureState, TimestampOutOfRange, FEATURES
from riskengine.modeling import RiskModel
from riskengine.persist import load_state, save_state, verify_model, write_digest
from riskengine.store import Store
from riskengine.training import train
from scripts.make_mock_data import make

API_KEY = "test-key-0123456789-abcdefghijkl"
KEY = {"X-API-Key": API_KEY}
HOUR_MS = 3_600_000


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("sec-model")
    train(normalize_chunk(make(40_000, n_users=500, seed=3, span_days=30)), str(d), verbose=False)
    return d


def _row(user, ip, ts_ms, ok=False, asn=7):
    return {"user": user, "ip": ip, "asn": asn, "country": 1, "browser": 2, "os": 3, "device": 4,
            "ts_ms": ts_ms, "success": ok, "rtt": 50.0}


# ----------------------------------------------------------------- timestamp / window poisoning
def test_future_timestamp_cannot_wipe_the_window():
    """THE ATTACK: the 1 h / 24 h counts expire relative to the newest timestamp seen, so a single
    future-dated login would evict an attacking IP's entire recent history and let it score as a
    quiet network. Before the guard, 60 failed logins became 1."""
    base = 1_700_000_000_000
    st = FeatureState()
    for i in range(60):
        st.update(_row(1000 + i, 42, base + i * 1000), trusted=False, counters=True)
    before = st.features(_row(999, 42, base + 60_000))
    assert round(math.expm1(before["ip_fail_24h_log"])) == 60

    with pytest.raises(TimestampOutOfRange):
        st.update(_row(5, 42, base + 10 * 365 * 24 * HOUR_MS), trusted=False, counters=True)

    after = st.features(_row(999, 42, base + 61_000))
    assert round(math.expm1(after["ip_fail_24h_log"])) == 60, "the IP's recent failures must survive"
    assert len(st.win_ip[42]) == 60


def test_normal_gaps_are_still_accepted():
    """The guard must not reject ordinary traffic: a quiet few days is legitimate."""
    base = 1_700_000_000_000
    st = FeatureState()
    st.update(_row(1, 42, base), trusted=True, counters=True)
    st.update(_row(1, 42, base + 3 * 24 * HOUR_MS), trusted=True, counters=True)   # 3-day gap: fine
    assert st.max_ts == base + 3 * 24 * HOUR_MS


def test_offline_replay_can_opt_out_of_the_guard():
    """import_history replays a file the operator controls, where long gaps are real history."""
    base = 1_700_000_000_000
    st = FeatureState(max_forward_jump_ms=None)
    st.update(_row(1, 42, base), trusted=True, counters=True)
    st.update(_row(1, 42, base + 400 * 24 * HOUR_MS), trusted=True, counters=True)
    assert st.glob_total == 2


def test_api_rejects_a_future_timestamp_in_replay_mode(trained, tmp_path):
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"),
                     API_KEY, timestamp_source="client")
    with TestClient(app) as c:
        ev = dict(user_id="u", ip="10.0.0.1", country="NO", asn=1, browser="b", os="o", device="d",
                  rtt_ms=10.0, success=True, timestamp="2999-01-01T00:00:00Z")
        assert c.post("/v1/score", json=ev, headers=KEY).status_code == 422


def test_server_mode_ignores_the_callers_timestamp(trained, tmp_path):
    """The secure default: the caller's claimed time is recorded but never used for scoring."""
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"),
                     API_KEY)                                        # timestamp_source defaults to "server"
    with TestClient(app) as c:
        assert c.get("/health").json()["timestamp_source"] == "server"
        ev = dict(user_id="u", ip="10.0.0.1", country="NO", asn=1, browser="b", os="o", device="d",
                  rtt_ms=10.0, success=True, timestamp="2999-01-01T00:00:00Z")
        r = c.post("/v1/score", json=ev, headers=KEY)
        assert r.status_code == 200                                  # accepted, but not believed
        row = json.loads(Store(str(tmp_path / "r.db")).get_event(r.json()["event_id"])["row_json"])
        assert row["ts_ms"] < 4_000_000_000_000, "the year-2999 claim must not reach the feature state"


# ----------------------------------------------------------------- no pickle in the load path
def test_state_round_trips_without_pickle(tmp_path):
    raw = make(2000, n_users=60, seed=4)
    prep = normalize_chunk(raw).sort_values("ts_ms", kind="stable")
    st = FeatureState()
    for r in prep.to_dict("records"):
        st.update(r, trusted=bool(r["success"]), counters=True)
    p = tmp_path / "s.json.gz"
    save_state(st, p)
    back = load_state(p)
    probe = prep.to_dict("records")[-1]
    a, b = st.features(probe), back.features(probe)
    for k in FEATURES:
        assert (a[k] == b[k]) or (np.isnan(a[k]) and np.isnan(b[k])), k
    assert back.glob_total == st.glob_total and back.max_ts == st.max_ts
    assert len(back.users) == len(st.users)


def test_a_pickled_state_file_is_refused(tmp_path):
    """Loading a pickle executes code inside it. The service must not do that, and must say so."""
    import pickle
    p = tmp_path / "state.pkl"
    with open(p, "wb") as fh:
        pickle.dump(FeatureState(), fh)
    with pytest.raises(ValueError, match="migrate_state_pickle"):
        load_state(p)


def test_state_file_is_plain_data(tmp_path):
    p = tmp_path / "s.json.gz"
    save_state(FeatureState(), p)
    with gzip.open(p, "rb") as fh:
        doc = json.loads(fh.read().decode())            # parses as JSON => no executable payload
    assert doc["format"] == "riskengine-feature-state"


# ----------------------------------------------------------------- model integrity
def test_model_digest_mismatch_is_refused(trained, tmp_path):
    import shutil
    m = tmp_path / "latest.joblib"
    shutil.copy(trained / "latest.joblib", m)
    good = write_digest(m)
    assert verify_model(m) == good                      # pinned and matching
    with open(m, "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ValueError, match="pinned digest"):
        verify_model(m)


def test_service_refuses_to_start_on_a_tampered_model(trained, tmp_path):
    import shutil
    m = tmp_path / "latest.joblib"
    shutil.copy(trained / "latest.joblib", m)
    digest = write_digest(m)
    with open(m, "ab") as fh:
        fh.write(b"x")
    app = create_app(str(m), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), API_KEY, model_sha256=digest)
    with pytest.raises(ValueError, match="pinned digest"):
        with TestClient(app):
            pass


# ----------------------------------------------------------------- feature contract
def test_model_with_a_different_feature_list_is_refused(trained, tmp_path, monkeypatch):
    """heads() finds each feature by position. A model fitted on a different feature list would be
    read column-by-column off by one and score confidently on the wrong numbers."""
    import joblib
    d = joblib.load(trained / "latest.joblib")
    d["meta"] = {**d["meta"], "features": list(FEATURES)[:-1]}      # pretend it predates one feature
    p = tmp_path / "stale.joblib"
    joblib.dump(d, p)
    with pytest.raises(ValueError, match="model/feature mismatch"):
        RiskModel.load(p)


def test_model_without_a_feature_list_is_refused(trained, tmp_path):
    import joblib
    d = joblib.load(trained / "latest.joblib")
    d["meta"] = {k: v for k, v in d["meta"].items() if k != "features"}
    p = tmp_path / "nofeat.joblib"
    joblib.dump(d, p)
    with pytest.raises(ValueError, match="no feature list"):
        RiskModel.load(p)


# ----------------------------------------------------------------- API surface
def test_short_api_key_is_refused(trained, tmp_path):
    with pytest.raises(RuntimeError, match="at least"):
        create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), "short")


def test_malformed_ip_is_rejected(trained, tmp_path):
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), API_KEY)
    with TestClient(app) as c:
        ev = dict(user_id="u", ip="not-an-ip", country="NO", asn=1, browser="b", os="o", device="d",
                  rtt_ms=10.0, success=True)
        assert c.post("/v1/score", json=ev, headers=KEY).status_code == 422
        ev["ip"] = "10.0.0.1; DROP TABLE events"
        assert c.post("/v1/score", json=ev, headers=KEY).status_code == 422


def test_oversized_identifiers_are_rejected(trained, tmp_path):
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), API_KEY)
    with TestClient(app) as c:
        ev = dict(user_id="u" * 5000, ip="10.0.0.1", country="NO", asn=1, browser="b", os="o",
                  device="d", rtt_ms=10.0, success=True)
        assert c.post("/v1/score", json=ev, headers=KEY).status_code == 422


def test_a_failed_login_cannot_be_confirmed_into_the_profile(trained, tmp_path):
    """Confirmation exists for 'the user passed the OTP'. A failed attempt has nothing to confirm,
    and allowing it would let a caller normalise an attacker's country and device by replay.

    The event is written straight to the log with a chosen tier so the test exercises the rule
    rather than whatever tier a cold model happens to give a failed login.
    """
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), API_KEY)
    raw = {"Login Timestamp": "2020-06-01T10:00:00Z", "User ID": "failer", "IP Address": "10.0.0.9",
           "Country": "RU", "ASN": 64500, "Browser Name and Version": "Python-urllib 3",
           "OS Name and Version": "Linux", "Device Type": "unknown", "Round-Trip Time [ms]": 10.0,
           "Login Successful": False}
    Store(str(tmp_path / "r.db")).add_event(
        "failed-1", "failer", raw, event_to_row(raw), {k: 0.0 for k in FEATURES},
        {"risk_score": 10.0, "tier": "Low", "action": "allow"}, [], 0)
    with TestClient(app) as c:
        r = c.post("/v1/confirm/failed-1", headers=KEY)
        assert r.status_code == 409 and "failed login" in r.json()["detail"]


def test_security_headers_are_set(trained, tmp_path):
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), API_KEY)
    with TestClient(app) as c:
        h = c.get("/health").headers
        assert h["X-Content-Type-Options"] == "nosniff" and h["X-Frame-Options"] == "DENY"
        assert h["Cache-Control"] == "no-store"


def test_every_decision_records_the_model_version(trained, tmp_path):
    """Without this you cannot answer 'which model blocked this login?' after a single retrain."""
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), API_KEY)
    with TestClient(app) as c:
        ev = dict(user_id="u", ip="10.0.0.1", country="NO", asn=1, browser="b", os="o", device="d",
                  rtt_ms=10.0, success=True)
        r = c.post("/v1/score", json=ev, headers=KEY).json()
        stored = Store(str(tmp_path / "r.db")).get_event(r["event_id"])
        assert stored["model_version"] == r["model_version"] and stored["model_version"]


def test_rate_limiter_blocks_a_flood(trained, tmp_path, monkeypatch):
    monkeypatch.setattr("riskengine.api.RATE_LIMIT_PER_MIN", 5)
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), API_KEY)
    with TestClient(app) as c:
        ev = dict(user_id="u", ip="10.0.0.1", country="NO", asn=1, browser="b", os="o", device="d",
                  rtt_ms=10.0, success=True)
        codes = [c.post("/v1/score", json=ev, headers=KEY).status_code for _ in range(12)]
        assert 429 in codes, codes


# ----------------------------------------------------------------- privacy / erasure
def test_forget_user_removes_profile_and_events(trained, tmp_path):
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), API_KEY)
    with TestClient(app) as c:
        ev = dict(user_id="erase-me", ip="10.0.0.5", country="NO", asn=11, browser="Chrome 80",
                  os="Windows 10", device="desktop", rtt_ms=10.0, success=True)
        for _ in range(4):
            c.post("/v1/score", json=ev, headers=KEY)
        store = Store(str(tmp_path / "r.db"))
        assert len(store.user_category_values("erase-me")["country"]) == 1

        out = c.post("/v1/admin/forget-user", json={"user_id": "erase-me"}, headers=KEY).json()
        assert out["database"]["events_deleted"] == 4
        assert out["state"]["category_tables_complete"] is True
        assert store.user_category_values("erase-me")["country"] == []

        uid = event_to_row({"Login Timestamp": "2020-01-01T00:00:00Z", "User ID": "erase-me",
                            "IP Address": "10.0.0.5", "Country": "NO", "ASN": 11,
                            "Browser Name and Version": "Chrome 80", "OS Name and Version": "Windows 10",
                            "Device Type": "desktop", "Round-Trip Time [ms]": 10.0,
                            "Login Successful": True})["user"]
        state = load_state(tmp_path / "s.json.gz")
        assert uid not in state.users
        assert all(uid not in members for members in state.ip_users.values())
        assert all(uid not in members for members in state.asn_users.values())


def test_forget_user_clears_the_per_category_tables():
    """Deterministic counterpart to the API test above: the per-category tables are keyed by a
    non-invertible mix(user, value), so erasure only works when the account's values are supplied."""
    from riskengine.features import mix_py
    base = 1_700_000_000_000
    st = FeatureState()
    row = _row(user=1234, ip=99, ts_ms=base, ok=True)
    for i in range(3):
        st.update({**row, "ts_ms": base + i * 60_000}, trusted=True, counters=True)
    assert mix_py(1234, row["country"]) in st.seen["country"]
    assert st.users[1234][0] == 3

    values = {c: [row[c]] for c in ("ip", "country", "asn", "browser", "os", "device")}
    out = st.forget_user(1234, values)

    assert out["profile_removed"] and out["category_entries_removed"] == 6
    assert 1234 not in st.users
    assert mix_py(1234, row["country"]) not in st.seen["country"]
    assert all(1234 not in m for m in st.ip_users.values())

    # Without the values, the tables cannot be reached and the result says so rather than lying.
    st2 = FeatureState()
    st2.update(row, trusted=True, counters=True)
    out2 = st2.forget_user(1234)
    assert out2["category_tables_complete"] is False
    assert mix_py(1234, row["country"]) in st2.seen["country"]


def test_raw_payload_purge_keeps_the_audit_trail(trained, tmp_path):
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"), API_KEY)
    with TestClient(app) as c:
        ev = dict(user_id="keep", ip="10.0.0.7", country="NO", asn=11, browser="b", os="o",
                  device="d", rtt_ms=10.0, success=True)
        eid = c.post("/v1/score", json=ev, headers=KEY).json()["event_id"]
        # days=1 leaves today's event alone; the mechanism is what is under test.
        assert c.post("/v1/admin/purge-raw?days=1", headers=KEY).json()["events_purged"] == 0
        store = Store(str(tmp_path / "r.db"))
        assert json.loads(store.get_event(eid)["raw_json"])["IP Address"] == "10.0.0.7"
        assert store.purge_raw_older_than(0) == 1                       # now past the retention window
        ev_row = store.get_event(eid)
        assert json.loads(ev_row["raw_json"]) == {"purged": True}
        assert ev_row["risk"] is not None and ev_row["features_json"]    # audit trail and features survive


def test_prune_bounds_state_growth():
    base = 1_700_000_000_000
    st = FeatureState(max_forward_jump_ms=None)        # the 200-day jump below is deliberate
    for i in range(50):
        st.update(_row(i, 1000 + i, base + i * 1000), trusted=False, counters=True)
    assert len(st.win_ip) == 50
    st.update(_row(999, 7, base + 200 * 24 * HOUR_MS), trusted=False, counters=True)
    removed = st.prune(retention_ms=30 * 24 * HOUR_MS)
    assert removed["ips"] == 50 and len(st.win_ip) == 1
    assert len(st.ip_users) == 1
