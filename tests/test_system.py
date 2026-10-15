import pandas as pd, pytest
from fastapi.testclient import TestClient
from riskengine.data import normalize_chunk
from riskengine.training import train
from riskengine.store import Store
from riskengine.api import create_app
from riskengine.modeling import RiskModel
from riskengine.features import FEATURES, build_features
from scripts.make_mock_data import make

KEY = {"X-API-Key": "test-key"}

@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    train(normalize_chunk(make(40_000, n_users=500, seed=3)), str(d), verbose=False)
    return d

@pytest.fixture()
def client(trained, tmp_path):
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.pkl"), str(tmp_path / "r.db"), "test-key")
    with TestClient(app) as c:
        yield c

def ev(user="u1", ip="10.0.0.1", country="NO", t="2020-06-01T10:00:00Z", **kw):
    d = dict(user_id=user, timestamp=t, ip=ip, country=country, asn=2119, browser="Chrome 80",
             os="Windows 10", device="desktop", rtt_ms=100.0, success=True); d.update(kw); return d

def test_training_outputs_valid_model(trained):
    m = RiskModel.load(trained / "latest.joblib")
    assert 0 <= m.medium_thr <= m.high_thr <= 100

def test_score_independent_of_batch(trained):
    """The old demo scored a lone event as 0.0. Scores must not depend on what else is in the batch."""
    m = RiskModel.load(trained / "latest.joblib")
    X = build_features(normalize_chunk(make(2000, n_users=100, seed=5))).to_numpy()
    assert abs(m.risk(X)[10] - m.risk(X[10:11])[0]) < 1e-9

def test_api_requires_key(client):
    assert client.post("/v1/score", json=ev()).status_code == 401
    assert client.post("/v1/score", json=ev(), headers={"X-API-Key": "wrong"}).status_code == 401

def test_score_response(client):
    r = client.post("/v1/score", json=ev(), headers=KEY).json()
    assert r["tier"] in ("Low", "Medium", "High") and 0 <= r["risk_score"] <= 100
    assert r["action"] in ("allow", "step_up_auth", "block_and_review") and r["event_id"]

def test_known_user_vs_stuffing_ip(client):
    for i in range(8):                                   # build a normal profile
        client.post("/v1/score", json=ev(t=f"2020-06-0{i+1}T10:00:00Z"), headers=KEY)
    normal = client.post("/v1/score", json=ev(t="2020-06-10T10:00:00Z"), headers=KEY).json()
    for i in range(30):                                  # one IP hammering many accounts from abroad
        client.post("/v1/score", json=ev(user=f"v{i}", ip="203.0.113.9", country="RU", asn=64500,
                    browser="Python-urllib 3", os="Linux", device="unknown", success=False,
                    t="2020-06-10T03:00:00Z"), headers=KEY)
    atk = client.post("/v1/score", json=ev(user="victim", ip="203.0.113.9", country="RU", asn=64500,
                      browser="Python-urllib 3", os="Linux", device="unknown", t="2020-06-10T03:05:00Z"), headers=KEY).json()
    assert atk["risk_score"] > normal["risk_score"] and atk["tier"] in ("Medium", "High")

def test_learning_gate(client, tmp_path):
    ids = [client.post("/v1/score", json=ev(user=f"g{i}", t=f"2020-06-01T1{i}:00:00Z"), headers=KEY).json()["event_id"]
           for i in range(4)]
    for eid, v in zip(ids[:3], ["benign", "confirmed_malicious", "suspicious"]):
        assert client.post("/v1/verdict", json={"event_id": eid, "verdict": v, "analyst": "a"}, headers=KEY).status_code == 200
    assert client.post("/v1/verdict", json={"event_id": ids[0], "verdict": "bogus"}, headers=KEY).status_code == 422
    assert client.post("/v1/verdict", json={"event_id": "nope", "verdict": "benign"}, headers=KEY).status_code == 404
    rows = Store(str(tmp_path / "r.db")).training_rows()
    assert sorted(rows["label"].tolist()) == [0, 1]       # benign + confirmed only; suspicious & unreviewed excluded
    assert list(rows.columns) == FEATURES + ["label"]

def _inject(tmp_path, tier, trusted=0, eid="inj1"):
    """Put an event with a chosen tier straight into the DB (the model decides tiers in real use)."""
    from riskengine.data import event_to_row
    raw = {"Login Timestamp": "2020-06-01T10:00:00Z", "User ID": "inj", "IP Address": "1.1.1.1", "Country": "NO",
           "ASN": 1, "Browser Name and Version": "b", "OS Name and Version": "o", "Device Type": "d",
           "Round-Trip Time [ms]": 1.0, "Login Successful": True}
    Store(str(tmp_path / "r.db")).add_event(eid, "inj", raw, event_to_row(raw), {k: 0.0 for k in FEATURES},
                                            {"risk_score": 99.0, "tier": tier, "action": "x"}, [], trusted)
    return eid

def test_high_risk_cannot_be_self_confirmed(client, tmp_path):
    eid = _inject(tmp_path, "High")
    assert client.post(f"/v1/confirm/{eid}", headers=KEY).status_code == 409

def test_step_up_confirmation_counts_once(client, tmp_path):
    eid = _inject(tmp_path, "Medium")
    assert client.post(f"/v1/confirm/{eid}", headers=KEY).json()["added_to_profile"] is True
    assert client.post(f"/v1/confirm/{eid}", headers=KEY).json()["added_to_profile"] is False   # no double counting
    assert client.post("/v1/confirm/missing", headers=KEY).status_code == 404

def test_queue_shows_unreviewed_medium_high_only(client, tmp_path):
    a, b = _inject(tmp_path, "High", eid="q1"), _inject(tmp_path, "Medium", eid="q2")
    _inject(tmp_path, "Low", eid="q3")
    client.post("/v1/verdict", json={"event_id": b, "verdict": "suspicious"}, headers=KEY)
    ids = [e["event_id"] for e in client.get("/v1/queue", headers=KEY).json()]
    assert "q1" in ids and "q2" not in ids and "q3" not in ids

def test_import_history_script(tmp_path):
    import subprocess, sys, pickle
    csv = tmp_path / "h.csv"; make(3000, n_users=50, seed=9).to_csv(csv, index=False)
    out = tmp_path / "st.pkl"
    r = subprocess.run([sys.executable, "scripts/import_history.py", "--csv", str(csv), "--state", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(pickle.load(open(out, "rb")).users) > 10
