import pytest
from fastapi.testclient import TestClient
from riskengine.data import normalize_chunk
from riskengine.training import train
from riskengine.store import Store
from riskengine.api import create_app
from riskengine.features import FeatureState
from riskengine.persist import load_state, save_state
from riskengine.modeling import RiskModel
from riskengine.features import FEATURES, build_features
from scripts.make_mock_data import make

# A real key, because the service now refuses keys shorter than 24 characters.
API_KEY = "test-key-0123456789-abcdefghijkl"
KEY = {"X-API-Key": API_KEY}

@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    train(normalize_chunk(make(40_000, n_users=500, seed=3, span_days=30)), str(d), verbose=False)
    return d

@pytest.fixture()
def client(trained, tmp_path):
    # timestamp_source="client": these tests replay a scripted 2020 timeline, which is exactly the
    # case the secure default (server clock) is designed to distrust. tests/test_security.py covers
    # the default and what it blocks.
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"),
                     API_KEY, timestamp_source="client")
    with TestClient(app) as c:
        yield c

@pytest.fixture()
def busy_client(trained, tmp_path, monkeypatch):
    """A system that has already seen a few days of ordinary traffic (as a real site would after launch)."""
    monkeypatch.setattr("riskengine.api.WARMUP_MIN_EVENTS", 1000)
    raw = make(30_000, n_users=300, seed=21, span_days=3)
    prep = normalize_chunk(raw).sort_values("ts_ms", kind="stable")
    st = FeatureState()
    for r in prep.to_dict("records"):
        st.update(r, trusted=bool(r["success"]) and not r["attack_ip"] and not r["ato"], counters=True)
    asn = int(raw.loc[~raw["Is Attack IP"], "ASN"].mode()[0])
    from riskengine.data import event_to_row                      # an established user with 6 trusted past logins
    for i in range(6):
        st.update(event_to_row({"Login Timestamp": f"2020-02-03T1{i}:00:00Z", "User ID": "est", "IP Address": "10.0.0.1",
                                "Country": "NO", "ASN": asn, "Browser Name and Version": "Chrome 80",
                                "OS Name and Version": "Windows 10", "Device Type": "desktop",
                                "Round-Trip Time [ms]": 100.0, "Login Successful": True}), trusted=True)
    save_state(st, tmp_path / "s.json.gz")
    app = create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"), str(tmp_path / "r.db"),
                     API_KEY, timestamp_source="client")
    with TestClient(app) as c:
        c.busy_asn = asn
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
    assert "ip_risk" in r and r["takeover_risk"] is None      # brand-new user: takeover head needs history

def test_known_user_vs_stuffing_ip(busy_client):
    """On a warmed-up system an established user's usual login scores below a login from an IP that just
    hammered many accounts from an unfamiliar network."""
    c, asn = busy_client, busy_client.busy_asn
    normal = c.post("/v1/score", json=ev(user="est", asn=asn, t="2020-02-04T09:00:00Z"), headers=KEY).json()
    for i in range(30):                                  # one IP hammering many accounts from an unfamiliar network
        c.post("/v1/score", json=ev(user=f"v{i}", ip="203.0.113.9", country="RU", asn=64500, browser="Python-urllib 3",
               os="Linux", device="unknown", success=False, t=f"2020-02-04T10:{i:02d}:00Z"), headers=KEY)
    atk = c.post("/v1/score", json=ev(user="victim", ip="203.0.113.9", country="RU", asn=64500, browser="Python-urllib 3",
                 os="Linux", device="unknown", t="2020-02-04T11:00:00Z"), headers=KEY).json()
    assert atk["ip_risk"] > normal["ip_risk"] and atk["tier"] in ("Medium", "High")

def test_learning_gate(client, tmp_path):
    ids = [client.post("/v1/score", json=ev(user=f"g{i}", t=f"2020-06-01T1{i}:00:00Z"), headers=KEY).json()["event_id"]
           for i in range(4)]
    for eid, v in zip(ids[:3], ["benign", "confirmed_malicious", "suspicious"]):
        assert client.post("/v1/verdict", json={"event_id": eid, "verdict": v, "analyst": "a"}, headers=KEY).status_code == 200
    assert client.post("/v1/verdict", json={"event_id": ids[0], "verdict": "bogus", "analyst": "a"}, headers=KEY).status_code == 422
    assert client.post("/v1/verdict", json={"event_id": "nope", "verdict": "benign", "analyst": "a"}, headers=KEY).status_code == 404
    # an unattributed verdict is refused: it would be an unauditable change to the next model
    assert client.post("/v1/verdict", json={"event_id": ids[0], "verdict": "benign"}, headers=KEY).status_code == 422
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
    b = _inject(tmp_path, "Medium", eid="q2"); _inject(tmp_path, "High", eid="q1")
    _inject(tmp_path, "Low", eid="q3")
    client.post("/v1/verdict", json={"event_id": b, "verdict": "suspicious", "analyst": "a"}, headers=KEY)
    ids = [e["event_id"] for e in client.get("/v1/queue", headers=KEY).json()]
    assert "q1" in ids and "q2" not in ids and "q3" not in ids

def test_import_history_script(tmp_path):
    import pathlib, subprocess, sys
    # Resolve the script from THIS file rather than the working directory. Running pytest from a
    # parent folder is a perfectly normal thing to do and must not break the test.
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "import_history.py"
    assert script.exists(), f"expected the script beside the tests, at {script}"
    csv = tmp_path / "h.csv"; make(3000, n_users=50, seed=9).to_csv(csv, index=False)
    out = tmp_path / "st.json.gz"
    r = subprocess.run([sys.executable, str(script), "--csv", str(csv), "--state", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(load_state(out).users) > 10        # loaded without pickle


def _login(client, confirm=True, **kw):
    """What a real site does: score the login; if the user is challenged (Medium) and passes the OTP, confirm it."""
    r = client.post("/v1/score", json=ev(**kw), headers=KEY).json()
    if confirm and r["tier"] == "Medium":
        client.post(f"/v1/confirm/{r['event_id']}", headers=KEY)
    return r

def test_empty_system_is_log_only(client):
    """Brand-new system: network-volume features are meaningless, so nothing may be enforced yet."""
    r = client.post("/v1/score", json=ev(), headers=KEY).json()
    assert r["warming_up"] is True and r["action"] == "allow" and "suggested_action" in r
    assert client.get("/health").json()["warming_up"] is True

def test_takeover_head_needs_history_and_uses_it(busy_client):
    """An established user who suddenly logs in from a new country/network gets a higher takeover score."""
    c, asn = busy_client, busy_client.busy_asn
    same = _login(c, confirm=False, user="est", asn=asn, t="2020-02-04T09:00:00Z")
    odd = _login(c, confirm=False, user="est", ip="198.51.100.77", country="BR", asn=64999, t="2020-02-04T10:00:00Z")
    fresh = _login(c, confirm=False, user="never-seen", asn=asn, t="2020-02-04T09:30:00Z")
    assert fresh["takeover_risk"] is None                                       # no history -> head abstains
    assert same["takeover_risk"] is not None and odd["takeover_risk"] is not None, (same, odd)
    assert odd["takeover_risk"] > same["takeover_risk"]
    assert odd["risk_score"] == max(odd["ip_risk"], odd["takeover_risk"])      # final risk = worse head

def test_profile_only_grows_from_trusted_logins(busy_client):
    """Unconfirmed failed/suspicious logins must not become the user's 'normal' (attacker cannot teach the system)."""
    r = None
    for i in range(6):
        r = busy_client.post("/v1/score", json=ev(user="unconf", ip=f"203.0.113.{i+1}", country="RU", asn=64500,
                             browser="Python-urllib 3", os="Linux", device="unknown", success=False,
                             t=f"2020-02-04T0{i+2}:00:00Z"), headers=KEY).json()
    assert r["takeover_risk"] is None          # failed, unconfirmed attempts never built a trusted history