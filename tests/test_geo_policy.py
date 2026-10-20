"""Server-side geo resolution and the home-country policy.

Two things are under test, and the first one is what makes the second worth having: if country can
be supplied by the caller, then "logins from outside India get challenged" is a rule an attacker
opts out of by typing IN.
"""
import pytest
from fastapi.testclient import TestClient

from riskengine.api import create_app
from riskengine.data import normalize_chunk
from riskengine.geoip import ClientSuppliedResolver, StaticResolver, apply_resolution, is_private
from riskengine.policy import HomeCountryPolicy
from riskengine.store import Store
from riskengine.training import train
from scripts.make_mock_data import make

API_KEY = "test-key-0123456789-abcdefghijkl"
KEY = {"X-API-Key": API_KEY}

INDIA_IP = "49.36.1.1"
FOREIGN_IP = "203.0.113.50"
GEO = StaticResolver({INDIA_IP: ("IN", 9829), FOREIGN_IP: ("RU", 64500)})


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("geo-model")
    train(normalize_chunk(make(40_000, n_users=500, seed=3, span_days=30)), str(d), verbose=False)
    return d


def _app(trained, tmp_path, **kw):
    return create_app(str(trained / "latest.joblib"), str(tmp_path / "s.json.gz"),
                      str(tmp_path / "r.db"), API_KEY, geo_resolver=GEO, **kw)


def _ev(ip=INDIA_IP, user="u1", country="IN", success=True):
    return dict(user_id=user, ip=ip, country=country, asn=1, browser="Chrome 120",
                os="Windows 11", device="desktop", rtt_ms=50.0, success=success)


# ------------------------------------------------------------------ resolution
def test_private_addresses_have_no_geography():
    assert is_private("10.0.0.1") and is_private("127.0.0.1") and is_private("192.168.1.5")
    assert is_private("::1") and is_private("not-an-ip")
    assert not is_private(INDIA_IP)


def test_resolver_overrides_the_caller():
    out = apply_resolution(GEO, FOREIGN_IP, "IN", 9829)
    assert out["country"] == "RU" and out["asn"] == 64500
    assert out["geo_source"] == "static"


def test_resolver_records_a_mismatch():
    """A caller insisting on a different country is a signal worth logging, not ignoring."""
    out = apply_resolution(GEO, FOREIGN_IP, "IN", 1)
    assert out["geo_mismatch"] == {"claimed": "IN", "resolved": "RU"}
    out2 = apply_resolution(GEO, INDIA_IP, "in", 1)
    assert out2["geo_mismatch"] is None            # same country, different case


def test_unresolvable_ip_falls_back_to_the_caller():
    out = apply_resolution(GEO, "198.51.100.9", "SE", 123)
    assert out["country"] == "SE" and out["geo_source"] == "caller"


def test_client_supplied_resolver_changes_nothing():
    out = apply_resolution(ClientSuppliedResolver(), FOREIGN_IP, "IN", 7)
    assert out["country"] == "IN" and out["asn"] == 7 and out["geo_source"] == "caller"


def test_spoofed_country_cannot_beat_the_policy(trained, tmp_path):
    """The whole point: claiming IN from a Russian address must not buy a domestic score.

    Asserts the outcome rather than which layer produced it - on a cold system the model may
    already rate this Medium on its own, and the policy then correctly reports that it is already
    above the floor. Either way the login is not waved through and RU is what gets recorded.
    """
    app = _app(trained, tmp_path, policy=HomeCountryPolicy(home_countries=frozenset({"IN"})))
    with TestClient(app) as c:
        r = c.post("/v1/score", json=_ev(ip=FOREIGN_IP, country="IN"), headers=KEY).json()
        assert r["tier"] in ("Medium", "High")
        assert r["policy"] != [], "the home-country rule must have had an opinion"
        stored = Store(str(tmp_path / "r.db")).get_event(r["event_id"])
        assert "RU" in stored["raw_json"] and '"IN"' not in stored["raw_json"]


# ------------------------------------------------------------------ the policy itself
def _feats(new_country=1.0):
    return {"new_country": new_country}


def test_policy_is_off_until_a_home_country_is_set():
    pol = HomeCountryPolicy()
    assert not pol.enabled
    out = pol.apply({"tier": "Low", "action": "allow"}, "RU", _feats(), warming=False)
    assert out["tier"] == "Low" and out["policy"] == []
    assert out["model_tier"] == "Low"


def test_domestic_login_is_untouched():
    pol = HomeCountryPolicy(home_countries=frozenset({"IN"}))
    out = pol.apply({"tier": "Low", "action": "allow"}, "IN", _feats(), warming=False)
    assert out["tier"] == "Low" and out["policy"] == []


def test_unfamiliar_foreign_login_is_raised_to_step_up():
    pol = HomeCountryPolicy(home_countries=frozenset({"IN"}))
    out = pol.apply({"tier": "Low", "action": "allow"}, "AE", _feats(new_country=1.0), warming=False)
    assert out["tier"] == "Medium" and out["action"] == "step_up_auth"
    assert out["model_tier"] == "Low" and "foreign_country_step_up" in out["policy"]


def test_a_country_the_user_already_uses_is_not_challenged_again():
    """An Indian user who works from Dubai every month should stop getting OTPs for Dubai."""
    pol = HomeCountryPolicy(home_countries=frozenset({"IN"}))
    out = pol.apply({"tier": "Low", "action": "allow"}, "AE", _feats(new_country=0.0), warming=False)
    assert out["tier"] == "Low"
    assert out["policy"] == ["foreign_country_known_to_user_no_action"]


def test_policy_can_be_set_to_challenge_every_foreign_login():
    pol = HomeCountryPolicy(home_countries=frozenset({"IN"}), only_unfamiliar=False)
    out = pol.apply({"tier": "Low", "action": "allow"}, "AE", _feats(new_country=0.0), warming=False)
    assert out["tier"] == "Medium"


def test_policy_never_lowers_the_model():
    pol = HomeCountryPolicy(home_countries=frozenset({"IN"}))
    out = pol.apply({"tier": "High", "action": "block_and_review"}, "RU", _feats(), warming=False)
    assert out["tier"] == "High" and out["action"] == "block_and_review"
    assert out["policy"] == ["foreign_country_already_above_floor"]


def test_unknown_country_is_not_escalated():
    """Private ranges and unmapped IPs resolve to nothing; escalating them breaks staging."""
    pol = HomeCountryPolicy(home_countries=frozenset({"IN"}))
    for country in (None, "", "NA"):
        out = pol.apply({"tier": "Low", "action": "allow"}, country, _feats(), warming=False)
        assert out["tier"] == "Low", country


def test_policy_works_during_warmup():
    """It needs no traffic history, so unlike the model it is useful on day one of a deployment."""
    pol = HomeCountryPolicy(home_countries=frozenset({"IN"}))
    out = pol.apply({"tier": "Low", "action": "allow"}, "RU", _feats(), warming=True)
    assert out["tier"] == "Medium"
    off = HomeCountryPolicy(home_countries=frozenset({"IN"}), apply_while_warming=False)
    assert off.apply({"tier": "Low", "action": "allow"}, "RU", _feats(), warming=True)["tier"] == "Low"


def test_from_env_reads_configuration(monkeypatch):
    monkeypatch.setenv("RISK_HOME_COUNTRIES", "in, np ,BT")
    monkeypatch.setenv("RISK_FOREIGN_MIN_TIER", "high")
    pol = HomeCountryPolicy.from_env()
    assert pol.home_countries == frozenset({"IN", "NP", "BT"}) and pol.min_tier == "High"
    monkeypatch.setenv("RISK_FOREIGN_MIN_TIER", "nonsense")
    with pytest.raises(RuntimeError, match="RISK_FOREIGN_MIN_TIER"):
        HomeCountryPolicy.from_env()


def test_escalated_login_does_not_join_the_normal_profile(trained, tmp_path):
    """A step-up is not an endorsement: the login must not teach the account that RU is normal."""
    app = _app(trained, tmp_path, policy=HomeCountryPolicy(home_countries=frozenset({"IN"})))
    with TestClient(app) as c:
        r = c.post("/v1/score", json=_ev(ip=FOREIGN_IP, user="trav", country="RU"), headers=KEY).json()
        assert r["tier"] != "Low"
        stored = Store(str(tmp_path / "r.db")).get_event(r["event_id"])
        assert stored["trusted"] == 0


def test_health_reports_the_active_configuration(trained, tmp_path):
    app = _app(trained, tmp_path, policy=HomeCountryPolicy(home_countries=frozenset({"IN"})))
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["geoip_source"] == "static" and h["home_countries"] == ["IN"]


def test_model_tier_is_always_reported(trained, tmp_path):
    """You cannot tune a rule whose cost you cannot separate from the model's."""
    app = _app(trained, tmp_path, policy=HomeCountryPolicy(home_countries=frozenset({"IN"})))
    with TestClient(app) as c:
        r = c.post("/v1/score", json=_ev(), headers=KEY).json()
        assert "model_tier" in r and "policy" in r
