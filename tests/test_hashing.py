"""Keyed identifier hashing and the production startup guard.

The privacy property: with RISK_HASH_KEY set, someone holding a state file or database dump cannot
recover user IDs by hashing candidate emails, because they do not have the key. The correctness
property: every artifact records which key made it, and a mismatch is refused instead of silently
treating every returning user as brand new.
"""
import pandas as pd
import pytest

from riskengine import hashing
from riskengine.api import check_production_config
from riskengine.data import (event_to_row, event_to_row_via_frame, load_prepared, normalize_chunk,
                             write_prepared_fingerprint)
from riskengine.features import FeatureState
from riskengine.persist import load_state, save_state
from scripts.make_mock_data import make

SECRET_A = "a" * 8 + "-secret-key-for-tests-0001"
SECRET_B = "b" * 8 + "-secret-key-for-tests-0002"


@pytest.fixture
def keyed(monkeypatch):
    def _set(secret):
        if secret is None:
            monkeypatch.delenv(hashing.ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(hashing.ENV_VAR, secret)
    yield _set
    monkeypatch.delenv(hashing.ENV_VAR, raising=False)


def _event():
    return make(50, n_users=5, seed=1).iloc[0].to_dict()


def test_unset_key_reproduces_the_original_hashes(keyed):
    """No key = pandas' public default, so existing benchmark models and state files keep working."""
    keyed(None)
    ev = _event()
    raw = str(ev["User ID"])
    assert event_to_row(ev)["user"] == int(pd.util.hash_array(pd.Series([raw]).to_numpy(dtype=object))[0])
    assert hashing.key_fingerprint() == hashing.UNKEYED


def test_a_key_changes_every_identifier_hash(keyed):
    ev = _event()
    keyed(None); plain = event_to_row(ev)
    keyed(SECRET_A); a = event_to_row(ev)
    keyed(SECRET_B); b = event_to_row(ev)
    for col in ("user", "ip", "country", "browser", "device"):
        assert len({plain[col], a[col], b[col]}) == 3, col
    # ...but the non-identifier fields are untouched
    assert plain["ts_ms"] == a["ts_ms"] and plain["success"] == a["success"]


def test_fast_path_and_training_path_agree_under_a_key(keyed):
    """Train/serve parity must hold for the keyed hash too, or live features read the wrong history."""
    keyed(SECRET_A)
    for _, r in make(200, n_users=20, seed=4).iterrows():
        ev = r.to_dict()
        assert event_to_row(ev) == event_to_row_via_frame(ev)


def test_short_key_is_refused(keyed):
    keyed("too-short")
    with pytest.raises(RuntimeError, match="at least"):
        hashing.hash_key()


def test_state_file_from_another_key_is_refused(keyed, tmp_path):
    keyed(SECRET_A)
    save_state(FeatureState(), tmp_path / "s.json.gz")          # even an empty state carries the fingerprint
    assert load_state(tmp_path / "s.json.gz").glob_total == 0      # same key: fine
    keyed(SECRET_B)
    with pytest.raises(RuntimeError, match="different hash|fingerprint"):
        load_state(tmp_path / "s.json.gz")
    keyed(None)
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_state(tmp_path / "s.json.gz")


def test_prepared_data_from_another_key_is_refused(keyed, tmp_path):
    keyed(SECRET_A)
    write_prepared_fingerprint(tmp_path)
    normalize_chunk(make(500, n_users=20, seed=2)).to_parquet(tmp_path / "part_0000.parquet")
    assert len(load_prepared(str(tmp_path))) == 500
    keyed(SECRET_B)
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_prepared(str(tmp_path))


def test_production_guard_lists_every_missing_setting(keyed, monkeypatch, tmp_path):
    keyed(None)
    monkeypatch.delenv("RISK_GEOIP_MODE", raising=False)
    with pytest.raises(RuntimeError) as e:
        check_production_config(ts_source="client", model_sha256=None,
                                model_path=str(tmp_path / "missing.joblib"))
    msg = str(e.value)
    for needle in ("RISK_TIMESTAMP_SOURCE", "RISK_HASH_KEY", "not pinned", "RISK_GEOIP_MODE"):
        assert needle in msg


def test_production_guard_passes_when_configured(keyed, monkeypatch):
    keyed(SECRET_A)
    monkeypatch.setenv("RISK_GEOIP_MODE", "maxmind")
    monkeypatch.delenv("RISK_ALLOW_SHORT_KEY", raising=False)
    check_production_config(ts_source="server", model_sha256="ab" * 32)
