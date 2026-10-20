import numpy as np, pandas as pd
from riskengine.data import normalize_chunk, event_to_row
from riskengine.features import build_features, FeatureState, FEATURES
from scripts.make_mock_data import make

def _prepared(n=4000, seed=1):
    raw = make(n, n_users=150, seed=seed)
    return raw, normalize_chunk(raw)

def test_online_equals_batch():
    """Replaying events one by one through FeatureState gives the SAME features as the vectorised
    training code. This is the guarantee against train/serve skew."""
    raw, prep = _prepared()
    batch = build_features(prep)
    st, rows = FeatureState(), []
    for rec in raw.to_dict("records"):
        r = event_to_row(rec)               # live path: raw dict -> hashed row
        rows.append(st.features(r)); st.update(r, trusted=True)
    online = pd.DataFrame(rows)[FEATURES].astype("float32")
    np.testing.assert_allclose(online.to_numpy(), batch.to_numpy(), rtol=1e-4, atol=1e-5, equal_nan=True)

def test_no_future_leakage():
    """A row's features must not change if later rows are removed."""
    _, prep = _prepared()
    full = build_features(prep)
    cut = build_features(prep.iloc[:2500])
    np.testing.assert_allclose(full.iloc[:2500].to_numpy(), cut.to_numpy(), equal_nan=True)

def test_first_login_is_cold_start():
    _, prep = _prepared()
    f = build_features(prep)
    first = f[f["history_count_log"] == 0]
    assert (first["new_country"] == 1).all() and first["hour_dev"].isna().all()

def test_untrusted_update_does_not_teach_country():
    raw, _ = _prepared(300)
    st = FeatureState()
    rec = raw.iloc[0].to_dict(); r = event_to_row(rec)
    st.update(r, trusted=False)
    assert st.features(r)["new_country"] == 1.0   # still 'new' -> attacker did not become normal
    st.update(r, trusted=True)
    assert st.features(r)["new_country"] == 0.0


def test_online_equals_batch_dense_windows():
    """Same check on data packed into 2 days so the 1h / 24h window features are non-trivial."""
    raw = make(3000, n_users=80, seed=11, span_days=2)
    prep = normalize_chunk(raw)
    batch = build_features(prep)
    assert batch["ip_n_24h_log"].max() > 0 and batch["asn_fail_rate_24h"].max() > 0
    st, rows = FeatureState(), []
    for rec in raw.to_dict("records"):
        r = event_to_row(rec); rows.append(st.features(r)); st.update(r, trusted=True)
    online = pd.DataFrame(rows)[FEATURES].astype("float32")
    np.testing.assert_allclose(online.to_numpy(), batch.to_numpy(), rtol=1e-4, atol=1e-5, equal_nan=True)
