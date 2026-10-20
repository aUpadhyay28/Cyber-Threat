import numpy as np
from riskengine.data import normalize_chunk
from riskengine.modeling import _ecdf, OffHead, RiskModel
from riskengine.training import train
from scripts.make_mock_data import make

def test_ecdf_ties_get_middle_rank():
    """If every reference score is identical, a login must NOT be ranked as the riskiest of all."""
    ref = np.zeros(1000)
    assert abs(_ecdf(ref, np.array([0.0]))[0] - 0.5) < 1e-9
    assert _ecdf(ref, np.array([1.0]))[0] == 1.0 and _ecdf(ref, np.array([-1.0]))[0] == 0.0

def test_ecdf_is_monotone():
    ref = np.sort(np.random.default_rng(0).normal(size=5000))
    x = np.linspace(-4, 4, 200)
    assert (np.diff(_ecdf(ref, x)) >= 0).all()

def test_bulk_pseudo_accounts_are_excluded_and_model_still_trains(tmp_path):
    df = normalize_chunk(make(30_000, n_users=400, seed=7, span_days=30))
    rng = np.random.default_rng(0)
    pseudo = rng.random(len(df)) < 0.25                    # a placeholder account that owns 25% of the rows, all failed
    df.loc[pseudo, "user"] = np.uint64(12345); df.loc[pseudo, "success"] = False
    df.loc[pseudo, ["attack_ip", "ato"]] = False
    heaviest_real = int(df.user[~pseudo].value_counts().max())      # threshold just above the busiest REAL user
    rep = train(df, str(tmp_path), bulk_threshold=heaviest_real + 1, verbose=False)
    assert 22 < rep["bulk_pseudo_account_rows_pct"] < 28
    assert RiskModel.load(tmp_path / "latest.joblib").medium_thr == 98.0

def test_takeover_head_switches_off_when_no_takeovers(tmp_path):
    df = normalize_chunk(make(20_000, n_users=300, seed=8, span_days=30)); df["ato"] = False
    rep = train(df, str(tmp_path), verbose=False)
    m = RiskModel.load(tmp_path / "latest.joblib")           # must be loadable (picklable) too
    assert isinstance(m.dev_lr, OffHead) and "not trained" in rep["takeover_head"]["note"]


def test_lean_loader_matches_concat_and_respects_max_rows(tmp_path):
    import pandas as pd
    from riskengine.data import load_prepared
    full = normalize_chunk(make(9_000, n_users=100, seed=5, span_days=10))
    for i in range(3):
        full.iloc[i * 3000:(i + 1) * 3000].to_parquet(tmp_path / f"part_{i:04d}.parquet")
    got = load_prepared(str(tmp_path))
    pd.testing.assert_frame_equal(got.reset_index(drop=True), full.reset_index(drop=True))
    assert len(load_prepared(str(tmp_path), max_rows=4_000)) == 4_000


def test_train_accepts_a_loader_function(tmp_path):
    df = normalize_chunk(make(20_000, n_users=300, seed=9, span_days=20))
    rep = train(lambda: df.copy(), str(tmp_path), verbose=False)
    assert rep["rows"]["total"] == 20_000
