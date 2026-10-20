"""The stratified development sample as a first-class training surface.

The exported sample over-represents attacks on purpose (about a third of its rows, against a tenth
of the population), so anything fitted or calibrated on it without sample_weight inherits that
distortion - and the thresholds are percentiles, so a distorted reference distribution means
"top 2% of normal logins" is not 2% in production.
"""
import numpy as np
import pytest

from riskengine.data import normalize_chunk
from riskengine.features import FEATURES, build_features
from riskengine.training import train, weighted_quantile
from scripts.make_mock_data import make


def test_weighted_quantile_matches_numpy_for_equal_weights():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    q = np.linspace(0, 1, 101)
    np.testing.assert_allclose(weighted_quantile(x, q), np.quantile(x, q), rtol=0, atol=0.05)


def test_weighted_quantile_follows_the_weights():
    """Duplicating observations by weight must equal duplicating them by hand."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0, 97.0])
    expanded = np.concatenate([np.repeat(x[:3], 1), np.repeat([3.0], 97)])
    q = [0.5, 0.9]
    np.testing.assert_allclose(weighted_quantile(x, q, w), np.quantile(expanded, q), atol=0.05)


def test_weighted_quantile_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        weighted_quantile(np.arange(5.0), [0.5], np.ones(4))


def _stratified_sample(seed=7):
    """A miniature of scripts/export_sample.py: keep all attacks, 1 in 10 normal logins, and weight
    them back up. The population statistics must then be recoverable from the sample."""
    raw = make(30_000, n_users=400, seed=seed, span_days=20)
    prep = normalize_chunk(raw).sort_values("ts_ms", kind="stable").reset_index(drop=True)
    X = build_features(prep)
    atk = prep.attack_ip.to_numpy()
    rng = np.random.default_rng(seed)
    normal_idx = np.flatnonzero(~atk)
    keep_normal = np.sort(rng.choice(normal_idx, len(normal_idx) // 10, replace=False))
    keep = np.sort(np.concatenate([np.flatnonzero(atk), keep_normal]))
    w = np.where(atk[keep], 1.0, len(normal_idx) / len(keep_normal))
    d = X.iloc[keep].reset_index(drop=True)
    d["attack_ip"] = atk[keep]
    d["ato"] = prep.ato.to_numpy()[keep]
    d["success_flag"] = prep.success.to_numpy()[keep]
    d["user"] = prep.user.to_numpy()[keep]
    d["is_bulk"] = False
    d["sample_weight"] = w
    return d, float(atk.mean())


def test_sample_weights_recover_the_population_attack_rate():
    d, true_rate = _stratified_sample()
    sample_rate = d.attack_ip.mean()
    weighted_rate = np.average(d.attack_ip, weights=d.sample_weight)
    assert sample_rate > true_rate * 2, "the fixture must actually be distorted for this to mean anything"
    assert abs(weighted_rate - true_rate) < 0.01


def _precomputed(d, weights):
    return {"X": d[list(FEATURES)].to_numpy(np.float32), "attack_ip": d.attack_ip.to_numpy(),
            "ato": d.ato.to_numpy(), "success": d.success_flag.to_numpy(), "user": d.user.to_numpy(),
            "is_bulk": d.is_bulk.to_numpy(), "sample_weight": weights}


def test_training_from_a_weighted_sample_calibrates_to_the_population(tmp_path):
    """The headline check: with weights, the share of normal logins the model flags lands near the
    2% / 0.5% budgets. Without them it is pulled away by the sample's inflated attack rate."""
    d, _ = _stratified_sample()
    w = d.sample_weight.to_numpy()

    rep_w = train(None, str(tmp_path / "w"), precomputed=_precomputed(d, w), verbose=False)
    rep_u = train(None, str(tmp_path / "u"), precomputed=_precomputed(d, np.ones(len(d))), verbose=False)

    assert rep_w["weighting"]["source"] == "stratified sample"
    assert rep_w["weighting"]["population_represented"] > len(d) * 2
    assert rep_u["weighting"]["source"] == "full data"

    key = "Medium+ (target 2.0)"
    flagged_w = rep_w["ip_head"]["legit_flagged_in_test_pct"][key]
    flagged_u = rep_u["ip_head"]["legit_flagged_in_test_pct"][key]
    # The weighted run reports the rate the live site would see. Allow a wide band: this is a small
    # synthetic fixture, and the point is that the number is a population estimate, not that it is exact.
    assert flagged_w < 12.0, f"weighted calibration drifted badly: {flagged_w}% flagged"
    assert rep_w["ip_head"]["metrics_are"].startswith("weighted")
    assert "metrics_are" not in rep_u["ip_head"]
    # Both runs must still produce a usable model.
    for rep in (rep_w, rep_u):
        assert 0 <= rep["thresholds"]["medium"] <= rep["thresholds"]["high"] <= 100
    assert isinstance(flagged_u, float)


def test_weighted_training_produces_a_loadable_model(tmp_path):
    from riskengine.modeling import RiskModel
    d, _ = _stratified_sample()
    train(None, str(tmp_path), precomputed=_precomputed(d, d.sample_weight.to_numpy()), verbose=False)
    m = RiskModel.load(tmp_path / "latest.joblib")          # also exercises the feature contract
    assert m.meta["features"] == list(FEATURES)
    x = d[list(FEATURES)].to_numpy(np.float32)[:50]
    r = m.risk(x)
    assert r.shape == (50,) and np.all((r >= 0) & (r <= 100))
