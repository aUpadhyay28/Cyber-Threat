"""Find out WHY the model is weak. Prints: attack rate over time, how many logins are 'cold start',
which single features carry signal, and model quality on cold-start vs users-with-history.
   python scripts/diagnose.py --data data/prepared_trial --model models_trial/latest.joblib"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from riskengine.features import build_features, FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/prepared"); ap.add_argument("--model", default=None)
ap.add_argument("--max-rows", type=int, default=None)
ap.add_argument("--auc-sample", type=int, default=1_000_000, help="rows sampled per window for AUCs (speed)")
a = ap.parse_args()
df = pd.concat((pd.read_parquet(p) for p in sorted(glob.glob(f"{a.data}/part_*.parquet"))), ignore_index=True)
df = df.sort_values("ts_ms", kind="stable").reset_index(drop=True)
if a.max_rows: df = df.iloc[: a.max_rows]
n = len(df); X = build_features(df); y = (df.attack_ip | df.ato).to_numpy(); ato_all = df.ato.to_numpy()
cold = (X["history_count_log"] < np.log1p(3)).to_numpy()
days = (df.ts_ms.max() - df.ts_ms.min()) / 86_400_000
print(f"\nrows={n:,}  time span={days:.1f} days  attack rate={y.mean():.2%}  distinct users={df.user.nunique():,}")
print(f"cold-start logins (<3 prior logins): {cold.mean():.1%}")

print("\n--- 1) attack rate and cold-start share over time (10 equal buckets) ---")
b = pd.cut(np.arange(n), 10, labels=False)
print(pd.DataFrame({"attack_rate_%": pd.Series(y).groupby(b).mean() * 100,
                    "cold_start_%": pd.Series(cold).groupby(b).mean() * 100}).round(2).to_string())

te = slice(int(n * 0.8), n)
_rng = np.random.default_rng(0)
def _pick(lo, hi):
    idx = np.arange(lo, hi)
    return np.sort(_rng.choice(idx, a.auc_sample, replace=False)) if len(idx) > a.auc_sample else idx
Xt_full, yt_full, ct_full = X.iloc[te], y[te], cold[te]
sel = _pick(int(n * 0.8), n) - int(n * 0.8)
Xt, yt, ct, at_ = Xt_full.iloc[sel], yt_full[sel], ct_full[sel], ato_all[te][sel]
print("\n--- 2) single-feature ROC-AUC on the test period (0.5 = no signal; <0.5 = inverted) ---")
print(f"(account-takeover positives in test period: {int(at_.sum())} -> its column is only meaningful if this is >= ~30)")
rows = []
for f in FEATURES:
    v = Xt[f].fillna(Xt[f].median()).to_numpy()
    if np.nanstd(v) > 0 and 0 < yt.sum() < len(yt):
        ato_auc = roc_auc_score(at_, v) if at_.sum() >= 5 else np.nan
        rows.append((f, roc_auc_score(yt, v), ato_auc))
r = pd.DataFrame(rows, columns=["feature", "auc_attack_ip", "auc_takeover"]).assign(strength=lambda d: (d.auc_attack_ip - .5).abs()).sort_values("strength", ascending=False)
print(r[["feature", "auc_attack_ip", "auc_takeover"]].round(3).to_string(index=False))

if a.model:
    from riskengine.modeling import RiskModel
    Xt, yt, ct = Xt_full, yt_full, ct_full           # section 3 scores the whole test period
    m = RiskModel.load(a.model); risk = m.risk(Xt.to_numpy(np.float32))
    print("\n--- 3) model quality: cold-start vs users with history (test period) ---")
    for name, mask in (("cold-start (<3 prior logins)", ct), ("has history (>=3)", ~ct)):
        yy = yt[mask]
        if 0 < yy.sum() < len(yy):
            print(f"{name:32s} rows={mask.sum():>9,}  attack rate={yy.mean():6.2%}  PR-AUC={average_precision_score(yy, risk[mask]):.3f}  ROC-AUC={roc_auc_score(yy, risk[mask]):.3f}")
        else:
            print(f"{name:32s} rows={mask.sum():>9,}  (no positives or all positives)")
    print("\nHow to read it: if 'has history' is much better than 'cold-start', the weak overall score comes from thin "
          "history in a short slice -> train on the full file. If both are weak, the features need work.")

# ---------- 4) drift ----------
print("\n--- 4) DRIFT: does each feature mean the same thing early vs late? ---")
e_idx = _pick(int(n * 0.2), int(n * 0.4)); l_idx = _pick(int(n * 0.8), n)
Xe, ye, Xl, yl = X.iloc[e_idx], y[e_idx], X.iloc[l_idx], y[l_idx]
out = []
for f in FEATURES:
    ve, vl = Xe[f].fillna(Xe[f].median()).to_numpy(), Xl[f].fillna(Xe[f].median()).to_numpy()
    if np.std(ve) == 0 or np.std(vl) == 0 or not (0 < ye.sum() < len(ye)) or not (0 < yl.sum() < len(yl)):
        continue
    out.append((f, roc_auc_score(ye, ve), roc_auc_score(yl, vl), float(np.median(ve)), float(np.median(vl)),
                float(np.quantile(ve, .9)), float(np.quantile(vl, .9))))
dr = pd.DataFrame(out, columns=["feature", "auc_early", "auc_late", "median_early", "median_late", "p90_early", "p90_late"])
dr["auc_change"] = (dr.auc_late - dr.auc_early).round(3)
print(dr.reindex(dr.auc_change.abs().sort_values(ascending=False).index).round(3).to_string(index=False))
print("\nHow to read it: a large auc_change or a big median/p90 shift means that feature stops meaning the same "
      "thing over time (typical for all-time running counts) -> replace it with a recent-window version.")

print("\n--- 5) where do account-takeover logins fall in time? (10 equal buckets) ---")
print(pd.Series(ato_all).groupby(pd.cut(np.arange(n), 10, labels=False)).sum().to_string())
print(f"total takeover logins: {int(ato_all.sum())}  | distinct users involved: {df.user[ato_all].nunique()}")
