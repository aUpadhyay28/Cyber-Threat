"""Two questions, one pass over the data.
 A) How good is the model at FIXED false-alarm rates, and do its tier thresholds hold over time?
 B) Do 'this login deviates from the user's own profile' signals detect the account-takeover logins?
    (descriptive: no training, so no leakage; 95% confidence intervals because there are only ~141 cases)
   python scripts/final_analysis.py --data data/prepared --model models_v3/latest.joblib"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, glob
import numpy as np, pandas as pd
from riskengine.features import build_features, FEATURES
from riskengine.modeling import RiskModel

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/prepared"); ap.add_argument("--model", default="models/latest.joblib")
ap.add_argument("--neg-sample", type=int, default=300_000); ap.add_argument("--max-rows", type=int, default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)

df = pd.concat((pd.read_parquet(p) for p in sorted(glob.glob(f"{a.data}/part_*.parquet"))), ignore_index=True)
df = df.sort_values("ts_ms", kind="stable").reset_index(drop=True)
if a.max_rows: df = df.iloc[: a.max_rows]
n = len(df); print(f"{n:,} logins; building features ...", flush=True)
Xdf = build_features(df); X = Xdf.to_numpy(np.float32)
atk, ato = df.attack_ip.to_numpy(), df.ato.to_numpy(); bad = atk | ato; legit = ~bad
ts = df.ts_ms.to_numpy(); del df

def score(M, Xs, chunk=1_000_000):
    return np.concatenate([M.risk(Xs[i:i + chunk]) for i in range(0, len(Xs), chunk)])

# ---------------------------------------------------------------- A
m = RiskModel.load(a.model)
va, te = slice(int(n * .6), int(n * .8)), slice(int(n * .8), n)
print("scoring validation and test periods ...", flush=True)
rv, rt = score(m, X[va]), score(m, X[te])
print("\n=== A) recall at FIXED false-alarm rates (test period, label = attack IP) ===")
rows = []
for fpr in (0.001, 0.005, 0.01, 0.02, 0.05):
    thr = np.quantile(rt[legit[te]], 1 - fpr)
    rows.append((f"{fpr:.1%}", round(float(thr), 1), round(float((rt[atk[te]] >= thr).mean() * 100), 2)))
print(pd.DataFrame(rows, columns=["legit logins flagged", "risk threshold", "attack-IP logins caught %"]).to_string(index=False))

print(f"\n=== A2) do the saved tier thresholds hold over time? (targets per head: Medium+ 2%, High 0.5% of normal logins) ===")
def flagged(r, mask):
    l = r[mask]; return (round(float((l >= m.medium_thr).mean() * 100), 2), round(float((l >= m.high_thr).mean() * 100), 3)) if len(l) else (None, None)
print("validation period (thresholds were tuned here): Medium+ %.2f%%  High %.3f%%" % flagged(rv, legit[va]))
k = np.array_split(np.arange(len(rt)), 6)
out = []
for i, ix in enumerate(k):
    mm, hh = flagged(rt[ix], legit[te][ix]); out.append((f"test part {i+1}/6", mm, hh))
print(pd.DataFrame(out, columns=["period", "Medium+ legit flagged %", "High legit flagged %"]).to_string(index=False))
print("legit-login risk percentiles  validation: p50=%.1f p90=%.1f p99=%.1f | test: p50=%.1f p90=%.1f p99=%.1f" % (
    *np.percentile(rv[legit[va]], [50, 90, 99]), *np.percentile(rt[legit[te]], [50, 90, 99])))

# ---------------------------------------------------------------- B
print("\n=== B) account takeover: do user-deviation signals detect it? (no training involved) ===")
last_ato = np.flatnonzero(ato).max()
period = np.arange(0, last_ato + 1)                      # only the time range in which takeovers exist
warm = (Xdf["history_count_log"].to_numpy()[period] >= np.log1p(3))
pos = period[ato[period] & warm]; neg_all = period[legit[period] & warm]
print(f"takeover logins in range: {int(ato[period].sum())}  with >=3 prior logins (deviation is defined): {len(pos)}  "
      f"| comparison legit logins: {len(neg_all):,} (sampled {min(a.neg_sample, len(neg_all)):,})")
neg = rng.choice(neg_all, min(a.neg_sample, len(neg_all)), replace=False)

def auc_ci(v_pos, v_neg, B=500):
    s = np.sort(v_neg)
    pct = (np.searchsorted(s, v_pos, "left") + np.searchsorted(s, v_pos, "right")) / 2 / len(s)   # P(neg < pos), ties = 0.5
    boots = [rng.choice(pct, len(pct)).mean() for _ in range(B)]
    return pct.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

cols = {f: Xdf[f].fillna(0).to_numpy() for f in FEATURES}
deviation = sum(cols[f] for f in ("new_country", "new_asn", "new_device", "new_browser", "new_os", "new_ip"))
rows = []
for name, v in [("COMPOSITE deviation = sum of new_* flags", deviation)] + [(f, cols[f]) for f in FEATURES]:
    arr = v if isinstance(v, np.ndarray) else np.asarray(v)
    A, lo, hi = auc_ci(arr[pos], arr[neg]); rows.append((name, A, lo, hi))
t = pd.DataFrame(rows, columns=["signal", "AUC", "ci_low", "ci_high"])
t["clear_signal"] = np.where((t.ci_low > .5) | (t.ci_high < .5), "yes", "no")
print(t.reindex((t.AUC - .5).abs().sort_values(ascending=False).index).round(3).head(14).to_string(index=False))
for f in ("new_country", "new_asn", "new_device", "new_ip"):
    print(f"share with {f}=1:  takeover logins {cols[f][pos].mean():.0%}   vs normal logins {cols[f][neg].mean():.0%}")
print("\nHow to read it: AUC 0.5 = no signal, 1.0 = perfect. If the confidence interval excludes 0.5, the signal is real "
      "even with this few cases. 'clear_signal' = yes means exactly that.")