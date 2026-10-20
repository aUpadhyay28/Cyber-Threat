"""Corrected takeover analysis. Fixes the first version's weak comparison group:
  * negatives = legitimate, SUCCESSFUL logins (a takeover is always a successful login)
  * negatives exclude 'bulk' pseudo-users (accounts with an absurd number of logins)
  * also prints the top users table (is one placeholder ID owning most rows?)
   python scripts/takeover_analysis.py --data data/prepared"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, glob
import numpy as np, pandas as pd
from riskengine.features import build_features, FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/prepared"); ap.add_argument("--neg-sample", type=int, default=300_000)
ap.add_argument("--bulk-threshold", type=int, default=20_000, help="users with more logins than this are treated as pseudo-users")
ap.add_argument("--max-rows", type=int, default=None)
a = ap.parse_args(); rng = np.random.default_rng(0)

df = pd.concat((pd.read_parquet(p, columns=["ts_ms", "user", "ip", "country", "asn", "browser", "os", "device", "rtt", "success", "attack_ip", "ato"])
                for p in sorted(glob.glob(f"{a.data}/part_*.parquet"))), ignore_index=True)
df = df.sort_values("ts_ms", kind="stable").reset_index(drop=True)
if a.max_rows: df = df.iloc[: a.max_rows]
n = len(df)

print("=== 1) top users by number of logins (is one placeholder ID owning the data?) ===")
g = df.groupby("user").agg(logins=("success", "size"), success_rate=("success", "mean"),
                           attack_ip_rate=("attack_ip", "mean"), takeovers=("ato", "sum")).sort_values("logins", ascending=False)
g["share_of_all_rows_%"] = (g.logins / n * 100).round(2)
print(g.head(8).round(3).to_string())
print(f"typical user: median {int(g.logins.median())} logins, mean {g.logins.mean():.1f}; distinct users {len(g):,}")
bulk_users = g.index[g.logins > a.bulk_threshold]
print(f"users above {a.bulk_threshold:,} logins: {len(bulk_users)}  -> {g.loc[bulk_users].logins.sum()/n:.1%} of all rows")
is_bulk = df.user.isin(bulk_users).to_numpy()

print("\nbuilding features ...", flush=True)
Xdf = build_features(df)
atk, ato, ok = df.attack_ip.to_numpy(), df.ato.to_numpy(), df.success.to_numpy()
del df
legit = ~(atk | ato)
last = np.flatnonzero(ato).max(); period = np.arange(last + 1)
warm = Xdf["history_count_log"].to_numpy()[period] >= np.log1p(3)
pos = period[ato[period] & warm]
neg_all = period[legit[period] & ok[period] & warm & ~is_bulk[period]]
print("\n=== 2) takeover logins vs ORDINARY users' successful logins ===")
print(f"takeover logins: {int(ato[period].sum())} total, {len(pos)} with >=3 prior logins ({len(pos)/ato[period].sum():.0%}); "
      f"comparison logins: {len(neg_all):,} (sampled {min(a.neg_sample, len(neg_all)):,})")
neg = rng.choice(neg_all, min(a.neg_sample, len(neg_all)), replace=False)

def auc_ci(vp, vn, B=500):
    s = np.sort(vn); pct = (np.searchsorted(s, vp, "left") + np.searchsorted(s, vp, "right")) / 2 / len(s)
    boots = [rng.choice(pct, len(pct)).mean() for _ in range(B)]
    return pct.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

cols = {f: Xdf[f].fillna(0).to_numpy() for f in FEATURES if f != "success"}
cols["COMPOSITE: sum of new_* flags"] = sum(cols[f] for f in ("new_country", "new_asn", "new_device", "new_browser", "new_os", "new_ip"))
rows = [(k, *auc_ci(v[pos], v[neg])) for k, v in cols.items()]
t = pd.DataFrame(rows, columns=["signal", "AUC", "ci_low", "ci_high"])
t["clear_signal"] = np.where((t.ci_low > .5) | (t.ci_high < .5), "yes", "no")
print(t.reindex((t.AUC - .5).abs().sort_values(ascending=False).index).round(3).head(14).to_string(index=False))

def wilson(k, m, z=1.96):
    if m == 0: return (np.nan, np.nan)
    p = k / m; d = 1 + z * z / m; c = p + z * z / (2 * m); r = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
    return (c - r) / d, (c + r) / d

print("\n=== 3) simple one-line rules (no model, no tuning) ===")
rules = {"new_country": cols["new_country"] > 0, "new_asn": cols["new_asn"] > 0,
         "new_country OR new_asn": (cols["new_country"] + cols["new_asn"]) > 0,
         "new_country OR new_asn OR new_device": (cols["new_country"] + cols["new_asn"] + cols["new_device"]) > 0}
out = []
for k, v in rules.items():
    kp, kn = int(v[pos].sum()), int(v[neg].sum()); lo, hi = wilson(kp, len(pos))
    out.append((k, f"{kp}/{len(pos)} = {kp/len(pos):.0%}  (95% CI {lo:.0%}-{hi:.0%})", f"{kn/len(neg):.2%}"))
print(pd.DataFrame(out, columns=["rule", "takeover logins caught (warm accounts)", "normal logins flagged"]).to_string(index=False))
print(f"\nLimit: rules need >=3 prior logins, so at most {len(pos)}/{int(ato[period].sum())} = {len(pos)/ato[period].sum():.0%} "
      "of ALL takeover logins can be caught this way; the rest hit accounts with little history.")
