"""Feature-group ablation on the exported sample (non-bulk users, weighted back to the real population).
   python scripts/ablation.py --sample data/sample.parquet"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

ap = argparse.ArgumentParser(); ap.add_argument("--sample", default="data/sample.parquet")
ap.add_argument("--include-bulk", action="store_true"); a = ap.parse_args()
d = pd.read_parquet(a.sample)
if not a.include_bulk: d = d[~d.is_bulk]
CAT = [f"{k}_{c}" for k in ("new", "freq") for c in ("ip", "country", "asn", "browser", "os", "device")]
G = {
    "profile_categorical": CAT + ["history_count_log", "user_fail_prior_log"],
    "profile_timing": ["hour_dev", "hour_conc", "since_last_log"],
    "ip_asn_identity": ["ip_distinct_users_log", "asn_distinct_users_log"],
    "recent_windows": ["ip_n_1h_log", "ip_n_24h_log", "ip_fail_24h_log", "ip_fail_rate_24h", "asn_n_24h_log", "asn_fail_rate_24h"],
    "string_rarity": ["browser_global_freq", "os_global_freq"],
    "success_and_rtt": ["success", "log_rtt"],
}
ALL = sum(G.values(), [])
tr, te = d[d.row_frac < .6], d[d.row_frac >= .8]
y_tr, y_te = tr.attack_ip.to_numpy(), te.attack_ip.to_numpy()
w_tr, w_te = tr.sample_weight.to_numpy(), te.sample_weight.to_numpy()

def run(cols, seeds=(0, 1, 2)):
    out = []
    for sd in seeds:
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=.1, random_state=sd).fit(tr[cols], y_tr, sample_weight=w_tr)
        p = m.predict_proba(te[cols])[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p, sample_weight=w_te)       # ties handled: one point per distinct score
        rec = lambda f, _t=tpr, _f=fpr: _t[_f <= f].max() * 100   # bind this iteration's curve
        out.append((roc_auc_score(y_te, p, sample_weight=w_te), average_precision_score(y_te, p, sample_weight=w_te), rec(.005), rec(.05)))
    o = np.array(out)
    return tuple(o.mean(0)) + (o[:, 0].std(),)

rows = [("ALL features", *run(ALL))]
for g in G:
    rows.append((f"drop {g}", *run([c for c in ALL if c not in G[g]])))
for g in G:
    rows.append((f"ONLY {g}", *run(G[g])))
rows.append(("drop recent_windows + profile_timing + rtt  (timestamp-free)", *run([c for c in ALL if c not in G["recent_windows"] + G["profile_timing"] + ["log_rtt"]])))
for f in G["recent_windows"]:
    rows.append((f"ONLY window feature: {f}", *run([f])))
rows.append(("windows WITHOUT 1-hour feature", *run([c for c in ALL if c != "ip_n_1h_log"])))
t = pd.DataFrame(rows, columns=["feature set", "ROC-AUC", "PR-AUC", "recall@0.5%FPR", "recall@5%FPR", "AUC std(3 seeds)"])
base = (w_te[y_te].sum() / w_te.sum())
print(f"test period: {len(te):,} sampled rows | weighted attack-IP rate {base:.1%} (random PR-AUC = {base:.3f})")
print(t.round(3).to_string(index=False))
