"""Watch the live system for the two failures that make a risk engine quietly useless.

   python scripts/monitor_drift.py --db data/risk.db --model models/latest.joblib [--json]

1. THRESHOLD DRIFT. The tiers are budgets: Medium+ should flag about 2% of normal logins and High
   about 0.5%. Those percentiles were fixed against a reference distribution at training time. As
   traffic changes, the realised rate moves away from the budget - a model that flags 12% of logins
   is not protecting anyone, it is training your analysts to ignore the queue.
2. FEATURE DRIFT. The project already showed that all-time counters stop meaning the same thing over
   time (scripts/diagnose.py, section 4). The same can happen to any feature once live. This
   compares each feature's live distribution against the window the model was trained on.

Run it on a schedule and alert when `alerts` is non-empty. It reads only the event log, so it costs
the scoring service nothing.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np

from riskengine import config as C
from riskengine.features import FEATURES
from riskengine.modeling import RiskModel

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/risk.db")
ap.add_argument("--model", default="models/latest.joblib")
ap.add_argument("--window-hours", type=float, default=24.0, help="how much recent traffic to judge")
ap.add_argument("--baseline-hours", type=float, default=24.0 * 7, help="the earlier window to compare against")
ap.add_argument("--tolerance", type=float, default=2.0, help="alert when the flag rate is this many times the budget")
ap.add_argument("--json", action="store_true")
a = ap.parse_args()

db = sqlite3.connect(a.db); db.row_factory = sqlite3.Row
now = datetime.now(timezone.utc)
recent_from = (now - timedelta(hours=a.window_hours)).isoformat(timespec="seconds")
base_from = (now - timedelta(hours=a.baseline_hours)).isoformat(timespec="seconds")


def rows(since, until=None):
    q = "SELECT tier, risk, features_json, model_version FROM events WHERE created_at >= ?"
    p = [since]
    if until:
        q += " AND created_at < ?"; p.append(until)
    return db.execute(q, p).fetchall()


recent = rows(recent_from)
baseline = rows(base_from, recent_from)
out = {"generated_at": now.isoformat(timespec="seconds"),
       "window_hours": a.window_hours, "events_in_window": len(recent),
       "events_in_baseline": len(baseline), "alerts": []}

if not recent:
    out["alerts"].append({"check": "no_traffic", "detail": f"no events logged in the last {a.window_hours} h"})
    print(json.dumps(out, indent=2) if a.json else f"no events in the last {a.window_hours} h"); raise SystemExit

# ---- 1) flag rates against the budgets ---------------------------------------------------------
tiers = [r["tier"] for r in recent]
med_rate = sum(t in ("Medium", "High") for t in tiers) / len(tiers)
high_rate = sum(t == "High" for t in tiers) / len(tiers)
out["flag_rates"] = {
    "medium_plus": {"observed_pct": round(med_rate * 100, 3), "budget_pct": round(C.HEAD_MEDIUM_BUDGET * 100, 3)},
    "high": {"observed_pct": round(high_rate * 100, 3), "budget_pct": round(C.HEAD_HIGH_BUDGET * 100, 3)},
}
# The budget is per head over NORMAL logins; live traffic also contains real attacks, so exceeding it
# a little is expected. `tolerance` is the multiple at which it stops looking like attack volume.
for name, observed, budget in (("medium_plus", med_rate, C.HEAD_MEDIUM_BUDGET),
                               ("high", high_rate, C.HEAD_HIGH_BUDGET)):
    if observed > budget * a.tolerance:
        out["alerts"].append({"check": f"flag_rate_{name}", "observed_pct": round(observed * 100, 3),
                              "budget_pct": round(budget * 100, 3),
                              "detail": f"{name} is flagging {observed/budget:.1f}x its budget; "
                                        f"either you are under attack or the thresholds need recalibrating"})

# ---- 2) which model made these decisions -------------------------------------------------------
versions = sorted({r["model_version"] for r in recent if r["model_version"]})
live = RiskModel.load(a.model)
out["model"] = {"file_version": live.meta.get("version"), "versions_seen_in_window": versions}
if versions and live.meta.get("version") not in versions:
    out["alerts"].append({"check": "model_version_mismatch",
                          "detail": f"the model on disk is {live.meta.get('version')} but recent decisions were "
                                    f"made by {versions} - the service has not reloaded, or you are looking at "
                                    f"the wrong model file"})

# ---- 3) feature drift, recent window vs the earlier one ----------------------------------------
def matrix(rs):
    if not rs:
        return None
    m = np.full((len(rs), len(FEATURES)), np.nan)
    for i, r in enumerate(rs):
        f = json.loads(r["features_json"])
        for j, name in enumerate(FEATURES):
            v = f.get(name)
            if v is not None:
                m[i, j] = v
    return m


A, B = matrix(baseline), matrix(recent)
if A is not None and len(A) >= 50 and len(B) >= 50:
    drift = []
    for j, name in enumerate(FEATURES):
        x, y = A[:, j][~np.isnan(A[:, j])], B[:, j][~np.isnan(B[:, j])]
        if len(x) < 20 or len(y) < 20:
            continue
        # Population-stability index on deciles of the baseline: the standard "has this moved?" number.
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        px = np.histogram(x, bins=edges)[0] / len(x)
        py = np.histogram(y, bins=edges)[0] / len(y)
        eps = 1e-6
        psi = float(np.sum((py - px) * np.log((py + eps) / (px + eps))))
        drift.append({"feature": name, "psi": round(psi, 3),
                      "median_baseline": round(float(np.median(x)), 4),
                      "median_recent": round(float(np.median(y)), 4)})
    drift.sort(key=lambda d: -d["psi"])
    out["feature_drift_top10"] = drift[:10]
    # PSI > 0.25 is the usual "the distribution has genuinely moved" threshold.
    for d in drift:
        if d["psi"] > 0.25:
            out["alerts"].append({"check": "feature_drift", "feature": d["feature"], "psi": d["psi"],
                                  "detail": f"{d['feature']} has moved (PSI {d['psi']}): median "
                                            f"{d['median_baseline']} -> {d['median_recent']}"})
else:
    out["feature_drift_top10"] = []
    out["note"] = "not enough history for a drift comparison yet (need 50+ events in each window)"

if a.json:
    print(json.dumps(out, indent=2))
else:
    print(f"events: {len(recent):,} in the last {a.window_hours} h (baseline {len(baseline):,})")
    print(f"Medium+ flagged {out['flag_rates']['medium_plus']['observed_pct']}% "
          f"(budget {out['flag_rates']['medium_plus']['budget_pct']}%)")
    print(f"High    flagged {out['flag_rates']['high']['observed_pct']}% "
          f"(budget {out['flag_rates']['high']['budget_pct']}%)")
    print(f"model on disk: {out['model']['file_version']}; decisions by: {out['model']['versions_seen_in_window'] or '-'}")
    if out["feature_drift_top10"]:
        print("\nlargest feature movements (PSI, >0.25 is a real shift):")
        for d in out["feature_drift_top10"][:5]:
            print(f"  {d['feature']:24s} psi={d['psi']:<7} {d['median_baseline']} -> {d['median_recent']}")
    print(f"\n{len(out['alerts'])} alert(s)" + (":" if out["alerts"] else ""))
    for al in out["alerts"]:
        print(f"  [{al['check']}] {al['detail']}")
raise SystemExit(1 if out["alerts"] else 0)
