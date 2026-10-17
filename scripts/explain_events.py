"""Shows WHY the model scored one logged login the way it did: the exact feature values the service computed,
and how the IP-head score changes if each feature is neutralised or set to a 'typical busy network' value.
   python scripts/explain_event.py --model models_v4/latest.joblib --db data/risk.db --user-like "victim-%-99"
   (or --event-id <id>; default = the most recent event)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, math, sqlite3
import numpy as np
from riskengine.features import FEATURES
from riskengine.modeling import RiskModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/latest.joblib"); ap.add_argument("--db", default="data/risk.db")
ap.add_argument("--event-id"); ap.add_argument("--user-like")
a = ap.parse_args()
db = sqlite3.connect(a.db); db.row_factory = sqlite3.Row
if a.event_id: q, p = "SELECT * FROM events WHERE event_id=?", (a.event_id,)
elif a.user_like: q, p = "SELECT * FROM events WHERE user_id LIKE ? ORDER BY created_at DESC LIMIT 1", (a.user_like,)
else: q, p = "SELECT * FROM events ORDER BY created_at DESC LIMIT 1", ()
row = db.execute(q, p).fetchone()
if row is None: raise SystemExit("no matching event in the database (is --db the right file? did the service log it?)")
f = {k: (float("nan") if v is None else v) for k, v in json.loads(row["features_json"]).items()}
m = RiskModel.load(a.model)
def ip_risk(feats):
    x = np.array([[feats[k] for k in FEATURES]], np.float32); return float(m.heads(x)[0][0])
base = ip_risk(f)
print(f"event {row['event_id']}  user={row['user_id']}  logged tier={row['tier']}  logged risk={row['risk']}")
print(f"IP-head risk recomputed from the stored features: {base:.1f}   (High >= {m.high_thr}, Medium >= {m.medium_thr})\n")
cnt = lambda v: f"{math.expm1(v):,.0f}" if v == v else "n/a"
print("what the service computed for this login (IP-head inputs):")
for k in m.ip_feats:
    v = f[k]; shown = cnt(v) if k.endswith("_log") else (f"{v:.2f}" if v == v else "n/a")
    print(f"  {k:20s} {shown}")
print("\nsensitivity: IP-head risk if ONE feature is changed (largest effects first)")
rows = []
for k in m.ip_feats:
    g = dict(f); g[k] = float("nan") if k in ("hour_dev", "hour_conc", "since_last_log") else 0.0
    rows.append((ip_risk(g) - base, f"set {k} to 0/none", ip_risk(g)))
for k, val in (("asn_n_24h_log", math.log1p(15000)), ("ip_n_24h_log", math.log1p(500)), ("ip_fail_24h_log", math.log1p(500))):
    g = dict(f); g[k] = val; rows.append((ip_risk(g) - base, f"set {k} to {math.expm1(val):,.0f}", ip_risk(g)))
for d, label, r in sorted(rows, key=lambda t: -abs(t[0]))[:8]:
    print(f"  {label:34s} -> {r:5.1f}  ({d:+.1f})")