"""Teach the live system about your site's HISTORIC logins (so users already have a 'normal' profile).
   python scripts/import_history.py --csv old_logins.csv --state data/state.pkl
CSV needs the same columns as the RBA dataset (see riskengine/config.py RAW_COLUMNS); attack columns optional."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, pickle
import pandas as pd
from riskengine.data import normalize_chunk
from riskengine.features import FeatureState

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--state", default="data/state.pkl")
a = ap.parse_args()
prep = normalize_chunk(pd.read_csv(a.csv)).sort_values("ts_ms", kind="stable")
st = pickle.load(open(a.state, "rb")) if pathlib.Path(a.state).exists() else FeatureState()
for r in prep.to_dict("records"):
    # historic logins flagged as attacks must NOT shape a user's normal profile
    st.update(r, trusted=bool(r["success"]) and not r["attack_ip"] and not r["ato"], counters=True)
pathlib.Path(a.state).parent.mkdir(parents=True, exist_ok=True)
pickle.dump(st, open(a.state, "wb"))
print(f"imported {len(prep):,} logins; {len(st.users):,} users now have history -> {a.state}")
