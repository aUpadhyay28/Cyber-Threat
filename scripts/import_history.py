"""Teach the live system about your site's HISTORIC logins (so users already have a 'normal' profile).
   python scripts/import_history.py --csv old_logins.csv --state data/state.json.gz
CSV needs the same columns as the RBA dataset (see riskengine/config.py RAW_COLUMNS); attack columns optional.

This is the trusted offline path: the timestamps come from a file the operator controls, not from a
caller, so the forward-jump guard that protects the live service is switched off here. A quiet month
in your own history is a legitimate gap; the same jump arriving over HTTP would not be."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
import pandas as pd
from riskengine.data import normalize_chunk
from riskengine.features import FeatureState
from riskengine.persist import load_state, save_state

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True)
ap.add_argument("--state", default="data/state.json.gz")
a = ap.parse_args()
prep = normalize_chunk(pd.read_csv(a.csv)).sort_values("ts_ms", kind="stable")
if pathlib.Path(a.state).exists():
    st = load_state(a.state, max_forward_jump_ms=None)
else:
    st = FeatureState(max_forward_jump_ms=None)
st.max_forward_jump_ms = None
for r in prep.to_dict("records"):
    # historic logins flagged as attacks must NOT shape a user's normal profile
    st.update(r, trusted=bool(r["success"]) and not r["attack_ip"] and not r["ato"], counters=True)
save_state(st, a.state)
print(f"imported {len(prep):,} logins; {len(st.users):,} users now have history -> {a.state}")
