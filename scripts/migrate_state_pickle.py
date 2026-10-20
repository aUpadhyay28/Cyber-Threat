"""Convert an old pickled state.pkl into the pickle-free format the service now loads.

   python scripts/migrate_state_pickle.py --in data/state.pkl --out data/state.json.gz

This is the ONLY file in the project that imports pickle for reading. Loading a pickle runs whatever
code is inside it, so run this only on a file you produced yourself and have kept under your control.
If you cannot vouch for the file, delete it and rebuild the state with scripts/import_history.py.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--out", dest="dst", required=True)
ap.add_argument("--i-trust-this-file", action="store_true",
                help="required: confirms you know unpickling the input executes code from it")
a = ap.parse_args()

if not a.i_trust_this_file:
    raise SystemExit(
        "Refusing to unpickle without --i-trust-this-file.\n"
        "Reading a pickle executes code embedded in it. Pass the flag only for a state file you\n"
        "created and have kept under your own control; otherwise rebuild from scripts/import_history.py.")

import pickle                                            # noqa: E402  (deliberate, see docstring)
from riskengine.features import FeatureState
from riskengine.persist import save_state

with open(a.src, "rb") as fh:
    st = pickle.load(fh)
if not isinstance(st, FeatureState):
    raise SystemExit(f"{a.src} does not contain a FeatureState (got {type(st).__name__})")
save_state(st, a.dst)
print(f"converted {a.src} -> {a.dst}: {len(st.users):,} users, {st.glob_total:,} logins recorded.\n"
      f"Point RISK_STATE_PATH at the new file and delete the pickle.")
