"""Step 2: train on the prepared data (FULL dataset = omit --max-rows).
   python scripts/train.py --data data/prepared --out models [--feedback-db data/risk.db]
Tip: close the scoring service, browsers and other big programs first - the full file needs ~8-10 GB of free RAM."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json
from riskengine.data import load_prepared
from riskengine.training import train

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/prepared"); ap.add_argument("--out", default="models")
ap.add_argument("--max-rows", type=int, default=None); ap.add_argument("--feedback-db", default=None)
ap.add_argument("--bulk-threshold", type=int, default=20_000, help="users with more logins than this are pseudo-accounts, excluded from training")
a = ap.parse_args()
fb = None
if a.feedback_db:
    from riskengine.store import Store
    fb = Store(a.feedback_db).training_rows()
# pass a loader, not the table: that way nothing outside train() keeps the raw data alive after features are built
print(json.dumps(train(lambda: load_prepared(a.data, a.max_rows), a.out, feedback_rows=fb, bulk_threshold=a.bulk_threshold), indent=2))