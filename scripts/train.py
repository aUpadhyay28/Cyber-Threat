"""Step 2: train on the prepared data (FULL dataset = omit --max-rows).
   python scripts/train.py --data data/prepared --out models [--feedback-db data/risk.db]"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, glob, json
import pandas as pd
from riskengine.training import train

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/prepared"); ap.add_argument("--out", default="models")
ap.add_argument("--max-rows", type=int, default=None); ap.add_argument("--feedback-db", default=None)
ap.add_argument("--bulk-threshold", type=int, default=20_000, help="users with more logins than this are pseudo-accounts, excluded from training")
a = ap.parse_args()
parts = sorted(glob.glob(f"{a.data}/part_*.parquet"))
if not parts: raise SystemExit(f"No parquet parts in {a.data} - run scripts/prepare_data.py first")
df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
if a.max_rows: df = df.iloc[: a.max_rows]
fb = None
if a.feedback_db:
    from riskengine.store import Store
    fb = Store(a.feedback_db).training_rows()
print(json.dumps(train(df, a.out, feedback_rows=fb, bulk_threshold=a.bulk_threshold), indent=2))