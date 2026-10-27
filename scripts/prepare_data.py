"""Step 1 (run once): convert the big raw CSV into compact parquet parts, chunk by chunk.
   python scripts/prepare_data.py --csv rba-dataset.csv --out data/prepared [--max-rows 3000000]"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
from pathlib import Path
import pandas as pd
from riskengine.config import RAW_COLUMNS
from riskengine.data import normalize_chunk, write_prepared_fingerprint

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--out", default="data/prepared")
ap.add_argument("--chunk", type=int, default=1_000_000); ap.add_argument("--max-rows", type=int, default=None)
ap.add_argument("--inspect", action="store_true", help="print the first rows / columns and exit")
a = ap.parse_args()
if a.inspect:
    d = pd.read_csv(a.csv, nrows=5); print(d.T); raise SystemExit
head = pd.read_csv(a.csv, nrows=1)
missing = [v for v in RAW_COLUMNS.values() if v not in head.columns]
if missing:
    raise SystemExit(f"Columns missing: {missing}\nYour columns: {list(head.columns)}\n-> edit RAW_COLUMNS in riskengine/config.py")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
write_prepared_fingerprint(out)          # which RISK_HASH_KEY produced these parts
total = 0
for i, ch in enumerate(pd.read_csv(a.csv, chunksize=a.chunk, usecols=list(RAW_COLUMNS.values()))):
    if a.max_rows and total >= a.max_rows: break
    if a.max_rows: ch = ch.iloc[: a.max_rows - total]
    normalize_chunk(ch).to_parquet(out / f"part_{i:04d}.parquet"); total += len(ch)
    print(f"part {i}: {total:,} rows", flush=True)
print("done ->", out)
