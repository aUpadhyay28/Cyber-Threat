"""Train and calibrate from the exported stratified sample instead of the 31 M-row file.

   python scripts/train_from_sample.py --sample data/sample.parquet --out models_sample

The sample (scripts/export_sample.py) holds features that were built on the FULL history, plus a
sample_weight telling you how many real logins each row stands for. Its base rates are deliberately
distorted - roughly a third of the rows are attack-IP logins where the population is about a tenth -
so fitting on it unweighted produces a model whose percentile thresholds do not mean what they say.
Everything here passes sample_weight into every fit, every metric and both calibration reference
distributions, so an 11 MB file yields population-calibrated thresholds.

What this CANNOT do, and why you still need the full file:
  * the sample has no raw columns, so feature definitions cannot be changed or re-derived from it
  * the parity guarantee (tests/test_parity.py) needs prepared rows, not features
  * the sample's per-stratum caps bound how finely rare cases can be studied
Use this for model and threshold development; retrain on the full data before shipping.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json

import numpy as np
import pandas as pd

from riskengine.features import FEATURES
from riskengine.training import train

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="data/sample.parquet")
ap.add_argument("--out", default="models_sample")
ap.add_argument("--unweighted", action="store_true",
                help="ignore sample_weight - only to demonstrate how far the calibration drifts")
a = ap.parse_args()

d = pd.read_parquet(a.sample)
missing = [c for c in FEATURES if c not in d.columns]
if missing:
    raise SystemExit(f"the sample is missing {len(missing)} feature columns, e.g. {missing[:5]}.\n"
                     f"Re-export it with scripts/export_sample.py from the current code - a sample "
                     f"built by an older version cannot be used, for the same reason an old model "
                     f"cannot (the feature list must match exactly).")
for c in ("sample_weight", "attack_ip", "ato", "success_flag", "is_bulk", "row_frac", "user"):
    if c not in d.columns:
        raise SystemExit(f"the sample is missing the '{c}' column; re-export it with scripts/export_sample.py")

# train() splits by position, so the rows must be in the same time order the full data had.
d = d.sort_values("row_frac", kind="stable").reset_index(drop=True)
w = np.ones(len(d)) if a.unweighted else d["sample_weight"].to_numpy(dtype=np.float64)

print(f"{len(d):,} sampled rows standing for {d.sample_weight.sum():,.0f} logins")
print(f"attack-IP rate: {d.attack_ip.mean():.1%} in the sample, "
      f"{np.average(d.attack_ip, weights=d.sample_weight):.1%} in the population")
if a.unweighted:
    print("WARNING: --unweighted, so thresholds will be calibrated to the sample's base rates, not the site's")

report = train(None, a.out, precomputed={
    "X": d[list(FEATURES)].to_numpy(np.float32),
    "attack_ip": d["attack_ip"].to_numpy(),
    "ato": d["ato"].to_numpy(),
    "success": d["success_flag"].to_numpy(),
    "user": d["user"].to_numpy(),
    "is_bulk": d["is_bulk"].to_numpy(),
    "sample_weight": w,
})
print(json.dumps(report, indent=2))
