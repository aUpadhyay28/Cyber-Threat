"""Build features on the FULL data, then save a small stratified sample (a few MB) that can be shared
for model development. Features need the whole history, so we sample AFTER building them.
Contains only pseudonymised dataset columns (hashed ids, no raw IPs).
   python scripts/export_sample.py --data data/prepared --out data/sample.parquet"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, glob, os
import numpy as np, pandas as pd
from riskengine.features import build_features

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/prepared"); ap.add_argument("--out", default="data/sample.parquet")
ap.add_argument("--n-ordinary", type=int, default=150_000, help="legit successful logins of ordinary users")
ap.add_argument("--n-attack", type=int, default=100_000, help="attack-IP logins")
ap.add_argument("--n-failed", type=int, default=30_000, help="legit failed logins of ordinary users")
ap.add_argument("--n-bulk", type=int, default=20_000, help="logins of the huge pseudo-accounts")
ap.add_argument("--bulk-threshold", type=int, default=20_000)
ap.add_argument("--max-rows", type=int, default=None)
a = ap.parse_args(); rng = np.random.default_rng(0)

df = pd.concat((pd.read_parquet(p) for p in sorted(glob.glob(f"{a.data}/part_*.parquet"))), ignore_index=True)
df = df.sort_values("ts_ms", kind="stable").reset_index(drop=True)
if a.max_rows: df = df.iloc[: a.max_rows]
n = len(df)
cnt = df.groupby("user")["success"].transform("size").to_numpy()
bulk = cnt > a.bulk_threshold
ato, atk, ok = df.ato.to_numpy(), df.attack_ip.to_numpy(), df.success.to_numpy()
ato_users = df.user[ato].unique()
print(f"{n:,} logins; bulk pseudo-account rows: {bulk.mean():.1%}; building features ...", flush=True)
X = build_features(df)

strata = {
    "takeover": ato,
    "ato_user_other": df.user.isin(ato_users).to_numpy() & ~ato,
    "ordinary_success": ~bulk & ~atk & ~ato & ok,
    "attack_ip": ~bulk & atk & ~ato,
    "ordinary_failed": ~bulk & ~atk & ~ato & ~ok,
    "bulk_pseudo_user": bulk & ~ato,
}
want = {"takeover": None, "ato_user_other": None, "ordinary_success": a.n_ordinary,
        "attack_ip": a.n_attack, "ordinary_failed": a.n_failed, "bulk_pseudo_user": a.n_bulk}
# The strata above overlap (a takeover user's other logins are also ordinary successes), so each row
# is assigned to the FIRST stratum it matches and later strata see only what is left. Earlier
# versions subtracted the rows already *sampled* instead of the rows already *claimed*; that happened
# to be correct only because every overlapping stratum was sampled in full, and would have silently
# double-counted rows - inflating sample_weight - the moment one of them was capped.
keep, weight, label = [], [], []
claimed = np.zeros(n, dtype=bool)
for name, mask in strata.items():
    idx = np.flatnonzero(mask & ~claimed)
    claimed |= mask
    k = len(idx) if want[name] is None else min(want[name], len(idx))
    pick = idx if k == len(idx) else np.sort(rng.choice(idx, k, replace=False))
    keep.append(pick); weight.append(np.full(k, len(idx) / max(k, 1))); label.extend([name] * k)
    print(f"  {name:18s} population {len(idx):>11,}  sampled {k:>8,}  (each row stands for {len(idx)/max(k,1):.1f})")
covered = int(claimed.sum())
if covered != n:
    print(f"  NOTE: {n - covered:,} rows ({(n-covered)/n:.2%}) match no stratum and are not represented; "
          f"weighted totals will fall short of {n:,} by that much.")
keep = np.concatenate(keep)
out = X.iloc[keep].copy()
out["stratum"] = label
out["sample_weight"] = np.concatenate(weight).astype("float32")
out["attack_ip"], out["ato"], out["success_flag"] = atk[keep], ato[keep], ok[keep]
out["user"] = df.user.to_numpy()[keep]; out["ts_ms"] = df.ts_ms.to_numpy()[keep]
out["row_frac"] = (keep / n).astype("float32")                  # position in time: 0 = first login, 1 = last
out["is_bulk"] = bulk[keep]
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.sort_values("ts_ms").to_parquet(a.out, compression="zstd")
print(f"\nsaved {len(out):,} rows -> {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")
