"""Raw login rows -> compact, hashed 'prepared' frame. The SAME function is used for the
big CSV (training) and for single live events (serving), so the two can never disagree."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import RAW_COLUMNS as RC, CATS
from .hashing import hash_key, key_fingerprint, check_fingerprint

_EPOCH = pd.Timestamp("1970-01-01")


def _hash(s: pd.Series) -> np.ndarray:
    # Keyed SipHash: deterministic for a given RISK_HASH_KEY, not reversible without it (riskengine/hashing.py).
    return pd.util.hash_array(s.to_numpy(dtype=object), hash_key=hash_key())


def _to_ms(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):                           # already epoch milliseconds
        return s.astype("int64").to_numpy()
    dt = pd.to_datetime(s, utc=True).dt.tz_localize(None)
    return ((dt - _EPOCH) // pd.Timedelta(milliseconds=1)).astype("int64").to_numpy()


def _to_bool(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy()
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"]).to_numpy()


def _as_text(col: pd.Series) -> pd.Series:
    return col.astype(object).where(col.notna(), "NA").astype(str)


def normalize_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    out = {"ts_ms": _to_ms(raw[RC["ts"]]), "user": _hash(_as_text(raw[RC["user"]]))}
    for c in CATS:
        col = raw[RC[c]]
        if c == "asn":
            col = pd.to_numeric(col, errors="coerce").astype("Int64").astype(object)
            col = pd.Series(col, index=raw.index)
        out[c] = _hash(_as_text(col))
    out["rtt"] = (pd.to_numeric(raw[RC["rtt"]], errors="coerce").astype("float32").to_numpy()
                  if RC["rtt"] in raw else np.full(len(raw), np.nan, "float32"))
    out["success"] = _to_bool(raw[RC["success"]]) if RC["success"] in raw else np.ones(len(raw), bool)
    out["attack_ip"] = _to_bool(raw[RC["attack_ip"]]) if RC["attack_ip"] in raw else np.zeros(len(raw), bool)
    out["ato"] = _to_bool(raw[RC["ato"]]) if RC["ato"] in raw else np.zeros(len(raw), bool)
    return pd.DataFrame(out)


def event_to_row_via_frame(ev: dict) -> dict:
    """Reference implementation: push one event through the very same code the 31M-row CSV uses.

    Correct by construction, but it builds a one-row DataFrame, which costs about 10 ms - over 80% of
    the time spent serving a login. It is kept as the definition of truth that the fast path below is
    tested against (tests/test_fastpath.py), and is not used to serve traffic.
    """
    p = normalize_chunk(pd.DataFrame([ev]))
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in p.iloc[0].to_dict().items()}


def _as_text_scalar(v) -> str:
    """The single-value form of _as_text: missing becomes the literal string "NA"."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return "NA"
    return str(v)


def _ts_ms_scalar(ts) -> int:
    if isinstance(ts, (int, np.integer)) and not isinstance(ts, bool):
        return int(ts)                                    # already epoch milliseconds
    if isinstance(ts, float) and not isinstance(ts, bool):
        return int(ts)
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return int((t - _EPOCH) // pd.Timedelta(milliseconds=1))


def _bool_scalar(v, default: bool = False) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return str(v).strip().lower() in ("true", "1", "yes")


def event_to_row(ev: dict) -> dict:
    """One live login event (dict keyed by RAW_COLUMNS values) -> prepared row (python scalars).

    The serving path. It avoids pandas' per-frame overhead but hashes with the SAME
    pd.util.hash_array call on a small object array, so every value is bit-identical to
    event_to_row_via_frame - which is what keeps the live features equal to the training features.
    Do not "optimise" the hashing itself: a different hash is a different feature, and the model
    would be reading someone else's history.
    """
    vals = [_as_text_scalar(ev.get(RC["user"]))]
    for c in CATS:
        v = ev.get(RC[c])
        if c == "asn":                                    # normalize_chunk coerces ASN to a nullable int
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = None
            else:
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    v = None
        vals.append(_as_text_scalar(v))
    h = pd.util.hash_array(np.array(vals, dtype=object), hash_key=hash_key())

    row = {"ts_ms": _ts_ms_scalar(ev[RC["ts"]]), "user": int(h[0])}
    for i, c in enumerate(CATS, start=1):
        row[c] = int(h[i])
    rtt = ev.get(RC["rtt"])
    if rtt is None or (isinstance(rtt, float) and np.isnan(rtt)):
        row["rtt"] = float("nan")
    else:
        try:
            row["rtt"] = float(np.float32(rtt))           # float32, as the parquet column is
        except (TypeError, ValueError):
            row["rtt"] = float("nan")
    row["success"] = _bool_scalar(ev.get(RC["success"], True), default=True) if RC["success"] in ev else True
    row["attack_ip"] = _bool_scalar(ev.get(RC["attack_ip"]), default=False)
    row["ato"] = _bool_scalar(ev.get(RC["ato"]), default=False)
    return row


PREPARED_KEY_FILE = "hash_key.txt"


def write_prepared_fingerprint(data_dir) -> None:
    """Record which hash key produced a prepared-data folder (the fingerprint, never the key)."""
    from pathlib import Path
    Path(data_dir, PREPARED_KEY_FILE).write_text(key_fingerprint() + "\n")


def read_prepared_fingerprint(data_dir) -> str | None:
    from pathlib import Path
    f = Path(data_dir, PREPARED_KEY_FILE)
    return f.read_text().strip() if f.exists() else None


def load_prepared(data_dir: str, max_rows: int | None = None) -> pd.DataFrame:
    """Read the prepared parquet parts into ONE table without holding two copies in memory
    (pd.concat of a list of parts needs about twice the final size)."""
    import glob
    import pyarrow.parquet as pq
    parts = sorted(glob.glob(f"{data_dir}/part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts in {data_dir} - run scripts/prepare_data.py first")
    check_fingerprint(read_prepared_fingerprint(data_dir), f"prepared data in {data_dir}")
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    n = total if max_rows is None else min(total, max_rows)
    first = pd.read_parquet(parts[0])
    out = {c: np.empty(n, dtype=first[c].dtype) for c in first.columns}
    del first
    pos = 0
    for p in parts:
        if pos >= n:
            break
        ch = pd.read_parquet(p)
        m = min(len(ch), n - pos)
        for c in out:
            out[c][pos:pos + m] = ch[c].to_numpy()[:m]
        pos += m
        del ch
    return pd.DataFrame(out, copy=False)
