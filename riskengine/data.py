"""Raw login rows -> compact, hashed 'prepared' frame. The SAME function is used for the
big CSV (training) and for single live events (serving), so the two can never disagree."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import RAW_COLUMNS as RC, CATS

_EPOCH = pd.Timestamp("1970-01-01")


def _hash(s: pd.Series) -> np.ndarray:
    return pd.util.hash_array(s.to_numpy(dtype=object))          # deterministic uint64


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


def event_to_row(ev: dict) -> dict:
    """One live login event (dict keyed by RAW_COLUMNS values) -> prepared row (python scalars)."""
    p = normalize_chunk(pd.DataFrame([ev]))
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in p.iloc[0].to_dict().items()}
