"""Saving and loading the live state WITHOUT pickle, and integrity checks for the model file.

Why this module exists
----------------------
The scoring service used to keep its live state in a pickle and load it with ``pickle.load``.
Unpickling executes code that is embedded in the file, so anything able to write ``state.pkl`` -
a mounted volume with loose permissions, a path traversal elsewhere in the host, a stale backup
restored by hand - could run arbitrary code inside the risk service. For a component whose whole
job is to decide whether a login is trustworthy, that is the wrong trust model.

State is therefore stored as gzipped JSON with an explicit version, and loaded with a parser that
can only produce data (see FeatureState.to_data / from_data). A file that is still a pickle is
refused with a pointer to scripts/migrate_state_pickle.py, which is the one place that imports
pickle and is meant to be run by an operator on a file they trust.

The model file is still a joblib (scikit-learn estimators have no portable plain-data form), so it
keeps the same risk. That is handled differently: record the digest of the model you intend to run
and verify it at load time. Set RISK_MODEL_SHA256, or keep a <model>.sha256 file next to it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

from .features import FeatureState

PICKLE_MAGICS = (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05", b"(d", b"}q")


def write_state_data(data: dict, path: str | os.PathLike) -> None:
    """Serialise an already-extracted state document and write it atomically.

    Split out from save_state so the scoring service can take the plain-data copy under its lock
    (cheap) and do the serialisation and the disk write on a background thread (not cheap).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    payload = json.dumps(data, separators=(",", ":")).encode()
    # Owner-only from the moment it exists: the state holds every account's behavioural profile.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6) as fh:
        fh.write(payload)
    os.replace(tmp, path)


def save_state(state: FeatureState, path: str | os.PathLike) -> None:
    """Write the state atomically: a crash mid-write leaves the previous file intact."""
    write_state_data(state.to_data(), path)


def load_state(path: str | os.PathLike, max_forward_jump_ms: int | None = None) -> FeatureState:
    path = Path(path)
    head = path.open("rb").read(2)
    if head[:2] != b"\x1f\x8b":                       # not gzip
        if head in PICKLE_MAGICS or head[:1] == b"\x80":
            raise ValueError(
                f"{path} is a pickled state file from an older version. This service no longer loads "
                f"pickles, because loading one executes whatever code it contains. Convert it with:\n"
                f"  python scripts/migrate_state_pickle.py --in {path} --out {path}.json.gz\n"
                f"Only do that for a file you trust.")
        raise ValueError(f"{path} is not a risk-engine state file (expected gzipped JSON)")
    with gzip.open(path, "rb") as fh:
        data = json.loads(fh.read().decode())
    kwargs = {} if max_forward_jump_ms is None else {"max_forward_jump_ms": max_forward_jump_ms}
    return FeatureState.from_data(data, **kwargs)


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_model(path: str | os.PathLike, expected: str | None = None) -> str:
    """Check the model file's digest before it is deserialised.

    `expected` wins; otherwise a <model>.sha256 sidecar is used if present. With neither, the digest
    is returned unverified so the caller can log it - absence of a pin is not treated as a failure,
    because that would stop a first run, but the API logs a warning so it is visible.
    """
    digest = sha256_file(path)
    if expected is None:
        side = Path(str(path) + ".sha256")
        if side.exists():
            expected = side.read_text().split()[0].strip()
    if expected and digest != expected.lower():
        raise ValueError(
            f"model file {path} does not match the pinned digest.\n  expected {expected.lower()}\n"
            f"  actual   {digest}\nRefusing to load it. If you deployed a new model on purpose, update "
            f"RISK_MODEL_SHA256 or the .sha256 file beside it.")
    return digest


def write_digest(path: str | os.PathLike) -> str:
    """Record a model's digest beside it, so later loads are pinned to this exact file."""
    digest = sha256_file(path)
    Path(str(path) + ".sha256").write_text(digest + "\n")
    return digest
