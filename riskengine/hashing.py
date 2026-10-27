"""Keyed hashing of identifiers (user IDs, IPs, devices, browsers, ...).

Every identifier is turned into a 64-bit number before it is stored or used as a feature key. The
hash is pandas' SipHash (``pd.util.hash_array``), which takes a 128-bit key. With the library's
public default key the hashes are only *deterministic*, not *secret*: anyone who has a state file,
a database dump or an exported sample can hash every plausible email address or account number and
match them back. That is harmless on the public RBA benchmark and a privacy breach on a real site.

Set ``RISK_HASH_KEY`` to a long random secret (store it like the API key) and every hash becomes a
keyed one. The same key must be used everywhere data is prepared or served - ``prepare_data.py``,
``import_history.py`` and the live service - so its fingerprint is recorded with the prepared data,
inside the state file (and in Redis), and a mismatch is refused rather than silently scoring each
user against an empty history.

Changing the key does not change what the model learned (no feature is a raw hash value; they are all
"seen before?" / frequency / count features), but it does orphan every profile in an existing state
file. Rotate it by re-running ``import_history.py`` with the new key.
"""
from __future__ import annotations

import base64
import hashlib
import os

DEFAULT_PANDAS_KEY = "0123456789123456"   # pandas' public default: deterministic, NOT secret
ENV_VAR = "RISK_HASH_KEY"
MIN_SECRET_LEN = 24
UNKEYED = "unkeyed"                        # fingerprint recorded when no key is configured

_cache: dict[str, tuple[str, str]] = {}


def _derive(secret: str) -> str:
    # SipHash needs exactly 16 bytes; pandas encodes the key as UTF-8, so derive 16 ASCII characters
    # (96 bits of the secret's SHA-256) rather than truncating a user-chosen string.
    return base64.b64encode(hashlib.sha256(b"riskengine-siphash-key:" + secret.encode()).digest()).decode()[:16]


def _resolve() -> tuple[str, str]:
    secret = os.getenv(ENV_VAR, "")
    hit = _cache.get(secret)
    if hit is not None:
        return hit
    if not secret:
        out = (DEFAULT_PANDAS_KEY, UNKEYED)
    else:
        if len(secret) < MIN_SECRET_LEN:
            raise RuntimeError(
                f"{ENV_VAR} is {len(secret)} characters; use at least {MIN_SECRET_LEN} random ones "
                f"(e.g. `python -c \"import secrets;print(secrets.token_urlsafe(32))\"`).")
        key = _derive(secret)
        fp = hashlib.sha256(b"riskengine-key-fingerprint:" + key.encode()).hexdigest()[:16]
        out = (key, fp)
    _cache.clear()
    _cache[secret] = out
    return out


def hash_key() -> str:
    """The 16-character SipHash key to pass to ``pd.util.hash_array``."""
    return _resolve()[0]


def key_fingerprint() -> str:
    """A non-reversible tag for the active key, or ``"unkeyed"``. Safe to store and to show in /health."""
    return _resolve()[1]


def is_keyed() -> bool:
    return key_fingerprint() != UNKEYED


def check_fingerprint(recorded: str | None, what: str) -> None:
    """Refuse an artifact (model, state, prepared data) hashed under a different key.

    Artifacts written before fingerprints existed record nothing; they were hashed with the default
    key, so they are treated as ``"unkeyed"``.
    """
    recorded = recorded or UNKEYED
    active = key_fingerprint()
    if recorded != active:
        raise RuntimeError(
            f"{what} was hashed with key fingerprint {recorded!r} but this process uses {active!r} "
            f"({ENV_VAR} {'is set' if is_keyed() else 'is not set'}). Every user and IP would be "
            f"looked up under a different hash, so their histories would be invisible. Use the same "
            f"{ENV_VAR} that produced it, or rebuild it (scripts/import_history.py / prepare_data.py).")
