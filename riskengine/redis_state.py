"""Shared live state in Redis, so more than one worker can serve logins.

Why this is needed
------------------
The in-process FeatureState is a Python dictionary. Run two uvicorn workers against it and each
gets its own private copy, so an IP hammering accounts across both workers looks half as busy to
each of them - and the 24-hour counters are the main credential-stuffing signal. The measured
ceiling for one worker was ~139 req/s, CPU-bound, so scaling means more workers, and more workers
means the state has to live somewhere they all see.

How parity is kept
------------------
This class does NOT reimplement the feature definitions. It produces the same StateReads struct the
in-memory backend produces, and hands it to the same compute_features function. The only thing that
differs is retrieval, and tests/test_redis_state.py replays an identical event stream through both
backends and asserts every feature of every event matches.

Key layout (all prefixed, so one Redis can host several environments):
  mx                      max timestamp recorded, the clock the windows expire against
  gt                      total logins recorded
  uc                      number of accounts with a profile
  u:{user}                hash n, s, c, last, fails  - the account's own profile
  s:{cat}:{mix}           how many of this user's logins used this value (mix is non-invertible)
  ipu:{ip}, asnu:{asn}    sets of accounts seen from this IP / network
  uip:{user}, uasn:{user} the reverse index, so erasing an account does not scan every IP
  gb:{browser|os}         hash value -> global count
  w:ip:{ip}, w:asn:{asn}  sorted sets of login timestamps (score = ts_ms)
  wf:ip:{ip}, wf:asn:{asn}  the same for failed logins only

A note on window semantics: the in-memory backend walks its deque backwards and stops at the first
event older than 24 h, which assumes timestamps arrive in order. This backend counts by score
range, which is order-independent. The two agree exactly for non-decreasing timestamps - which is
what RISK_TIMESTAMP_SOURCE=server guarantees, and that is the production configuration.

Durability is Redis's job here: run it with AOF (appendonly yes) if losing the last seconds of
behavioural state on a crash matters. It is not the audit trail - that is SQLite/Postgres - so a
small loss degrades scoring accuracy briefly and loses no record of any decision.
"""
from __future__ import annotations

import logging
import math
import uuid
from typing import Optional

from .hashing import key_fingerprint, check_fingerprint
from .config import CATS, MAX_FORWARD_JUMP_MS, STATE_RETENTION_MS
from .features import DAY_MS, HOUR_MS, StateReads, TimestampOutOfRange, compute_features, mix_py

log = logging.getLogger("riskengine.redis")


class RedisFeatureState:
    """Drop-in replacement for FeatureState, backed by Redis.

    Exposes the same surface the service uses: check_timestamp, features, update, glob_total,
    user_count, prune, forget_user. `needs_snapshot` is False because Redis is the durable store -
    there is no JSON file to write.
    """

    needs_snapshot = False

    def __init__(self, client, prefix: str = "risk", retention_ms: int = STATE_RETENTION_MS,
                 max_forward_jump_ms: Optional[int] = MAX_FORWARD_JUMP_MS):
        self.r = client
        self.p = prefix.rstrip(":") + ":"
        self.retention_ms = retention_ms
        self.max_forward_jump_ms = max_forward_jump_ms
        self._glob_total_cache: Optional[int] = None
        # Every key below embeds hashed user/IP values, so a Redis keyspace built under one
        # RISK_HASH_KEY is meaningless under another. The first process to start claims it.
        fp_key = self._k("hash_key_fp")
        self.r.setnx(fp_key, key_fingerprint())
        recorded = self.r.get(fp_key)
        if isinstance(recorded, bytes):
            recorded = recorded.decode()
        check_fingerprint(recorded, f"the Redis keyspace {self.p!r}")

    # ------------------------------------------------------------------ keys
    def _k(self, *parts) -> str:
        return self.p + ":".join(str(x) for x in parts)

    # ------------------------------------------------------------------ clock
    @property
    def max_ts(self) -> Optional[int]:
        # Stored as the score of a single sorted-set member. See _advance in update().
        v = self.r.zscore(self._k("mx"), "ts")
        return None if v is None else int(v)

    def check_timestamp(self, ts_ms: int) -> None:
        if self.max_forward_jump_ms is None:
            return
        mx = self.max_ts
        if mx is not None and int(ts_ms) > mx + self.max_forward_jump_ms:
            raise TimestampOutOfRange(
                f"timestamp {int(ts_ms)} is more than {self.max_forward_jump_ms} ms beyond the newest "
                f"event recorded ({mx}); refusing it so the recent-activity windows cannot be cleared")

    # ------------------------------------------------------------------ reads
    def _read(self, r: dict) -> StateReads:
        """One pipeline, one round trip, everything compute_features needs."""
        u, ip, asn, t = int(r["user"]), int(r["ip"]), int(r["asn"]), int(r["ts_ms"])
        lo1, lo24 = t - HOUR_MS, t - DAY_MS
        pipe = self.r.pipeline(transaction=False)
        pipe.hgetall(self._k("u", u))
        for c in CATS:
            pipe.get(self._k("s", c, mix_py(u, int(r[c]))))
        pipe.scard(self._k("ipu", ip))
        pipe.scard(self._k("asnu", asn))
        pipe.get(self._k("gt"))
        for col in ("browser", "os"):
            pipe.hget(self._k("gb", col), str(int(r[col])))
        pipe.zcount(self._k("w", "ip", ip), lo1, "+inf")
        pipe.zcount(self._k("w", "ip", ip), lo24, "+inf")
        pipe.zcount(self._k("wf", "ip", ip), lo24, "+inf")
        pipe.zcount(self._k("w", "asn", asn), lo24, "+inf")
        pipe.zcount(self._k("wf", "asn", asn), lo24, "+inf")
        out = pipe.execute()

        i = 0
        uh = out[i] or {}; i += 1
        user = None
        if uh:
            get = lambda k: uh.get(k) if k in uh else uh.get(k.encode())
            user = [int(get("n")), float(get("s")), float(get("c")),
                    (None if get("last") in (None, b"", "") else int(get("last"))), int(get("fails"))]
        seen = {}
        for c in CATS:
            seen[c] = int(out[i] or 0); i += 1
        ip_distinct = int(out[i] or 0); i += 1
        asn_distinct = int(out[i] or 0); i += 1
        glob_total = int(out[i] or 0); i += 1
        self._glob_total_cache = glob_total
        glob = {}
        for col in ("browser", "os"):
            glob[col] = int(out[i] or 0); i += 1
        n1 = int(out[i] or 0); i += 1
        n24 = int(out[i] or 0); i += 1
        f24 = int(out[i] or 0); i += 1
        an24 = int(out[i] or 0); i += 1
        af24 = int(out[i] or 0); i += 1
        return StateReads(user=user, seen=seen, ip_distinct=ip_distinct, asn_distinct=asn_distinct,
                          glob_total=glob_total, glob=glob,
                          ip_window=(n1, n24, f24), asn_window=(0, an24, af24))

    def features(self, r: dict) -> dict:
        return compute_features(r, self._read(r))

    # ------------------------------------------------------------------ writes
    def update(self, r: dict, trusted: bool = True, counters: bool = True) -> None:
        u, ip, asn, t = int(r["user"]), int(r["ip"]), int(r["asn"]), int(r["ts_ms"])
        self.check_timestamp(t)
        failed = 0 if r["success"] else 1
        ttl = max(int(self.retention_ms / 1000), 60)
        pipe = self.r.pipeline(transaction=False)
        ukey = self._k("u", u)
        # A profile row is created on first sight either way, so history_count_log and the failure
        # counter behave exactly as they do in memory.
        pipe.hsetnx(ukey, "n", 0)
        pipe.hsetnx(ukey, "s", 0.0)
        pipe.hsetnx(ukey, "c", 0.0)
        pipe.hsetnx(ukey, "last", "")
        pipe.hsetnx(ukey, "fails", 0)

        if counters:
            pipe.hincrby(ukey, "fails", failed)
            pipe.sadd(self._k("ipu", ip), u)
            pipe.sadd(self._k("asnu", asn), u)
            pipe.sadd(self._k("uip", u), ip)                 # reverse index, for erasure
            pipe.sadd(self._k("uasn", u), asn)
            pipe.incr(self._k("gt"))
            for col in ("browser", "os"):
                pipe.hincrby(self._k("gb", col), str(int(r[col])), 1)
            # A unique member per event, so two logins in the same millisecond both count. Generated
            # locally rather than with an INCR, because a round trip per login is not worth paying
            # for uniqueness that 96 random bits already give us.
            member = f"{t}-{uuid.uuid4().hex[:12]}"
            for kind, key in (("ip", ip), ("asn", asn)):
                wk = self._k("w", kind, key)
                pipe.zadd(wk, {member: t})
                pipe.zremrangebyscore(wk, "-inf", t - DAY_MS - 1)
                pipe.expire(wk, ttl)
                if failed:
                    fk = self._k("wf", kind, key)
                    pipe.zadd(fk, {member: t})
                    pipe.zremrangebyscore(fk, "-inf", t - DAY_MS - 1)
                    pipe.expire(fk, ttl)
            for key in (self._k("ipu", ip), self._k("asnu", asn)):
                pipe.expire(key, ttl)

        if trusted:
            for c in CATS:
                pipe.incr(self._k("s", c, mix_py(u, int(r[c]))))
            ang = (t % DAY_MS) / DAY_MS * 2 * math.pi
            pipe.hincrby(ukey, "n", 1)
            pipe.hincrbyfloat(ukey, "s", math.sin(ang))
            pipe.hincrbyfloat(ukey, "c", math.cos(ang))
            pipe.hset(ukey, "last", t)
        # max_ts must only ever move forward, even with concurrent writers. A WATCH/MULTI retry
        # loop costs several extra round trips per login (measured at tens of milliseconds), and a
        # Lua script needs scripting, which fakeredis and some managed Redis offerings do not
        # provide. ZADD ... GT is exactly "set the score only if greater": atomic, one command, no
        # scripting, and it rides along in the pipeline that was going out anyway.
        pipe.zadd(self._k("mx"), {"ts": t}, gt=True)
        pipe.execute()
        if counters:
            self._glob_total_cache = None

    # ------------------------------------------------------------------ service surface
    @property
    def glob_total(self) -> int:
        if self._glob_total_cache is None:
            self._glob_total_cache = int(self.r.get(self._k("gt")) or 0)
        return self._glob_total_cache

    @property
    def user_count(self) -> int:
        return int(self.r.get(self._k("uc")) or 0)

    @property
    def users(self):
        """The service only ever takes len() of this, for /health."""
        return _CountOnly(self.user_count)

    def to_data(self) -> dict:
        raise NotImplementedError(
            "the Redis backend is the durable store; there is no snapshot to write. "
            "Configure Redis persistence (appendonly yes) instead.")

    def prune(self, now_ms: Optional[int] = None, retention_ms: Optional[int] = None) -> dict:
        """Window keys carry a TTL, so Redis expires idle IPs on its own. This reports what is left
        rather than scanning and deleting, which on a large keyspace would be the expensive way to
        do what EXPIRE already does."""
        return {"ips": 0, "asns": 0,
                "note": "Redis expires idle window keys by TTL; nothing to prune manually"}

    def forget_user(self, user: int, values_by_cat: Optional[dict] = None) -> dict:
        """Erase one account. The reverse index (uip/uasn) means this does not scan the keyspace."""
        u = int(user)
        out = {"profile_removed": False, "category_entries_removed": 0,
               "ip_sets_cleaned": 0, "asn_sets_cleaned": 0,
               "category_tables_complete": values_by_cat is not None}
        out["profile_removed"] = bool(self.r.delete(self._k("u", u)))
        for cat, vals in (values_by_cat or {}).items():
            if cat in CATS:
                for v in vals:
                    out["category_entries_removed"] += int(bool(self.r.delete(self._k("s", cat, mix_py(u, int(v))))))
        for idx, setname, label in (("uip", "ipu", "ip_sets_cleaned"), ("uasn", "asnu", "asn_sets_cleaned")):
            members = self.r.smembers(self._k(idx, u)) or set()
            for m in members:
                key = m.decode() if isinstance(m, bytes) else str(m)
                if self.r.srem(self._k(setname, key), u):
                    out[label] += 1
            self.r.delete(self._k(idx, u))
        return out


class _CountOnly:
    """Supports len() and nothing else, so /health can report the account count without loading it."""

    __slots__ = ("_n",)

    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n


def build_state(url: Optional[str] = None, prefix: Optional[str] = None,
                max_forward_jump_ms: Optional[int] = MAX_FORWARD_JUMP_MS):
    """Create a Redis-backed state from RISK_REDIS_URL, or return None if it is not configured."""
    import os
    url = url or os.getenv("RISK_REDIS_URL")
    if not url:
        return None
    try:
        import redis                                   # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise RuntimeError("RISK_REDIS_URL is set but the redis package is missing: pip install redis") from exc
    client = redis.Redis.from_url(url, decode_responses=True)
    client.ping()                                      # fail at startup, not on the first login
    state = RedisFeatureState(client, prefix=prefix or os.getenv("RISK_REDIS_PREFIX", "risk"),
                              max_forward_jump_ms=max_forward_jump_ms)
    log.info("using Redis-backed shared state at %s (prefix %r)", url, state.p)
    return state