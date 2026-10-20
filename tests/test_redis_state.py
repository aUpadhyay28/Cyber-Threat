"""The Redis backend must compute exactly what the in-memory backend computes.

This is the same guarantee test_parity.py makes between batch and online, one layer out: if the
shared-state backend disagrees with the local one, then moving to multiple workers silently changes
every score, and the model is reading histories that were computed by different rules than it was
trained on.

Runs against a real Redis when one is reachable, otherwise fakeredis. Both are exercised in CI.
"""
import math

import pytest

from riskengine.data import event_to_row
from riskengine.features import FEATURES, FeatureState, TimestampOutOfRange
from riskengine.redis_state import RedisFeatureState
from scripts.make_mock_data import make


@pytest.fixture
def client():
    """A clean Redis-like client. Prefers a real server; falls back to fakeredis."""
    try:
        import redis
        c = redis.Redis.from_url("redis://127.0.0.1:6379/15", decode_responses=True,
                                 socket_connect_timeout=0.5)
        c.ping()
        c.flushdb()
        yield c
        c.flushdb()
        return
    except Exception:                                    # noqa: BLE001 - no server, use the fake
        pass
    try:
        import fakeredis
    except ImportError:
        pytest.skip("neither a local Redis nor fakeredis is available")
    c = fakeredis.FakeRedis(decode_responses=True)
    yield c


def _states(client, jump=None):
    return FeatureState(max_forward_jump_ms=jump), RedisFeatureState(client, max_forward_jump_ms=jump)


def _assert_same(a: dict, b: dict, label: str = ""):
    for k in FEATURES:
        av, bv = a[k], b[k]
        if isinstance(av, float) and isinstance(bv, float) and math.isnan(av) and math.isnan(bv):
            continue
        assert av == pytest.approx(bv, rel=1e-12, abs=1e-12), f"{label} {k}: {av!r} != {bv!r}"


def test_both_backends_agree_on_a_realistic_stream(client):
    """Every feature of every login, replayed through both backends in the same order."""
    mem, red = _states(client, jump=None)
    raw = make(900, n_users=70, seed=5, span_days=2)
    for i, rec in enumerate(raw.to_dict("records")):
        r = event_to_row(rec)
        trusted = bool(r["success"])
        _assert_same(mem.features(r), red.features(r), label=f"event {i}")
        mem.update(r, trusted=trusted, counters=True)
        red.update(r, trusted=trusted, counters=True)
    assert red.glob_total == mem.glob_total


def test_both_backends_agree_with_dense_windows(client):
    """Packed into a short span so the 1 h / 24 h counts are the interesting part."""
    mem, red = _states(client, jump=None)
    raw = make(700, n_users=25, seed=31, span_days=1)
    seen_nonzero = False
    for i, rec in enumerate(raw.to_dict("records")):
        r = event_to_row(rec)
        fm, fr = mem.features(r), red.features(r)
        if fm["ip_n_24h_log"] > 0:
            seen_nonzero = True
        _assert_same(fm, fr, label=f"event {i}")
        mem.update(r, trusted=True, counters=True)
        red.update(r, trusted=True, counters=True)
    assert seen_nonzero, "the fixture never produced a non-empty window; test proves nothing"


def test_counters_only_update_matches(client):
    """The split the service uses: counters recorded under the lock, profile applied after."""
    mem, red = _states(client, jump=None)
    raw = make(300, n_users=20, seed=9, span_days=1)
    for rec in raw.to_dict("records"):
        r = event_to_row(rec)
        mem.update(r, trusted=False, counters=True)
        red.update(r, trusted=False, counters=True)
        mem.update(r, trusted=True, counters=False)
        red.update(r, trusted=True, counters=False)
        _assert_same(mem.features(r), red.features(r))


def test_untrusted_update_does_not_teach_the_profile(client):
    """The trust gate has to behave identically, or an attacker gets a different answer per worker."""
    mem, red = _states(client, jump=None)
    raw = make(60, n_users=5, seed=3)
    r = event_to_row(raw.to_dict("records")[0])
    mem.update(r, trusted=False, counters=True)
    red.update(r, trusted=False, counters=True)
    assert mem.features(r)["new_country"] == 1.0
    assert red.features(r)["new_country"] == 1.0
    mem.update(r, trusted=True, counters=False)
    red.update(r, trusted=True, counters=False)
    assert mem.features(r)["new_country"] == 0.0
    assert red.features(r)["new_country"] == 0.0


def test_timestamp_guard_applies_in_redis_too(client):
    """The evasion fix must not be lost when the state moves out of process."""
    _, red = _states(client, jump=7 * 86_400_000)
    base = 1_700_000_000_000
    row = {"user": 1, "ip": 2, "asn": 3, "country": 4, "browser": 5, "os": 6, "device": 7,
           "ts_ms": base, "success": False, "rtt": 10.0}
    red.update(row, trusted=False, counters=True)
    with pytest.raises(TimestampOutOfRange):
        red.update({**row, "ts_ms": base + 10 * 365 * 86_400_000}, trusted=False, counters=True)
    assert red.max_ts == base


def test_max_ts_only_moves_forward(client):
    _, red = _states(client, jump=None)
    base = 1_700_000_000_000
    row = {"user": 1, "ip": 2, "asn": 3, "country": 4, "browser": 5, "os": 6, "device": 7,
           "ts_ms": base, "success": True, "rtt": 10.0}
    red.update(row, trusted=True, counters=True)
    red.update({**row, "ts_ms": base - 50_000}, trusted=True, counters=True)   # late arrival
    assert red.max_ts == base


def test_two_workers_share_one_view(client):
    """The reason this backend exists: an IP hammering accounts through different workers must look
    as busy as it really is, from either one."""
    import redis as _redis  # noqa: F401 - only to keep the import local to this test's intent
    a = RedisFeatureState(client, max_forward_jump_ms=None)
    b = RedisFeatureState(client, max_forward_jump_ms=None)
    base = 1_700_000_000_000
    def row(u, t, ok=False):
        return {"user": u, "ip": 999, "asn": 3, "country": 4, "browser": 5, "os": 6, "device": 7,
                "ts_ms": t, "success": ok, "rtt": 10.0}
    for i in range(40):                                  # alternate the two "workers"
        (a if i % 2 == 0 else b).update(row(1000 + i, base + i * 1000), trusted=False, counters=True)
    seen_from_a = a.features(row(1, base + 41_000))
    seen_from_b = b.features(row(1, base + 41_000))
    assert round(math.expm1(seen_from_a["ip_fail_24h_log"])) == 40
    _assert_same(seen_from_a, seen_from_b)

    local = FeatureState(max_forward_jump_ms=None)       # what one isolated worker would have seen
    for i in range(0, 40, 2):
        local.update(row(1000 + i, base + i * 1000), trusted=False, counters=True)
    assert round(math.expm1(local.features(row(1, base + 41_000))["ip_fail_24h_log"])) == 20


def test_forget_user_clears_redis_state(client):
    _, red = _states(client, jump=None)
    base = 1_700_000_000_000
    row = {"user": 4242, "ip": 77, "asn": 88, "country": 4, "browser": 5, "os": 6, "device": 7,
           "ts_ms": base, "success": True, "rtt": 10.0}
    for i in range(3):
        red.update({**row, "ts_ms": base + i * 60_000}, trusted=True, counters=True)
    assert red.features(row)["new_country"] == 0.0

    out = red.forget_user(4242, {c: [row[c]] for c in ("ip", "country", "asn", "browser", "os", "device")})
    assert out["profile_removed"] and out["category_entries_removed"] == 6
    assert out["ip_sets_cleaned"] == 1 and out["asn_sets_cleaned"] == 1
    assert red.features(row)["new_country"] == 1.0
    assert red.features(row)["history_count_log"] == 0.0
    assert red.features(row)["ip_distinct_users_log"] == 0.0


def test_snapshotting_is_not_pretended(client):
    """Redis is the durable store; silently writing a stale JSON file would be worse than refusing."""
    _, red = _states(client)
    assert red.needs_snapshot is False
    with pytest.raises(NotImplementedError, match="durable store"):
        red.to_data()