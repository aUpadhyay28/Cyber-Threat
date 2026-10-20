"""The serving row-conversion must equal the training one, value for value.

event_to_row is the hot path: it runs on every login and used to account for over 80% of the time
spent serving one. The fast version skips pandas' per-frame machinery but must produce exactly the
same hashed row as event_to_row_via_frame, which pushes the event through the identical code the
31M-row CSV goes through. If these two ever disagree, the live service computes a user's history
against different hashes than the model was trained on, and every feature silently becomes wrong -
the same class of failure test_parity.py guards against, one layer lower down.
"""
import math

import numpy as np
import pandas as pd
import pytest

from riskengine.config import RAW_COLUMNS as RC
from riskengine.data import event_to_row, event_to_row_via_frame, normalize_chunk
from riskengine.features import FeatureState, build_features, FEATURES
from scripts.make_mock_data import make


def _same(a: dict, b: dict) -> None:
    assert set(a) == set(b), f"different keys: {set(a) ^ set(b)}"
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(av, float) and isinstance(bv, float) and math.isnan(av) and math.isnan(bv):
            continue
        assert av == bv and type(av) is type(bv), f"{k}: {av!r} ({type(av).__name__}) != {bv!r} ({type(bv).__name__})"


def _event(**over):
    ev = {RC["ts"]: "2021-03-04T09:30:00Z", RC["user"]: "u-1", RC["ip"]: "203.0.113.7",
          RC["country"]: "IN", RC["asn"]: 9829, RC["browser"]: "Chrome 120",
          RC["os"]: "Windows 11", RC["device"]: "desktop", RC["rtt"]: 80.0,
          RC["success"]: True}
    ev.update(over)
    return ev


def test_fast_and_frame_agree_on_a_plain_event():
    ev = _event()
    _same(event_to_row(ev), event_to_row_via_frame(ev))


@pytest.mark.parametrize("over", [
    {RC["country"]: None},
    {RC["asn"]: None},
    {RC["asn"]: "64500"},                      # ASN arriving as a string
    {RC["asn"]: 64500.0},                      # ASN arriving as a float
    {RC["asn"]: "not-a-number"},               # unparseable -> treated as missing
    {RC["browser"]: None, RC["os"]: None, RC["device"]: None},
    {RC["rtt"]: None},
    {RC["rtt"]: float("nan")},
    {RC["rtt"]: 0.0},
    {RC["rtt"]: 310.5},                        # exercises the float32 round-trip
    {RC["success"]: False},
    {RC["success"]: "true"},                   # string booleans, as a CSV would give
    {RC["success"]: "FALSE"},
    {RC["ts"]: "2021-03-04 09:30:00"},         # naive, no timezone
    {RC["ts"]: "2021-03-04T09:30:00+05:30"},   # a non-UTC offset
    {RC["ts"]: 1614850200000},                 # already epoch milliseconds
    {RC["user"]: "user with spaces"},
    {RC["user"]: "ユーザー-42"},                  # non-ASCII identifiers must hash the same
    {RC["country"]: "भारत"},
    {RC["user"]: "x" * 300},
    {RC["ip"]: "2001:db8::1"},                 # IPv6
])
def test_fast_and_frame_agree_on_edge_cases(over):
    ev = _event(**over)
    _same(event_to_row(ev), event_to_row_via_frame(ev))


def test_agreement_over_many_generated_events():
    """Every row of a generated dataset, through both paths."""
    raw = make(600, n_users=60, seed=17)
    for rec in raw.to_dict("records"):
        _same(event_to_row(rec), event_to_row_via_frame(rec))


def test_fast_path_matches_the_batch_pipeline_end_to_end():
    """The real guarantee: features built from fast-path rows equal features built by the
    vectorised training code on the same logins."""
    raw = make(1500, n_users=80, seed=23, span_days=2)
    batch = build_features(normalize_chunk(raw))
    st, rows = FeatureState(max_forward_jump_ms=None), []
    for rec in raw.to_dict("records"):
        r = event_to_row(rec)                       # the fast serving path
        rows.append(st.features(r))
        st.update(r, trusted=True)
    online = pd.DataFrame(rows)[FEATURES].astype("float32")
    np.testing.assert_allclose(online.to_numpy(), batch.to_numpy(), rtol=1e-4, atol=1e-5, equal_nan=True)


def test_fast_path_is_actually_faster():
    """Guards the reason this code exists: if someone reverts it to the frame version, say so."""
    import time
    evs = [_event(**{RC["user"]: f"u-{i}"}) for i in range(60)]
    t = time.perf_counter()
    for ev in evs:
        event_to_row_via_frame(ev)
    slow = time.perf_counter() - t
    t = time.perf_counter()
    for ev in evs:
        event_to_row(ev)
    fast = time.perf_counter() - t
    assert fast < slow / 3, f"fast path only {slow/fast:.1f}x faster; expected at least 3x"
