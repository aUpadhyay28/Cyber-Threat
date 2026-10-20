"""The fail-open client.

Every test here is about what happens when the risk service is NOT healthy, because that is the
case that decides whether this engine protects a login flow or takes it down. A client that raises
on timeout turns a Redis hiccup into "nobody can sign in".
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from riskengine.client import FAIL_CLOSED_RESULT, FAIL_OPEN_RESULT, RiskClient


class _Handler(BaseHTTPRequestHandler):
    behaviour = "ok"
    delay = 0.0
    protocol_version = "HTTP/1.1"                        # so Content-Length framing is honoured

    def do_POST(self):                                   # noqa: N802 - BaseHTTPRequestHandler API
        # Drain the request body before replying. Skipping this works on Linux but makes Windows
        # close the socket abortively (WinError 10053), and the client loses a response that was
        # actually sent - so a healthy service looked like a failing one.
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if self.delay:
            time.sleep(self.delay)
        if _Handler.behaviour == "ok":
            body = b'{"event_id":"abc","tier":"Medium","action":"step_up_auth","risk_score":98.5}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif _Handler.behaviour == "500":
            self.send_response(500); self.send_header("Content-Length", "0"); self.end_headers()
        else:
            self.send_response(401); self.send_header("Content-Length", "0"); self.end_headers()

    def log_message(self, *a):                           # keep pytest output clean
        return


@pytest.fixture
def server():
    # Threaded: the timeout test makes a handler sleep, and a single-threaded server would stall
    # teardown behind it.
    _Handler.behaviour = "ok"
    _Handler.delay = 0.0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _client(url, **kw):
    return RiskClient(base_url=url, api_key="k" * 32, **kw)


def test_happy_path_returns_the_services_verdict(server):
    c = _client(server)
    out = c.score(user_id="u1", ip="203.0.113.5")
    assert out["tier"] == "Medium" and out["action"] == "step_up_auth"
    assert out["degraded"] is False
    assert c.stats.failures == 0


def test_timeout_fails_open_instead_of_raising(server):
    """The whole point. A slow risk service must not block a login."""
    _Handler.delay = 1.0
    c = _client(server, timeout_seconds=0.15)
    t0 = time.perf_counter()
    out = c.score(user_id="u1", ip="203.0.113.5")
    elapsed = time.perf_counter() - t0
    assert out["tier"] == "Low" and out["action"] == "allow" and out["degraded"] is True
    assert elapsed < 0.9, "the client waited past its own timeout"
    assert c.stats.timeouts == 1 and c.stats.degraded == 1


def test_unreachable_service_fails_open():
    c = _client("http://127.0.0.1:1", timeout_seconds=0.2)   # nothing listens on port 1
    out = c.score(user_id="u1", ip="203.0.113.5")
    assert out == {**FAIL_OPEN_RESULT, "degraded": True}
    assert c.stats.failures == 1


def test_server_error_fails_open(server):
    _Handler.behaviour = "500"
    c = _client(server)
    assert c.score(user_id="u1", ip="203.0.113.5")["degraded"] is True


def test_fail_closed_is_available_for_sensitive_flows(server):
    """Defensible for a payout change; a bad default for a login page."""
    _Handler.behaviour = "500"
    c = _client(server, fail_open=False)
    out = c.score(user_id="u1", ip="203.0.113.5")
    assert out == {**FAIL_CLOSED_RESULT, "degraded": True}
    assert out["action"] == "block_and_review"


def test_degraded_callback_fires_so_failures_are_visible(server):
    """A service failing open silently looks exactly like one that works."""
    _Handler.behaviour = "500"
    seen = []
    c = _client(server, on_degraded=lambda exc: seen.append(exc))
    c.score(user_id="u1", ip="203.0.113.5")
    assert len(seen) == 1


def test_a_raising_callback_cannot_break_a_login(server):
    _Handler.behaviour = "500"
    def boom(exc):
        raise RuntimeError("metrics backend down")
    c = _client(server, on_degraded=boom)
    assert c.score(user_id="u1", ip="203.0.113.5")["degraded"] is True


def test_circuit_breaker_stops_paying_the_timeout(server):
    """Without a breaker, a dead service makes every login wait the full timeout."""
    _Handler.behaviour = "500"
    c = _client(server, breaker_threshold=3, breaker_reset_seconds=60)
    for _ in range(3):
        c.score(user_id="u", ip="203.0.113.5")
    assert c.stats.breaker_open_count == 1
    calls_before = c.stats.failures
    t0 = time.perf_counter()
    out = c.score(user_id="u", ip="203.0.113.5")
    assert time.perf_counter() - t0 < 0.02, "breaker open but the client still made a call"
    assert out["degraded"] is True
    assert c.stats.failures == calls_before, "a short-circuited call must not count as a new failure"


def test_breaker_closes_again_after_the_reset(server):
    _Handler.behaviour = "500"
    c = _client(server, breaker_threshold=2, breaker_reset_seconds=0.2)
    c.score(user_id="u", ip="203.0.113.5")
    c.score(user_id="u", ip="203.0.113.5")
    time.sleep(0.3)
    _Handler.behaviour = "ok"
    out = c.score(user_id="u", ip="203.0.113.5")
    assert out["degraded"] is False and out["tier"] == "Medium"


def test_confirm_failure_never_raises():
    """A lost confirmation only means the user may be challenged again. Never block on it."""
    c = _client("http://127.0.0.1:1", timeout_seconds=0.2)
    assert c.confirm("some-event") is False


def test_stats_summarise_health(server):
    _Handler.behaviour = "500"
    c = _client(server)
    for _ in range(4):
        c.score(user_id="u", ip="203.0.113.5")
    d = c.stats.as_dict()
    assert d["calls"] == 4 and d["failures"] == 4 and d["failure_rate"] == 1.0


def test_optional_fields_are_omitted_not_nulled(server):
    """The service resolves geo itself; sending country=None should not override that."""
    captured = {}
    orig = RiskClient._post

    def spy(self, path, payload):
        captured.update(payload)
        return orig(self, path, payload)

    c = _client(server)
    object.__setattr__(c, "_post", spy.__get__(c, RiskClient))
    c.score(user_id="u1", ip="203.0.113.5")
    assert "country" not in captured and "asn" not in captured
    assert captured["user_id"] == "u1" and captured["success"] is True