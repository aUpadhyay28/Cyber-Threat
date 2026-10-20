"""Risk scoring service. Run:
   RISK_API_KEY=change-me uvicorn riskengine.api:create_app --factory --port 8000
Keep this service INTERNAL (only your web backend calls it) and never show scores/reasons to end users.

Security model
--------------
* The caller is trusted to say WHO logged in and WHETHER it succeeded; it is not trusted to say WHEN.
  By default the service timestamps events with its own clock (RISK_TIMESTAMP_SOURCE=server), because
  the 1 h / 24 h window features expire relative to the newest timestamp seen, so one future-dated
  event would clear an attacking IP's recent-failure history. Replay and backfill can opt into
  client timestamps, and then FeatureState bounds how far forward each event may jump.
* The live state is plain data, never a pickle (riskengine/persist.py).
* The model file's digest is verified before it is deserialised (RISK_MODEL_SHA256 or a .sha256 file).
* Every decision records which model version made it.
* Scores and reasons are internal. `explain()` output tells an attacker what to evade.
"""
from __future__ import annotations
import hmac, ipaddress, json, logging, os, queue, threading, time, uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from . import config as C
from .config import RAW_COLUMNS as RC
from .data import event_to_row
from .features import FeatureState, TimestampOutOfRange
from .geoip import apply_resolution, build_resolver
from .policy import HomeCountryPolicy
from .modeling import RiskModel, explain
from .observability import Metrics, configure_logging
from .persist import load_state, verify_model, write_state_data
from .redis_state import build_state
from .store import Store

log = logging.getLogger("riskengine")

SAVE_EVERY = 200
# The network-volume features only mean something once the system has seen real traffic. Until it has seen
# this many logins, results are log-only: action is forced to "allow" and the would-be action is reported.
WARMUP_MIN_EVENTS = int(os.getenv("RISK_WARMUP_MIN_EVENTS", "10000"))
MIN_API_KEY_LEN = 24
RATE_LIMIT_PER_MIN = int(os.getenv("RISK_RATE_LIMIT_PER_MIN", "1200"))   # 0 disables
MAX_ID_LEN = 200
WRITE_QUEUE_MAX = int(os.getenv("RISK_WRITE_QUEUE_MAX", "64"))


class ScoreRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    ip: str = Field(min_length=1, max_length=64)
    timestamp: Optional[datetime] = None   # only used when RISK_TIMESTAMP_SOURCE=client (see module docstring)
    country: Optional[str] = Field(default=None, max_length=8)   # derive from IP server-side (MaxMind GeoLite2)
    asn: Optional[Union[int, str]] = None
    browser: Optional[str] = Field(default=None, max_length=MAX_ID_LEN)
    os: Optional[str] = Field(default=None, max_length=MAX_ID_LEN)
    device: Optional[str] = Field(default=None, max_length=MAX_ID_LEN)
    rtt_ms: Optional[float] = Field(default=None, ge=0, le=600_000)
    success: bool = True


class VerdictRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=64)
    verdict: str = Field(description="benign | suspicious | confirmed_malicious")
    analyst: str = Field(default="", max_length=MAX_ID_LEN)
    note: str = Field(default="", max_length=2000)


class ForgetRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=MAX_ID_LEN)


class _RedisRateLimiter:
    """Fixed-window counter in Redis, so the limit is the limit across every worker.

    The in-process limiter below counts per worker: with four workers a "1200/min" limit is really
    4800/min, which is not what anyone configuring it intends.
    """

    def __init__(self, client, per_min: int, prefix: str = "risk"):
        self.r, self.per_min, self.p = client, per_min, prefix.rstrip(":") + ":rl:"

    def allow(self, key: str) -> bool:
        if self.per_min <= 0:
            return True
        bucket = int(time.time() // 60)
        k = f"{self.p}{key}:{bucket}"
        try:
            pipe = self.r.pipeline(transaction=False)
            pipe.incr(k)
            pipe.expire(k, 120)
            count = pipe.execute()[0]
            return int(count) <= self.per_min
        except Exception:                               # noqa: BLE001
            # A limiter that fails closed would turn a Redis blip into a site outage.
            log.warning("rate limiter unavailable; allowing the request")
            return True


class _RateLimiter:
    """Fixed-window counter per caller. In-process: with several workers the effective limit is
    per worker, so treat it as a backstop against a runaway or probing client, not as a quota."""

    def __init__(self, per_min: int):
        self.per_min, self.hits, self.lock = per_min, {}, threading.Lock()

    def allow(self, key: str) -> bool:
        if self.per_min <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            dq = self.hits.setdefault(key, deque())
            while dq and dq[0] < now - 60:
                dq.popleft()
            if len(dq) >= self.per_min:
                return False
            dq.append(now)
            if len(self.hits) > 10_000:                       # bound memory under spoofed keys
                for k in [k for k, v in self.hits.items() if not v]:
                    self.hits.pop(k, None)
            return True


def _valid_ip(value: str) -> str:
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        # "from None": the parse error itself must not reach the caller.
        raise HTTPException(422, "ip must be a valid IPv4 or IPv6 address") from None


def create_app(model_path=None, state_path=None, db_path=None, api_key=None,
               timestamp_source=None, model_sha256=None, geo_resolver=None, policy=None) -> FastAPI:
    model_path = model_path or os.getenv("RISK_MODEL_PATH", "models/latest.joblib")
    state_path = state_path or os.getenv("RISK_STATE_PATH", "data/state.json.gz")
    db_path = db_path or os.getenv("RISK_DB_PATH", "data/risk.db")
    api_key = api_key or os.getenv("RISK_API_KEY")
    model_sha256 = model_sha256 or os.getenv("RISK_MODEL_SHA256")
    ts_source = (timestamp_source or os.getenv("RISK_TIMESTAMP_SOURCE", C.TIMESTAMP_SOURCE)).lower()
    if ts_source not in ("server", "client"):
        raise RuntimeError("RISK_TIMESTAMP_SOURCE must be 'server' or 'client'")
    if not api_key:
        raise RuntimeError("Set RISK_API_KEY (the secret your web backend sends in the X-API-Key header)")
    if len(api_key) < MIN_API_KEY_LEN and not os.getenv("RISK_ALLOW_SHORT_KEY"):
        raise RuntimeError(
            f"RISK_API_KEY is {len(api_key)} characters; use at least {MIN_API_KEY_LEN} random ones "
            f"(e.g. `python -c \"import secrets;print(secrets.token_urlsafe(32))\"`). "
            f"Set RISK_ALLOW_SHORT_KEY=1 only for tests.")

    # In "server" mode timestamps come from this process's clock, which is trusted, so the state does
    # not need the forward-jump guard - a genuine long gap SHOULD expire the activity windows. In
    # "client" mode the timestamp is attacker-influenced, so the guard is on.
    configure_logging()
    metrics = Metrics(os.getenv("RISK_METRICS", "1") not in ("0", "false", "False"))
    geo = geo_resolver if geo_resolver is not None else build_resolver()
    pol = policy if policy is not None else HomeCountryPolicy.from_env()
    jump_limit = None if ts_source == "server" else C.MAX_FORWARD_JUMP_MS
    ctx = {"lock": threading.Lock(), "since_save": 0, "ts_source": ts_source,
           "geo": geo, "policy": pol, "metrics": metrics, "ready": False}
    limiter_holder = {"limiter": _RateLimiter(RATE_LIMIT_PER_MIN)}
    # Audit writes and state snapshots are I/O. Doing them inside the scoring lock made every other
    # login wait for a disk write, and the 200-event snapshot was a ~200 ms stall visible as the
    # latency maximum under load. They now run on a background thread.
    writer_q: "queue.Queue" = queue.Queue(maxsize=WRITE_QUEUE_MAX)
    writer_stop = threading.Event()

    def _persist(job):
        kind, payload = job
        if kind == "state":
            write_state_data(payload, state_path)
        else:
            raise ValueError(f"unknown write job {kind!r}")

    def _writer_loop():
        while not writer_stop.is_set() or not writer_q.empty():
            try:
                job = writer_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                _persist(job)
            except Exception:                              # noqa: BLE001 - a writer must never die
                log.exception("background write failed (%s)", job[0])
            finally:
                writer_q.task_done()

    def _enqueue(job) -> None:
        """Queue a write, or do it inline if the queue is full.

        Falling back to an inline write is deliberate: this queue carries the audit trail, and
        dropping entries to stay fast would mean losing the record of decisions already returned to
        the caller. Under sustained overload the service gets slower rather than forgetful.
        """
        try:
            writer_q.put_nowait(job)
        except queue.Full:
            log.warning("write queue full (%d): writing %s inline", WRITE_QUEUE_MAX, job[0])
            metrics.degraded.inc()
            _persist(job)

    def _flush() -> None:
        """Block until every queued write has landed on disk.

        Used by the admin endpoints that make a promise: when /v1/admin/forget-user returns 200 the
        erasure has to be durable, not merely scheduled. Scoring never calls this.
        """
        writer_q.join()

    def _snapshot_locked():
        """Take a consistent plain-data copy under the lock; serialise and write it off-thread.

        Measured on a 60,000-login state: to_data() 22 ms, json+gzip+write 157 ms. Only the first
        part has to see a quiet state, so only the first part holds the lock.
        """
        if not getattr(ctx["state"], "needs_snapshot", True):
            ctx["since_save"] = 0          # Redis is durable; there is nothing to write out
            return
        data = ctx["state"].to_data()
        ctx["since_save"] = 0
        _enqueue(("state", data))

    @asynccontextmanager
    async def lifespan(_):
        digest = verify_model(model_path, model_sha256)
        if not model_sha256 and not Path(str(model_path) + ".sha256").exists():
            log.warning("model %s loaded WITHOUT an integrity pin (sha256=%s). Pin it with "
                        "scripts/pin_model.py so a swapped file cannot be loaded silently.",
                        model_path, digest)
        ctx["model_sha256"] = digest
        ctx["model"] = RiskModel.load(model_path)
        shared = build_state(max_forward_jump_ms=jump_limit)
        if shared is not None:
            # Several workers, one view of the world. No snapshot file: Redis is the durable store.
            ctx["state"] = shared
            if RATE_LIMIT_PER_MIN > 0:
                limiter_holder["limiter"] = _RedisRateLimiter(shared.r, RATE_LIMIT_PER_MIN)
        elif Path(state_path).exists():
            ctx["state"] = load_state(state_path, max_forward_jump_ms=jump_limit)
            ctx["state"].max_forward_jump_ms = jump_limit
        else:
            ctx["state"] = FeatureState(max_forward_jump_ms=jump_limit)
            ctx["state"].max_forward_jump_ms = jump_limit
        ctx["store"] = Store(db_path)
        writer = threading.Thread(target=_writer_loop, name="risk-writer", daemon=True)
        writer.start()
        ctx["writer"] = writer
        ctx["ready"] = True
        log.info("risk service ready: model=%s timestamp_source=%s events_in_state=%s",
                 ctx["model"].meta.get("version"), ts_source, ctx["state"].glob_total)
        yield
        # Shutdown: stop accepting work, flush whatever is queued, then write a final snapshot
        # synchronously so the state on disk matches the last login actually scored.
        ctx["ready"] = False
        writer_stop.set()
        writer.join(timeout=30)
        if getattr(ctx["state"], "needs_snapshot", True):
            with ctx["lock"]:
                data = ctx["state"].to_data()
            write_state_data(data, state_path)
            log.info("risk service stopped; final state written to %s", state_path)
        else:
            log.info("risk service stopped; shared state remains in Redis")

    app = FastAPI(title="Adaptive Login Risk Service", lifespan=lifespan)

    @app.middleware("http")
    async def _headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def auth(request: Request, x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key, api_key):
            # Rate-limit by source address on failure so the key cannot be probed quickly.
            peer = request.client.host if request.client else "unknown"
            if not limiter_holder["limiter"].allow(f"bad:{peer}"):
                raise HTTPException(429, "too many requests")
            raise HTTPException(401, "invalid API key")
        if not limiter_holder["limiter"].allow("ok"):
            raise HTTPException(429, "too many requests")

    def _event_ts(req: ScoreRequest) -> str:
        """The timestamp this login is scored with, as ISO-8601. data.event_to_row turns it into
        the ms value the features use, so there is exactly one source of truth for the time."""
        now_ms = int(time.time() * 1000)
        if ctx["ts_source"] == "server":
            return datetime.now(timezone.utc).isoformat()
        if req.timestamp is None:
            raise HTTPException(422, "timestamp is required when the service runs with "
                                     "RISK_TIMESTAMP_SOURCE=client")
        ts = req.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if int(ts.timestamp() * 1000) > now_ms + C.MAX_CLOCK_SKEW_MS:
            raise HTTPException(422, "timestamp is in the future beyond the allowed clock skew")
        return ts.isoformat()

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/console", include_in_schema=False)
    def console():
        """The analyst triage console, served from this same origin.

        Same-origin on purpose: the console calls /v1/queue and /v1/verdict with the API key, and
        serving it from anywhere else would mean either CORS or a second place to configure the
        secret. It is static HTML with no build step and no external requests.
        """
        page = Path(__file__).with_name("console.html")
        if not page.exists():
            raise HTTPException(404, "console.html is missing from the riskengine package")
        return Response(content=page.read_text(encoding="utf-8"), media_type="text/html; charset=utf-8")

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint():
        """Prometheus scrape target. Deliberately unauthenticated and free of personal data, the way
        a scrape endpoint has to be - keep it reachable only from inside your network."""
        st = ctx.get("state")
        if st is not None:
            metrics.warming.set(1 if st.glob_total < WARMUP_MIN_EVENTS else 0)
            metrics.state_logins.set(st.glob_total)
            metrics.state_users.set(len(st.users))
        body, content_type = metrics.render()
        return Response(content=body, media_type=content_type)

    @app.get("/ready", include_in_schema=False)
    def ready():
        """Readiness, as distinct from liveness.

        /health answers "is this process alive" and is what a restart policy should watch. /ready
        answers "should traffic be sent here", which is false while the model is still loading and
        false during shutdown - otherwise a rolling deploy sends logins to a worker that has no
        model yet and they all fail at once.
        """
        if not ctx.get("ready"):
            raise HTTPException(503, "not ready")
        return {"ready": True, "model_version": ctx["model"].meta.get("version")}

    @app.get("/health")
    def health():
        m = ctx["model"].meta
        return {"status": "ok", "model_version": m.get("version"),
                "model_sha256": ctx.get("model_sha256"),
                "timestamp_source": ctx["ts_source"],
                "geoip_source": ctx["geo"].name,
                "state_backend": "redis" if not getattr(ctx["state"], "needs_snapshot", True) else "in-process",
                "home_countries": sorted(ctx["policy"].home_countries) or None,
                "users_in_state": len(ctx["state"].users),
                "logins_in_state": ctx["state"].glob_total,
                "warming_up": ctx["state"].glob_total < WARMUP_MIN_EVENTS}

    @app.post("/v1/score", dependencies=[Depends(auth)])
    def score(req: ScoreRequest):
        t_start = time.perf_counter()
        ip = _valid_ip(req.ip)
        ts_iso = _event_ts(req)
        # Resolve the origin from the address itself. Country and ASN are among the strongest
        # signals here, so letting the caller supply them means letting an attacker choose them.
        geo = apply_resolution(ctx["geo"], ip, req.country, req.asn)
        if geo["geo_mismatch"]:
            log.warning("geo mismatch for %s from %s: caller said %s, %s says %s", req.user_id, ip,
                        geo["geo_mismatch"]["claimed"], ctx["geo"].name, geo["geo_mismatch"]["resolved"])
        raw = {RC["ts"]: ts_iso, RC["user"]: req.user_id, RC["ip"]: ip,
               RC["country"]: geo["country"], RC["asn"]: geo["asn"], RC["browser"]: req.browser,
               RC["os"]: req.os, RC["device"]: req.device, RC["rtt"]: req.rtt_ms, RC["success"]: req.success}
        row = event_to_row(raw)
        # Phase 1, under the lock: read this login's features and record the network counters in the
        # SAME critical section. These two must not be separated. If scoring sat between them, a
        # burst of concurrent logins from one IP would each compute features before any of them was
        # recorded, so every request in the burst would see an idle network - letting an attacker
        # parallelise past the recent-activity counts that detect credential stuffing.
        with ctx["lock"]:
            try:
                ctx["state"].check_timestamp(row["ts_ms"])
            except TimestampOutOfRange as e:
                log.warning("rejected a login for %s: %s", req.user_id, e)
                metrics.rejected.labels(reason="timestamp_out_of_range").inc()
                raise HTTPException(422, "timestamp is too far ahead of the events already recorded") from None
            feats = ctx["state"].features(row)
            warming = ctx["state"].glob_total < WARMUP_MIN_EVENTS
            ctx["state"].update(row, trusted=False, counters=True)
            ctx["since_save"] += 1
            due_snapshot = ctx["since_save"] >= SAVE_EVERY
            if due_snapshot:
                _snapshot_locked()
            model = ctx["model"]

        # Phase 2, no lock: the model is immutable and feats is a plain dict, so scoring needs no
        # exclusive access. This is the expensive part (~1.5 ms) and it no longer blocks anyone.
        res = model.score_features(feats)
        # Policy runs BEFORE `trusted` is decided, so a login the home-country rule escalates does
        # not quietly become part of the user's normal pattern.
        res = ctx["policy"].apply(res, geo["country"], feats, warming)
        reasons = explain(feats) if res["tier"] != "Low" else []
        if "foreign_country_step_up" in res["policy"]:
            reasons = reasons + ["Login from a country this account has not used, outside the home region"]
        trusted = res["tier"] == "Low" and req.success      # only clean successful logins shape the 'normal' profile
        if warming:                                         # log-only until enough traffic has been seen
            res = {**res, "suggested_action": res["action"], "action": "allow"}

        # Phase 3: fold a clean login into the user's OWN profile. Safe to apply a moment later -
        # it adds to that one account's normal pattern and touches no shared counter, which is
        # exactly what /v1/confirm does minutes afterwards.
        if trusted:
            with ctx["lock"]:
                ctx["state"].update(row, trusted=True, counters=False)

        # The audit write stays synchronous, just no longer inside the scoring lock. Queueing it
        # would let /v1/confirm or /v1/verdict run before the event reached the database and get a
        # 404 for an event_id this call had already returned. At 0.3 ms it is not worth that race;
        # Store has its own lock, so these serialise among themselves without blocking scoring.
        event_id = uuid.uuid4().hex
        version = model.meta.get("version")
        ctx["store"].add_event(event_id, req.user_id, raw, row, feats, res, reasons, trusted,
                               model_version=version)
        metrics.scored.labels(tier=res["tier"], model_tier=res["model_tier"],
                              warming=str(bool(warming)).lower()).inc()
        for rule in res.get("policy", ()):
            metrics.policy.labels(rule=rule).inc()
        metrics.latency.observe(time.perf_counter() - t_start)
        return {"event_id": event_id, **res, "warming_up": warming, "reasons": reasons,
                "model_version": version}

    def _trust(event_id):
        ev = ctx["store"].get_event(event_id)
        if ev is None: raise HTTPException(404, "unknown event_id")
        with ctx["lock"]:
            if ctx["store"].mark_trusted(event_id):
                row = json.loads(ev["row_json"])
                # A replayed confirmation must not move the window clock; profile-only update.
                try:
                    ctx["state"].update(row, trusted=True, counters=False)
                except TimestampOutOfRange:
                    log.warning("confirmation for %s had an out-of-range timestamp; profile not updated", event_id)
                    return False
                return True
        return False

    @app.post("/v1/confirm/{event_id}", dependencies=[Depends(auth)])
    def confirm(event_id: str):
        """Call after the user PASSES the step-up check (OTP etc.) -> the login joins their normal profile."""
        ev = ctx["store"].get_event(event_id)
        if ev is None: raise HTTPException(404, "unknown event_id")
        if ev["tier"] == "High": raise HTTPException(409, "High-risk logins need an analyst verdict, not a user confirmation")
        row = json.loads(ev["row_json"])
        if not row.get("success"):
            # A failed attempt has nothing to confirm: folding it into the profile would let a caller
            # normalise an attacker's country/device by replaying confirmations for failed logins.
            raise HTTPException(409, "a failed login cannot be confirmed into the user's normal profile")
        return {"event_id": event_id, "added_to_profile": _trust(event_id)}

    @app.get("/v1/queue", dependencies=[Depends(auth)])
    def review_queue(limit: int = Query(50, ge=1, le=500)):
        return ctx["store"].queue(limit)

    @app.get("/v1/stats", dependencies=[Depends(auth)])
    def stats(hours: float = Query(24.0, gt=0, le=24 * 90)):
        """Counts for the analyst console: tier mix, queue depth, verdicts, model versions seen."""
        out = ctx["store"].stats(hours)
        st = ctx["state"]
        out["service"] = {
            "model_version": ctx["model"].meta.get("version"),
            "warming_up": st.glob_total < WARMUP_MIN_EVENTS,
            "logins_in_state": st.glob_total,
            "geoip_source": ctx["geo"].name,
            "timestamp_source": ctx["ts_source"],
            "state_backend": "redis" if not getattr(st, "needs_snapshot", True) else "in-process",
            "home_countries": sorted(ctx["policy"].home_countries) or None,
            "medium_threshold": ctx["model"].medium_thr,
            "high_threshold": ctx["model"].high_thr,
            "budget_medium_pct": C.HEAD_MEDIUM_BUDGET * 100,
            "budget_high_pct": C.HEAD_HIGH_BUDGET * 100,
        }
        return out

    @app.get("/v1/event/{event_id}", dependencies=[Depends(auth)])
    def event_detail(event_id: str):
        """One event, for the console's review pane."""
        ev = ctx["store"].get_event(event_id)
        if ev is None:
            raise HTTPException(404, "unknown event_id")
        from .store import _json_safe          # rows written before the NaN fix may hold bare NaN
        return {
            "event_id": ev["event_id"], "created_at": ev["created_at"], "user_id": ev["user_id"],
            "risk": ev["risk"], "tier": ev["tier"], "action": ev["action"],
            "trusted": bool(ev["trusted"]), "model_version": ev["model_version"],
            "reasons": json.loads(ev["reasons_json"] or "[]"),
            "raw": _json_safe(json.loads(ev["raw_json"] or "{}")),
            "features": _json_safe(json.loads(ev["features_json"] or "{}")),
        }

    @app.post("/v1/verdict", dependencies=[Depends(auth)])
    def verdict(v: VerdictRequest):
        """Analyst decision. Stored for the NEXT retraining run - the live model never changes by itself."""
        try:
            ctx["store"].add_verdict(v.event_id, v.verdict, v.analyst, v.note)
        except ValueError as e: raise HTTPException(422, str(e)) from None
        except KeyError: raise HTTPException(404, "unknown event_id") from None
        if v.verdict == "benign": _trust(v.event_id)
        return {"event_id": v.event_id, "verdict": v.verdict, "used_for_training": v.verdict != "suspicious"}

    @app.post("/v1/admin/reload-model", dependencies=[Depends(auth)])
    def reload_model():
        digest = verify_model(model_path, model_sha256)
        with ctx["lock"]:
            ctx["model"] = RiskModel.load(model_path)
            ctx["model_sha256"] = digest
        log.info("model reloaded: version=%s sha256=%s", ctx["model"].meta.get("version"), digest)
        return {"model_version": ctx["model"].meta.get("version"), "model_sha256": digest}

    @app.post("/v1/admin/forget-user", dependencies=[Depends(auth)])
    def forget_user(req: ForgetRequest):
        """Right to erasure: remove an account from the live profiles and delete its logged events."""
        with ctx["lock"]:
            values = ctx["store"].user_category_values(req.user_id)
            row = event_to_row({RC["ts"]: datetime.now(timezone.utc).isoformat(), RC["user"]: req.user_id,
                                RC["ip"]: "0.0.0.0", RC["country"]: None, RC["asn"]: None,
                                RC["browser"]: None, RC["os"]: None, RC["device"]: None,
                                RC["rtt"]: None, RC["success"]: True})
            state_result = ctx["state"].forget_user(row["user"], values)
            db_result = ctx["store"].forget_user(req.user_id)
            _snapshot_locked()
        _flush()                       # erasure must be durable before we report it done
        log.info("erased account %s: %s %s", req.user_id, state_result, db_result)
        return {"user_id": req.user_id, "state": state_result, "database": db_result}

    @app.post("/v1/admin/purge-raw", dependencies=[Depends(auth)])
    def purge_raw(days: int = Query(C.RAW_EVENT_RETENTION_DAYS, ge=1, le=3650)):
        """Delete the raw login payload (IP, user agent) from events older than `days`."""
        return {"events_purged": ctx["store"].purge_raw_older_than(days), "retention_days": days}

    @app.post("/v1/admin/prune-state", dependencies=[Depends(auth)])
    def prune_state(days: int = Query(C.STATE_RETENTION_MS // 86_400_000, ge=1, le=3650)):
        """Drop per-IP / per-ASN state idle for longer than `days`: bounds memory and enforces retention."""
        with ctx["lock"]:
            removed = ctx["state"].prune(retention_ms=days * 86_400_000)
            _snapshot_locked()
        _flush()
        return {"removed": removed, "retention_days": days}

    return app