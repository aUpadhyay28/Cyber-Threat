"""Risk scoring service. Run:
   RISK_API_KEY=change-me uvicorn riskengine.api:create_app --factory --port 8000
Keep this service INTERNAL (only your web backend calls it) and never show scores/reasons to end users."""
from __future__ import annotations
import hmac, os, pickle, threading, uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .config import RAW_COLUMNS as RC, VERDICTS
from .data import event_to_row
from .features import FeatureState
from .modeling import RiskModel, explain
from .store import Store

SAVE_EVERY = 200
# The network-volume features only mean something once the system has seen real traffic. Until it has seen
# this many logins, results are log-only: action is forced to "allow" and the would-be action is reported.
WARMUP_MIN_EVENTS = int(os.getenv("RISK_WARMUP_MIN_EVENTS", "10000"))


class ScoreRequest(BaseModel):
    user_id: str
    timestamp: datetime
    ip: str
    country: Optional[str] = None      # in production derive from IP server-side (MaxMind GeoLite2)
    asn: Optional[Union[int, str]] = None
    browser: Optional[str] = None
    os: Optional[str] = None
    device: Optional[str] = None
    rtt_ms: Optional[float] = None
    success: bool = True


class VerdictRequest(BaseModel):
    event_id: str
    verdict: str = Field(description="benign | suspicious | confirmed_malicious")
    analyst: str = ""
    note: str = ""


def create_app(model_path=None, state_path=None, db_path=None, api_key=None) -> FastAPI:
    model_path = model_path or os.getenv("RISK_MODEL_PATH", "models/latest.joblib")
    state_path = state_path or os.getenv("RISK_STATE_PATH", "data/state.pkl")
    db_path = db_path or os.getenv("RISK_DB_PATH", "data/risk.db")
    api_key = api_key or os.getenv("RISK_API_KEY")
    if not api_key:
        raise RuntimeError("Set RISK_API_KEY (the secret your web backend sends in the X-API-Key header)")

    ctx = {"lock": threading.Lock(), "since_save": 0}

    def save_state():
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{state_path}.tmp"
        with open(tmp, "wb") as fh: pickle.dump(ctx["state"], fh)
        os.replace(tmp, state_path); ctx["since_save"] = 0

    @asynccontextmanager
    async def lifespan(_):
        ctx["model"] = RiskModel.load(model_path)
        ctx["state"] = pickle.load(open(state_path, "rb")) if Path(state_path).exists() else FeatureState()
        ctx["store"] = Store(db_path)
        yield
        with ctx["lock"]: save_state()

    app = FastAPI(title="Adaptive Login Risk Service", lifespan=lifespan)

    def auth(x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(401, "invalid API key")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health")
    def health():
        m = ctx["model"].meta
        return {"status": "ok", "model_version": m.get("version"), "users_in_state": len(ctx["state"].users),
                "logins_in_state": ctx["state"].glob_total, "warming_up": ctx["state"].glob_total < WARMUP_MIN_EVENTS}

    @app.post("/v1/score", dependencies=[Depends(auth)])
    def score(req: ScoreRequest):
        raw = {RC["ts"]: req.timestamp.isoformat(), RC["user"]: req.user_id, RC["ip"]: req.ip,
               RC["country"]: req.country, RC["asn"]: req.asn, RC["browser"]: req.browser,
               RC["os"]: req.os, RC["device"]: req.device, RC["rtt"]: req.rtt_ms, RC["success"]: req.success}
        row = event_to_row(raw)
        with ctx["lock"]:
            feats = ctx["state"].features(row)
            warming = ctx["state"].glob_total < WARMUP_MIN_EVENTS
            res = ctx["model"].score_features(feats)
            reasons = explain(feats) if res["tier"] != "Low" else []
            trusted = res["tier"] == "Low" and req.success      # only clean successful logins shape the 'normal' profile
            if warming:                                         # log-only until enough traffic has been seen
                res = {**res, "suggested_action": res["action"], "action": "allow"}
            ctx["state"].update(row, trusted=trusted, counters=True)
            event_id = uuid.uuid4().hex
            ctx["store"].add_event(event_id, req.user_id, {**raw, "ip": req.ip}, row, feats, res, reasons, trusted)
            ctx["since_save"] += 1
            if ctx["since_save"] >= SAVE_EVERY: save_state()
        return {"event_id": event_id, **res, "warming_up": warming, "reasons": reasons,
                "model_version": ctx["model"].meta.get("version")}

    def _trust(event_id):
        ev = ctx["store"].get_event(event_id)
        if ev is None: raise HTTPException(404, "unknown event_id")
        import json
        with ctx["lock"]:
            if ctx["store"].mark_trusted(event_id):
                ctx["state"].update(json.loads(ev["row_json"]), trusted=True, counters=False)
                return True
        return False

    @app.post("/v1/confirm/{event_id}", dependencies=[Depends(auth)])
    def confirm(event_id: str):
        """Call after the user PASSES the step-up check (OTP etc.) -> the login joins their normal profile."""
        ev = ctx["store"].get_event(event_id)
        if ev is None: raise HTTPException(404, "unknown event_id")
        if ev["tier"] == "High": raise HTTPException(409, "High-risk logins need an analyst verdict, not a user confirmation")
        return {"event_id": event_id, "added_to_profile": _trust(event_id)}

    @app.get("/v1/queue", dependencies=[Depends(auth)])
    def queue(limit: int = Query(50, ge=1, le=500)):
        return ctx["store"].queue(limit)

    @app.post("/v1/verdict", dependencies=[Depends(auth)])
    def verdict(v: VerdictRequest):
        """Analyst decision. Stored for the NEXT retraining run - the live model never changes by itself."""
        try:
            ctx["store"].add_verdict(v.event_id, v.verdict, v.analyst, v.note)
        except ValueError as e: raise HTTPException(422, str(e))
        except KeyError: raise HTTPException(404, "unknown event_id")
        if v.verdict == "benign": _trust(v.event_id)
        return {"event_id": v.event_id, "verdict": v.verdict, "used_for_training": v.verdict != "suspicious"}

    @app.post("/v1/admin/reload-model", dependencies=[Depends(auth)])
    def reload_model():
        with ctx["lock"]: ctx["model"] = RiskModel.load(model_path)
        return {"model_version": ctx["model"].meta.get("version")}

    return app