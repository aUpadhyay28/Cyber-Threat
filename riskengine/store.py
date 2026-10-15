"""SQLite store for scored events and analyst verdicts. Only analyst-validated rows
(benign / confirmed_malicious) can ever become training data - 'suspicious' and unreviewed
events never do. This is the human-validation gate of the project."""
from __future__ import annotations
import json, sqlite3, threading
from datetime import datetime, timezone
import pandas as pd
from .config import VERDICTS, LEARNABLE_VERDICTS
from .features import FEATURES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY, created_at TEXT, user_id TEXT, raw_json TEXT, row_json TEXT,
  features_json TEXT, risk REAL, tier TEXT, action TEXT, reasons_json TEXT, trusted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS verdicts(
  event_id TEXT PRIMARY KEY, verdict TEXT NOT NULL, analyst TEXT, note TEXT, created_at TEXT,
  FOREIGN KEY(event_id) REFERENCES events(event_id));
CREATE INDEX IF NOT EXISTS ix_events_tier ON events(tier, risk DESC);
"""
_now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str = "data/risk.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(_SCHEMA); self.db.commit()

    def add_event(self, event_id, user_id, raw, row, feats, res, reasons, trusted):
        with self.lock:
            self.db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, _now(), user_id, json.dumps(raw, default=str), json.dumps(row),
                 json.dumps(feats), res["risk_score"], res["tier"], res["action"],
                 json.dumps(reasons), int(trusted)))
            self.db.commit()

    def get_event(self, event_id):
        with self.lock:
            r = self.db.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        return dict(r) if r else None

    def mark_trusted(self, event_id) -> bool:
        """Atomically flip trusted 0->1; True only the first time (prevents double-counting)."""
        with self.lock:
            cur = self.db.execute("UPDATE events SET trusted=1 WHERE event_id=? AND trusted=0", (event_id,))
            self.db.commit()
            return cur.rowcount == 1

    def add_verdict(self, event_id, verdict, analyst="", note=""):
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        if self.get_event(event_id) is None:
            raise KeyError(event_id)
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO verdicts VALUES(?,?,?,?,?)",
                            (event_id, verdict, analyst, note, _now()))
            self.db.commit()

    def queue(self, limit=50):
        with self.lock:
            rows = self.db.execute(
                "SELECT e.event_id,e.created_at,e.user_id,e.risk,e.tier,e.action,e.reasons_json FROM events e "
                "LEFT JOIN verdicts v ON v.event_id=e.event_id WHERE v.event_id IS NULL AND e.tier IN ('Medium','High') "
                "ORDER BY e.risk DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def training_rows(self) -> pd.DataFrame:
        """Analyst-validated rows ready to append to the training matrix: FEATURES + label."""
        with self.lock:
            rows = self.db.execute(
                "SELECT e.features_json, v.verdict FROM events e JOIN verdicts v ON v.event_id=e.event_id "
                "WHERE v.verdict IN (%s)" % ",".join("?" * len(LEARNABLE_VERDICTS)),
                tuple(LEARNABLE_VERDICTS)).fetchall()
        recs = []
        for r in rows:
            f = json.loads(r["features_json"]); f = {k: (float("nan") if f.get(k) is None else f[k]) for k in FEATURES}
            f["label"] = LEARNABLE_VERDICTS[r["verdict"]]; recs.append(f)
        return pd.DataFrame(recs, columns=FEATURES + ["label"])
