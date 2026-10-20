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
CREATE INDEX IF NOT EXISTS ix_events_user ON events(user_id);
CREATE INDEX IF NOT EXISTS ix_events_created ON events(created_at);
"""
# Added after the first release; SQLite has no "ADD COLUMN IF NOT EXISTS", so this is applied by hand.
# model_version records WHICH model made each decision. Without it, once you retrain you can no longer
# answer "what scored this login?", which is the first question asked in any review of a bad decision.
_MIGRATIONS = [("events", "model_version", "ALTER TABLE events ADD COLUMN model_version TEXT")]
_now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(obj):
    """Replace NaN/Infinity with None before storing.

    Python's json.dumps happily writes a bare `NaN` literal, which is NOT valid JSON: every strict
    parser rejects it, including the one FastAPI uses to serialise a response. Features are full of
    NaN by design (a first-ever login has no "hours since last login"), so the audit log was
    accumulating rows that could be written but not read back. Storing null instead round-trips
    correctly - training_rows already turns a missing value back into NaN when it rebuilds the
    training matrix.
    """
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class Store:
    def __init__(self, path: str = "data/risk.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(_SCHEMA)
            try:                                       # write-ahead logging: readers no longer block the writer
                self.db.execute("PRAGMA journal_mode=WAL")
                self.db.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.DatabaseError:
                pass                                   # e.g. an in-memory or read-only database
            for table, column, ddl in _MIGRATIONS:
                cols = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
                if column not in cols:
                    self.db.execute(ddl)
            self.db.commit()

    def add_event(self, event_id, user_id, raw, row, feats, res, reasons, trusted, model_version=None):
        with self.lock:
            self.db.execute(
                "INSERT INTO events(event_id,created_at,user_id,raw_json,row_json,features_json,"
                "risk,tier,action,reasons_json,trusted,model_version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, _now(), user_id, json.dumps(_json_safe(raw), default=str),
                 json.dumps(_json_safe(row)), json.dumps(_json_safe(feats)),
                 res["risk_score"], res["tier"], res["action"],
                 json.dumps(reasons), int(trusted), model_version))
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
        # Existence first, so an unknown event_id still reports "not found" rather than being
        # masked by the attribution check below.
        if self.get_event(event_id) is None:
            raise KeyError(event_id)
        # Verdicts are training data at 20x weight, so an unattributed one is an unauditable change
        # to the next model. If every verdict carries an identity, a single account suddenly marking
        # thousands of attacks "benign" is visible in verdict_counts_by_analyst(); without it, the
        # poisoning is indistinguishable from ordinary review.
        if not str(analyst).strip():
            raise ValueError("verdict must record which analyst made it")
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

    # ------------------------------------------------------------------ privacy / retention
    def user_category_values(self, user_id: str) -> dict:
        """The hashed values this account used per category, read back from its logged events.

        FeatureState keys its per-user tables by a non-invertible mix(user, value), so erasing an
        account needs the values it used. They are in the stored prepared rows.
        """
        from .config import CATS
        with self.lock:
            rows = self.db.execute("SELECT row_json FROM events WHERE user_id=?", (user_id,)).fetchall()
        out = {c: set() for c in CATS}
        for r in rows:
            row = json.loads(r["row_json"])
            for c in CATS:
                if row.get(c) is not None:
                    out[c].add(int(row[c]))
        return {c: sorted(v) for c, v in out.items()}

    def forget_user(self, user_id: str) -> dict:
        """Delete every logged event (and verdict) for one account - right to erasure."""
        with self.lock:
            n_v = self.db.execute(
                "DELETE FROM verdicts WHERE event_id IN (SELECT event_id FROM events WHERE user_id=?)",
                (user_id,)).rowcount
            n_e = self.db.execute("DELETE FROM events WHERE user_id=?", (user_id,)).rowcount
            self.db.commit()
        return {"events_deleted": max(n_e, 0), "verdicts_deleted": max(n_v, 0)}

    def purge_raw_older_than(self, days: int) -> int:
        """Strip the raw login payload (IP address, user agent strings) from events older than `days`.

        The derived rows, features and verdicts are kept, so the audit trail and training data
        survive; only the directly identifying copy goes. Implements the retention period the README
        promises, which previously had no mechanism behind it at all.
        """
        from datetime import timedelta
        # Full precision on purpose: created_at is stored truncated to seconds, so a cutoff that is
        # also truncated would skip an event written in the same second (days=0 would purge nothing).
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self.lock:
            cur = self.db.execute(
                "UPDATE events SET raw_json=? WHERE created_at < ? AND raw_json IS NOT NULL AND raw_json != ?",
                (json.dumps({"purged": True}), cutoff, json.dumps({"purged": True})))
            self.db.commit()
        return max(cur.rowcount, 0)

    def stats(self, hours: float = 24.0) -> dict:
        """Counts the analyst console needs: tier mix, decisions by model vs policy, queue depth.

        The tier mix is the one external signal that distinguishes a working engine from one that
        has quietly started scoring everything the same way, so it is the first thing shown.
        """
        from datetime import timedelta
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self.lock:
            tiers = self.db.execute(
                "SELECT tier, COUNT(*) n FROM events WHERE created_at >= ? GROUP BY tier", (since,)).fetchall()
            total = self.db.execute("SELECT COUNT(*) n FROM events WHERE created_at >= ?", (since,)).fetchone()["n"]
            pending = self.db.execute(
                "SELECT COUNT(*) n FROM events e LEFT JOIN verdicts v ON v.event_id=e.event_id "
                "WHERE v.event_id IS NULL AND e.tier IN ('Medium','High')").fetchone()["n"]
            verdicts = self.db.execute(
                "SELECT verdict, COUNT(*) n FROM verdicts GROUP BY verdict").fetchall()
            analysts = self.db.execute(
                "SELECT analyst, COUNT(*) n FROM verdicts GROUP BY analyst ORDER BY n DESC LIMIT 10").fetchall()
            versions = self.db.execute(
                "SELECT model_version, COUNT(*) n FROM events WHERE created_at >= ? "
                "GROUP BY model_version", (since,)).fetchall()
        by_tier = {r["tier"]: r["n"] for r in tiers}
        return {
            "window_hours": hours,
            "events": total,
            "by_tier": {t: by_tier.get(t, 0) for t in ("Low", "Medium", "High")},
            "flag_rate_pct": {
                "medium_plus": round((by_tier.get("Medium", 0) + by_tier.get("High", 0)) / total * 100, 3) if total else 0.0,
                "high": round(by_tier.get("High", 0) / total * 100, 3) if total else 0.0,
            },
            "queue_pending": pending,
            "verdicts": {r["verdict"]: r["n"] for r in verdicts},
            "verdicts_by_analyst": {(r["analyst"] or "(unattributed)"): r["n"] for r in analysts},
            "model_versions_seen": {(r["model_version"] or "(unrecorded)"): r["n"] for r in versions},
        }

    def verdict_counts_by_analyst(self) -> dict:
        """Who has been recording verdicts. A single identity suddenly producing thousands of
        'benign' calls is what poisoning the feedback loop looks like from the outside."""
        with self.lock:
            rows = self.db.execute(
                "SELECT analyst, verdict, COUNT(*) n FROM verdicts GROUP BY analyst, verdict").fetchall()
        out: dict = {}
        for r in rows:
            out.setdefault(r["analyst"] or "(unattributed)", {})[r["verdict"]] = r["n"]
        return out

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