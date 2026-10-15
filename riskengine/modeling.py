"""Two-head risk model.
  * IP/network head  - small gradient-boosted model of recent-window network behaviour (attack-IP label)
  * Takeover head    - logistic regression on 'does this login deviate from THIS user's own history'
Each head's score is converted to 0-100 = percentile among normal logins (fixed reference distribution
saved at training time), so a single login scores the same alone or in a batch. Final risk = max of heads."""
from __future__ import annotations
import math
import joblib
import numpy as np
from .config import TIER_ACTIONS
from .features import FEATURES

_F = {f: i for i, f in enumerate(FEATURES)}


def _ecdf(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentile (0-1) of x among the reference scores. Tied scores get the MIDDLE rank of their tie group:
    tree models give many identical scores, and using the top rank would flag every login in a big tie."""
    return (np.searchsorted(ref, x, side="left") + np.searchsorted(ref, x, side="right")) / 2.0 / len(ref)


class OffHead:
    """Stand-in for the takeover head when there are too few takeover examples to fit one (scores minimum)."""
    def decision_function(self, Z):
        return np.full(len(Z), -1e9)


class RiskModel:
    def __init__(self, ip_gb, ip_feats, ip_ref, dev_lr, dev_feats, dev_ref, warm_min,
                 medium_thr, high_thr, meta):
        self.ip_gb, self.ip_feats, self.ip_ref = ip_gb, list(ip_feats), ip_ref
        self.dev_lr, self.dev_feats, self.dev_ref = dev_lr, list(dev_feats), dev_ref
        self.warm_min, self.medium_thr, self.high_thr, self.meta = warm_min, medium_thr, high_thr, meta

    def heads(self, X: np.ndarray):
        """-> (ip_risk, takeover_risk, warm_mask); takeover_risk is 0 where history is too short."""
        X = np.asarray(X, dtype=np.float32)
        ip = 100.0 * _ecdf(self.ip_ref, self.ip_gb.predict_proba(X[:, [_F[f] for f in self.ip_feats]])[:, 1])
        warm = X[:, _F["history_count_log"]] >= math.log1p(self.warm_min)
        dev = np.zeros(len(X))
        if warm.any():
            Xd = np.nan_to_num(X[warm][:, [_F[f] for f in self.dev_feats]], nan=0.0)
            dev[warm] = 100.0 * _ecdf(self.dev_ref, self.dev_lr.decision_function(Xd))
        return ip, dev, warm

    def risk(self, X: np.ndarray) -> np.ndarray:
        ip, dev, _ = self.heads(X)
        return np.maximum(ip, dev)

    def tier(self, risk) -> np.ndarray:
        risk = np.asarray(risk)
        return np.where(risk >= self.high_thr, "High", np.where(risk >= self.medium_thr, "Medium", "Low"))

    def score_features(self, feats: dict) -> dict:
        x = np.array([[feats[k] for k in FEATURES]], dtype=np.float32)
        ip, dev, warm = self.heads(x)
        r = float(max(ip[0], dev[0])); t = str(self.tier(r))
        return {"risk_score": round(r, 2), "tier": t, "action": TIER_ACTIONS[t],
                "ip_risk": round(float(ip[0]), 2),
                "takeover_risk": round(float(dev[0]), 2) if warm[0] else None}

    def save(self, path):
        joblib.dump(self.__dict__, path, compress=3)

    @classmethod
    def load(cls, path) -> "RiskModel":
        d = joblib.load(path); obj = cls.__new__(cls); obj.__dict__.update(d); return obj


def explain(f: dict) -> list[str]:
    """Human-readable reasons derived from the features (for the analyst and the audit log).
    Never shown to the end user - that would teach attackers what to evade."""
    r = []
    cold = f["history_count_log"] < np.log1p(3)
    if cold:
        r.append("Little login history for this account (cold start)")
    else:
        if f["new_country"]: r.append("Country not used by this account before")
        if f["new_asn"]: r.append("Network (ASN) not used by this account before")
        if f["new_device"] or f["new_os"] or f["new_browser"]: r.append("New device / OS / browser for this account")
        if f["new_ip"] and not f["new_asn"]: r.append("New IP address (same network)")
        if f["hour_conc"] == f["hour_conc"] and f["hour_conc"] > 0.5 and f["hour_dev"] > 0.5:
            r.append("Login far outside this user's usual hours")
    if f["ip_distinct_users_log"] >= np.log1p(5): r.append("Same IP has touched many different accounts")
    if f["ip_fail_24h_log"] >= np.log1p(10): r.append("This IP has many failed logins in the last 24 hours")
    if f["ip_n_1h_log"] >= np.log1p(30): r.append("Unusually high login volume from this IP in the last hour")
    if f["user_fail_prior_log"] >= np.log1p(5): r.append("Account has many earlier failed logins")
    if not f["success"]: r.append("Failed login attempt")
    return r