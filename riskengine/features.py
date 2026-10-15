"""
Behavioural features for risk-based authentication. Every feature for a login is computed
from the user's / IP's PAST ONLY (never the current row or the future).

Two implementations of the SAME definitions:
  * build_features(df)   - vectorised, for training on tens of millions of rows
  * FeatureState         - one event at a time with a running history, for the live site
tests/test_parity.py proves they return identical numbers, so there is no train/serve skew.

v3: network behaviour is measured over RECENT windows (1 h / 24 h). All-time running counts were
dropped because their meaning drifts as data accumulates (see scripts/diagnose.py, section 4).
"""
from __future__ import annotations
import math
from collections import deque
import numpy as np
import pandas as pd
from .config import CATS

_C = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
TWO_PI = 2 * math.pi
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
WIN_MAX = DAY_MS
DEQUE_CAP = 200_000          # safety cap on events remembered per IP / ASN inside the window

FEATURES = (
    ["history_count_log"]
    + [f"new_{c}" for c in CATS]          # 1 = this user never used this value before
    + [f"freq_{c}" for c in CATS]         # share of this user's past logins that used this value
    + ["hour_dev", "hour_conc",           # distance from usual login hour, how regular the user is
       "since_last_log",                  # log(1 + hours since the user's previous login)
       "user_fail_prior_log",
       "ip_distinct_users_log",           # distinct accounts this IP touched before (credential stuffing)
       "success", "log_rtt",
       "asn_distinct_users_log",
       "browser_global_freq", "os_global_freq",   # rarity of the browser / OS string across ALL logins
       # --- v3: recent-window network behaviour ---
       "ip_n_1h_log", "ip_n_24h_log", "ip_fail_24h_log", "ip_fail_rate_24h",
       "asn_n_24h_log", "asn_fail_rate_24h"]
)


def mix_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (a.astype(np.uint64) * np.uint64(_C)) ^ b.astype(np.uint64)


def mix_py(a: int, b: int) -> int:
    return ((a * _C) & _MASK) ^ b


def _prior_count(key: np.ndarray) -> np.ndarray:
    """For each row: how many EARLIER rows share the same key (rows must be time-sorted)."""
    codes = pd.factorize(key)[0]
    return pd.Series(codes).groupby(codes).cumcount().to_numpy()


def _window_counts(codes: np.ndarray, ts: np.ndarray, failed: np.ndarray, window_ms: int):
    """For each row: number of EARLIER rows of the same group within the last `window_ms`
    (boundary inclusive), and how many of those failed. Rows must be time-sorted; ts relative (>=0)."""
    n = len(codes)
    if ts.max() >= (1 << 36) or codes.max() >= (1 << 26):
        raise ValueError("data span / group count too large for the packed window key")
    order = np.argsort(codes, kind="stable")                 # groups contiguous, time order kept inside
    c = codes[order].astype(np.int64)
    t = ts[order].astype(np.int64)
    key = (c << 36) | t
    left = np.searchsorted(key, (c << 36) | np.maximum(t - window_ms, 0), side="left")
    pos = np.arange(n)
    cs = np.concatenate(([0], np.cumsum(failed[order])))
    n_prior = np.empty(n, dtype=np.int64); f_prior = np.empty(n, dtype=np.int64)
    n_prior[order] = pos - left
    f_prior[order] = cs[pos] - cs[left]
    return n_prior, f_prior


def _hour_features(ms, uc, hist):
    """(hour_dev, hour_conc): distance from the user's usual login hour and how regular the user is."""
    h = np.maximum(hist, 1)
    ang = (ms % DAY_MS) / DAY_MS * TWO_PI
    s, c = np.sin(ang), np.cos(ang)
    g = pd.DataFrame({"u": uc, "s": s, "c": c}).groupby("u")
    m_s = (g["s"].cumsum().to_numpy() - s) / h
    m_c = (g["c"].cumsum().to_numpy() - c) / h
    d = np.abs(ang - np.arctan2(m_s, m_c)) % TWO_PI
    d = np.minimum(d, TWO_PI - d) / math.pi
    return np.where(hist > 0, d, np.nan), np.where(hist > 0, np.hypot(m_s, m_c), np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: prepared frame (see data.normalize_chunk), SORTED by ts_ms.
    Memory-lean: writes into one float32 table (column-major) and lets temporaries go out of scope."""
    n = len(df)
    X = np.empty((n, len(FEATURES)), dtype=np.float32, order="F")
    idx = {name: i for i, name in enumerate(FEATURES)}

    def put(name, arr):
        X[:, idx[name]] = arr                                  # cast to float32 on assignment

    u = df["user"].to_numpy()
    uc = pd.factorize(u)[0]
    hist = pd.Series(uc).groupby(uc).cumcount().to_numpy()
    put("history_count_log", np.log1p(hist))

    for c in CATS:
        prior = _prior_count(mix_np(u, df[c].to_numpy()))
        put(f"new_{c}", prior == 0)
        put(f"freq_{c}", np.where(hist > 0, prior / np.maximum(hist, 1), 0.0))
        del prior

    ms = df["ts_ms"].to_numpy()
    hd, hc = _hour_features(ms, uc, hist)
    put("hour_dev", hd); put("hour_conc", hc); del hd, hc
    put("since_last_log", np.log1p(np.clip(pd.Series(ms).groupby(uc).diff().to_numpy() / 3_600_000.0, 0, None)))

    failed = (~df["success"].to_numpy()).astype(np.int64)
    put("user_fail_prior_log", np.log1p(pd.Series(failed).groupby(uc).cumsum().to_numpy() - failed))
    del uc, hist
    ipc = pd.factorize(df["ip"].to_numpy())[0]
    pair_new = (_prior_count(mix_np(df["ip"].to_numpy(), u)) == 0).astype(np.int64)
    put("ip_distinct_users_log", np.log1p(pd.Series(pair_new).groupby(ipc).cumsum().to_numpy() - pair_new))
    del pair_new

    put("success", df["success"].to_numpy())
    put("log_rtt", np.log1p(np.clip(np.nan_to_num(df["rtt"].to_numpy(dtype=float), nan=0.0), 0, None)))

    ac = pd.factorize(df["asn"].to_numpy())[0]
    apair_new = (_prior_count(mix_np(df["asn"].to_numpy(), u)) == 0).astype(np.int64)
    put("asn_distinct_users_log", np.log1p(pd.Series(apair_new).groupby(ac).cumsum().to_numpy() - apair_new))
    del apair_new, u
    rows_before = np.arange(n)
    for col in ("browser", "os"):
        prior = _prior_count(df[col].to_numpy())
        put(f"{col}_global_freq", np.where(rows_before > 0, prior / np.maximum(rows_before, 1), 0.0))
        del prior
    del rows_before

    # ---- v3: recent-window network behaviour ----
    rel = ms - ms.min()
    n1, _ = _window_counts(ipc, rel, failed, HOUR_MS)
    put("ip_n_1h_log", np.log1p(n1)); del n1, _
    n24, f24 = _window_counts(ipc, rel, failed, DAY_MS)
    put("ip_n_24h_log", np.log1p(n24)); put("ip_fail_24h_log", np.log1p(f24))
    put("ip_fail_rate_24h", np.where(n24 > 0, f24 / np.maximum(n24, 1), 0.0)); del n24, f24, ipc
    an24, af24 = _window_counts(ac, rel, failed, DAY_MS)
    put("asn_n_24h_log", np.log1p(an24))
    put("asn_fail_rate_24h", np.where(an24 > 0, af24 / np.maximum(an24, 1), 0.0))
    return pd.DataFrame(X, index=df.index, columns=FEATURES, copy=False)


class FeatureState:
    """Running history for the live system (replaces the full table scan used in training).

    features(row)  -> dict of the FEATURES for this login (does NOT change state)
    update(row, trusted, counters) -> record the login. Network/attacker signals (failures, accounts
        per IP, recent windows) always update. A user's own 'normal' profile (countries, devices,
        hours, last login) is only updated when trusted=True, so an attacker cannot teach the
        system that their country/device is normal.
    """

    def __init__(self):
        self.users: dict[int, list] = {}     # user -> [n, sum_sin, sum_cos, last_ms, fails]
        self.seen = {c: {} for c in CATS}    # per category: mix(user,value) -> count
        self.ip_users: dict[int, set] = {}
        self.asn_users: dict[int, set] = {}
        self.glob_total = 0
        self.glob = {"browser": {}, "os": {}}
        self.win_ip: dict[int, deque] = {}   # ip  -> deque[(ts_ms, failed)] for the last 24 h
        self.win_asn: dict[int, deque] = {}

    def __setstate__(self, d):             # lets an older saved state.pkl load after an upgrade
        self.__dict__.update(d)
        self.__dict__.setdefault("ip_users", {}); self.__dict__.setdefault("asn_users", {})
        self.__dict__.setdefault("glob_total", 0); self.__dict__.setdefault("glob", {"browser": {}, "os": {}})
        self.__dict__.setdefault("win_ip", {}); self.__dict__.setdefault("win_asn", {})

    @staticmethod
    def _window(dq, t: int):
        """(# in last hour, # in last 24 h, # failed in last 24 h) among events already recorded."""
        n1 = n24 = f24 = 0
        if dq:
            lo1, lo24 = t - HOUR_MS, t - DAY_MS
            for ts, fl in reversed(dq):
                if ts < lo24:
                    break
                n24 += 1; f24 += fl
                if ts >= lo1:
                    n1 += 1
        return n1, n24, f24

    def features(self, r: dict) -> dict:
        u = int(r["user"])
        t = int(r["ts_ms"])
        st = self.users.get(u)
        n = st[0] if st else 0
        f = {"history_count_log": math.log1p(n)}
        for c in CATS:
            prior = self.seen[c].get(mix_py(u, int(r[c])), 0)
            f[f"new_{c}"] = 1.0 if prior == 0 else 0.0
            f[f"freq_{c}"] = prior / n if n > 0 else 0.0
        ang = (t % DAY_MS) / DAY_MS * TWO_PI
        if n > 0:
            m_s, m_c = st[1] / n, st[2] / n
            d = abs(ang - math.atan2(m_s, m_c)) % TWO_PI
            f["hour_dev"] = min(d, TWO_PI - d) / math.pi
            f["hour_conc"] = math.hypot(m_s, m_c)
            f["since_last_log"] = math.log1p(max((t - st[3]) / 3_600_000.0, 0.0))
        else:
            f["hour_dev"] = f["hour_conc"] = f["since_last_log"] = float("nan")
        f["user_fail_prior_log"] = math.log1p(st[4] if st else 0)
        ip, asn = int(r["ip"]), int(r["asn"])
        f["ip_distinct_users_log"] = math.log1p(len(self.ip_users.get(ip, ())))
        f["success"] = 1.0 if r["success"] else 0.0
        rtt = r.get("rtt")
        rtt = 0.0 if rtt is None or (isinstance(rtt, float) and math.isnan(rtt)) else max(float(rtt), 0.0)
        f["log_rtt"] = math.log1p(rtt)
        f["asn_distinct_users_log"] = math.log1p(len(self.asn_users.get(asn, ())))
        for col in ("browser", "os"):
            f[f"{col}_global_freq"] = self.glob[col].get(int(r[col]), 0) / self.glob_total if self.glob_total > 0 else 0.0
        n1, n24, f24 = self._window(self.win_ip.get(ip), t)
        an1, an24, af24 = self._window(self.win_asn.get(asn), t)
        f["ip_n_1h_log"] = math.log1p(n1)
        f["ip_n_24h_log"] = math.log1p(n24)
        f["ip_fail_24h_log"] = math.log1p(f24)
        f["ip_fail_rate_24h"] = f24 / n24 if n24 > 0 else 0.0
        f["asn_n_24h_log"] = math.log1p(an24)
        f["asn_fail_rate_24h"] = af24 / an24 if an24 > 0 else 0.0
        return f

    def update(self, r: dict, trusted: bool = True, counters: bool = True) -> None:
        """counters: record network/attacker signals (always safe to record).
        trusted:  add this login to the user's own 'normal' profile."""
        u, ip, asn, t = int(r["user"]), int(r["ip"]), int(r["asn"]), int(r["ts_ms"])
        failed = 0 if r["success"] else 1
        st = self.users.setdefault(u, [0, 0.0, 0.0, None, 0])
        if counters:
            st[4] += failed
            self.ip_users.setdefault(ip, set()).add(u)
            self.asn_users.setdefault(asn, set()).add(u)
            self.glob_total += 1
            for col in ("browser", "os"):
                k = int(r[col]); self.glob[col][k] = self.glob[col].get(k, 0) + 1
            for table, key in ((self.win_ip, ip), (self.win_asn, asn)):
                dq = table.setdefault(key, deque(maxlen=DEQUE_CAP))
                dq.append((t, failed))
                lo = t - WIN_MAX
                while dq and dq[0][0] < lo:
                    dq.popleft()
        if trusted:
            for c in CATS:
                k = mix_py(u, int(r[c]))
                self.seen[c][k] = self.seen[c].get(k, 0) + 1
            ang = (t % DAY_MS) / DAY_MS * TWO_PI
            st[0] += 1
            st[1] += math.sin(ang)
            st[2] += math.cos(ang)
            st[3] = t