"""Generate a fake login CSV with the SAME columns as the RBA dataset, for smoke-testing the
pipeline before you run on the real data.  python scripts/make_mock_data.py --out data/mock.csv --n 200000"""
import argparse
import numpy as np
import pandas as pd

def make(n=200_000, n_users=3000, seed=0, span_days=330) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    countries = np.array(["NO", "SE", "DK", "US", "DE", "GB", "IN", "FR"])
    home = rng.choice(countries[:3], n_users, p=[.7, .2, .1])
    user_hour = rng.integers(7, 22, n_users)
    user_asn = rng.integers(1000, 9000, n_users)
    user_dev = rng.integers(0, 3, n_users)
    user_os = rng.integers(0, 4, n_users)
    user_ip = rng.integers(0, 2**31, n_users)
    user_w = rng.pareto(1.2, n_users) + 1; user_w /= user_w.sum()
    t0 = pd.Timestamp("2020-02-01").value // 10**6
    ts = np.sort(rng.integers(0, span_days * 86_400_000, n)) + t0
    uid = rng.choice(n_users, n, p=user_w)
    hour_ms = (np.clip(rng.normal(user_hour[uid], 2.5), 0, 23.9) * 3_600_000).astype("int64")
    ts = ts - (ts % 86_400_000) + hour_ms

    country = home[uid].astype(object)
    asn = user_asn[uid].copy()
    ipn = (user_ip[uid] + rng.integers(0, 3, n)).astype("int64")     # sticky IP, occasional churn
    ip = np.array([f"{(x>>24)&255}.{(x>>16)&255}.{(x>>8)&255}.{x&255}" for x in ipn], dtype=object)
    dev = np.array(["desktop", "mobile", "tablet"])[user_dev[uid]].astype(object)
    os_ = np.array(["Windows 10", "iOS 13", "Android 10", "Mac OS X 10"])[user_os[uid]].astype(object)
    br = np.where(os_ == "iOS 13", "Safari 13", "Chrome 80").astype(object)
    success = rng.random(n) > 0.04
    rtt = rng.integers(20, 900, n).astype(float)

    # occasional legitimate novelty (travel / new phone / new network) -> the model must tolerate it
    trav = rng.random(n) < 0.02
    country[trav] = rng.choice(countries, trav.sum()); asn[trav] = rng.integers(1000, 9000, trav.sum())
    newdev = rng.random(n) < 0.02
    dev[newdev] = "mobile"; os_[newdev] = "Android 10"; br[newdev] = "Chrome 80"

    # attackers: a few IPs hammering many accounts from foreign networks (credential stuffing) + rare takeovers
    attack_ips = [f"{a}.{b}.{c}.{d}" for a, b, c, d in rng.integers(11, 220, (60, 4))]
    atk = rng.random(n) < 0.02
    ip[atk] = rng.choice(attack_ips, atk.sum())
    country[atk] = rng.choice(["RU", "CN", "NG", "BR", "VN"], atk.sum())
    asn[atk] = rng.integers(20000, 60000, atk.sum())
    dev[atk] = "unknown"; os_[atk] = rng.choice(["Linux", "Windows 7"], atk.sum()); br[atk] = "Python-urllib 3"
    success[atk] = rng.random(atk.sum()) < 0.15
    ato = atk & success & (rng.random(n) < 0.3)
    night = atk & (rng.random(n) < 0.5)
    ts = np.where(night, ts - (ts % 86_400_000) + rng.integers(0, 5, n) * 3_600_000, ts)
    order = np.argsort(ts, kind="stable")
    df = pd.DataFrame({
        "Login Timestamp": pd.to_datetime(ts, unit="ms").strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3],
        "User ID": uid, "Round-Trip Time [ms]": rtt, "IP Address": ip, "Country": country, "ASN": asn,
        "Browser Name and Version": br, "OS Name and Version": os_, "Device Type": dev,
        "Login Successful": success, "Is Attack IP": atk, "Is Account Takeover": ato,
    }).iloc[order].reset_index(drop=True)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/mock.csv")
    ap.add_argument("--n", type=int, default=200_000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); d = make(a.n, seed=a.seed); d.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(d):,} rows, attack rate {d['Is Attack IP'].mean():.2%}, ATO {d['Is Account Takeover'].sum()}")
