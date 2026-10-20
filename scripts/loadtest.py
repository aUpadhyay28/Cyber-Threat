"""Measure what the scoring service can actually take, before changing anything to make it faster.

   python scripts/loadtest.py --key "$RISK_API_KEY" --concurrency 32 --duration 20

Reports throughput and latency percentiles for POST /v1/score. The number that matters for a login
flow is not the mean - it is p99, because that is the tail your users feel, and the caller needs a
timeout set above it or it will start failing open on healthy traffic.

Every request is a distinct user and a plausible IP, so the run exercises real feature computation
(dictionary growth, window deques) rather than hitting the same key repeatedly.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, asyncio, json, random, statistics, time

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--key", required=True)
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--duration", type=float, default=20.0, help="seconds of load")
ap.add_argument("--warmup", type=float, default=2.0, help="seconds discarded before measuring")
ap.add_argument("--users", type=int, default=5000, help="size of the rotating user pool")
ap.add_argument("--json", action="store_true")
a = ap.parse_args()

COUNTRIES = ["IN", "NO", "SE", "US", "DE", "GB"]
BROWSERS = ["Chrome 120", "Firefox 121", "Safari 17", "Edge 120"]
OSES = ["Windows 11", "Android 14", "iOS 17", "Ubuntu 22.04"]


def body(rng: random.Random) -> dict:
    u = rng.randrange(a.users)
    return {"user_id": f"load-user-{u}",
            "ip": f"10.{rng.randrange(256)}.{rng.randrange(256)}.{rng.randrange(1, 255)}",
            "country": rng.choice(COUNTRIES),
            "asn": rng.randrange(1000, 60000),
            "browser": rng.choice(BROWSERS),
            "os": rng.choice(OSES),
            "device": rng.choice(["desktop", "mobile", "tablet"]),
            "rtt_ms": float(rng.randrange(10, 400)),
            "success": rng.random() > 0.08}


async def worker(client: httpx.AsyncClient, wid: int, stop_at: float, measure_from: float, out: list, errs: dict):
    rng = random.Random(wid * 7919)
    headers = {"X-API-Key": a.key}
    while time.perf_counter() < stop_at:
        t0 = time.perf_counter()
        try:
            r = await client.post("/v1/score", json=body(rng), headers=headers)
            dt = time.perf_counter() - t0
            if r.status_code != 200:
                errs[r.status_code] = errs.get(r.status_code, 0) + 1
                continue
        except Exception as exc:                                  # noqa: BLE001 - any failure counts
            errs[type(exc).__name__] = errs.get(type(exc).__name__, 0) + 1
            continue
        if t0 >= measure_from:
            out.append(dt)


async def main():
    limits = httpx.Limits(max_connections=a.concurrency + 8, max_keepalive_connections=a.concurrency + 8)
    async with httpx.AsyncClient(base_url=a.url, limits=limits, timeout=30.0) as client:
        health_before = (await client.get("/health")).json()
        start = time.perf_counter()
        measure_from = start + a.warmup
        stop_at = measure_from + a.duration
        lat: list[float] = []
        errs: dict = {}
        await asyncio.gather(*[worker(client, i, stop_at, measure_from, lat, errs)
                               for i in range(a.concurrency)])
        health_after = (await client.get("/health")).json()

    if not lat:
        raise SystemExit(f"no successful requests were measured; errors: {errs}")
    lat_ms = sorted(x * 1000 for x in lat)
    pct = lambda p: lat_ms[min(int(len(lat_ms) * p / 100), len(lat_ms) - 1)]
    result = {
        "concurrency": a.concurrency,
        "measured_seconds": round(a.duration, 1),
        "requests_measured": len(lat_ms),
        "requests_per_second": round(len(lat_ms) / a.duration, 1),
        "errors": errs,
        "latency_ms": {"p50": round(pct(50), 1), "p90": round(pct(90), 1), "p95": round(pct(95), 1),
                       "p99": round(pct(99), 1), "max": round(lat_ms[-1], 1),
                       "mean": round(statistics.fmean(lat_ms), 1)},
        "logins_in_state": {"before": health_before.get("logins_in_state"),
                            "after": health_after.get("logins_in_state")},
        "model_version": health_after.get("model_version"),
    }
    if a.json:
        print(json.dumps(result, indent=2))
    else:
        L = result["latency_ms"]
        print(f"concurrency {a.concurrency} for {a.duration:g}s")
        print(f"  throughput   {result['requests_per_second']:>8} req/s   ({result['requests_measured']:,} measured)")
        print(f"  latency ms   p50 {L['p50']:<7} p90 {L['p90']:<7} p95 {L['p95']:<7} p99 {L['p99']:<7} max {L['max']}")
        print(f"  state grew   {result['logins_in_state']['before']:,} -> {result['logins_in_state']['after']:,} logins")
        if errs:
            print(f"  ERRORS       {errs}")


asyncio.run(main())
