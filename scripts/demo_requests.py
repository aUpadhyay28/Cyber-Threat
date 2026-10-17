"""Runs a demo scenario against the running scoring service and prints one table.
   (service must be running)   python scripts/demo_requests.py --key my-secret-123
Scenario: a normal user builds a profile -> logs in normally -> a takeover-like login (new country, network,
browser, OS, odd hour) -> credential stuffing (40 failed logins from one IP on 40 accounts) -> one more from that IP.
Uses only the standard library. Every run uses fresh user ids / IPs, so you can repeat it."""
import argparse, json, os, random, urllib.request
from datetime import datetime, timedelta, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--key", default=os.getenv("RISK_API_KEY", ""))
a = ap.parse_args()
run = random.randint(1000, 9999)
ip_home = f"203.0.{run % 200}.7"; ip_odd = f"198.51.{run % 200}.20"; ip_bot = f"192.0.{run % 200}.9"
D = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def post(body):
    req = urllib.request.Request(a.url + "/v1/score", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "X-API-Key": a.key})
    return json.load(urllib.request.urlopen(req))


def confirm(event_id):                      # the user passed the OTP -> login joins their normal profile
    req = urllib.request.Request(f"{a.url}/v1/confirm/{event_id}", b"", {"X-API-Key": a.key}, method="POST")
    return json.load(urllib.request.urlopen(req))


def login(user, when, ip, country, asn, browser, os_, device, ok=True, rtt=80):
    return post({"user_id": user, "timestamp": when.isoformat().replace("+00:00", "Z"), "ip": ip, "country": country,
                 "asn": asn, "browser": browser, "os": os_, "device": device, "rtt_ms": rtt, "success": ok})


user = f"demo-user-{run}"; rows = []
home = dict(ip=ip_home, country="IN", asn=9829, browser="Chrome 120", os_="Windows 11", device="desktop")
blocked = []
def note(label, r, trusted_flow=False):
    if trusted_flow and r["tier"] == "Medium":
        confirm(r["event_id"]); label += "  (OTP passed)"
    if trusted_flow and r["tier"] == "High":
        blocked.append(label)
    rows.append((label, r["tier"], r["ip_risk"], r["takeover_risk"], r["suggested_action"], "; ".join(r["reasons"])[:70], r["warming_up"]))

for i in range(4, 0, -1):
    note(f"1. normal login, {i} days ago", login(user, D - timedelta(days=i) + timedelta(hours=9, minutes=30), **home), True)
note("2. normal login today 09:30", login(user, D + timedelta(hours=9, minutes=30), **home), True)
note("3. TAKEOVER-LIKE login (new country/network/browser/OS, 03:10)",
     login(user, D + timedelta(days=1, hours=3, minutes=10), ip_odd, "RU", 12389, "Firefox 100", "Linux", "desktop", rtt=300))
for i in range(40):
    login(f"victim-{run}-{i}", D + timedelta(days=1, hours=4, minutes=i), ip_bot, "RU", 12389, "Python-urllib 3", "Linux", "unknown", ok=False, rtt=50)
note("4. login from the IP that just hit 40 accounts", login(f"victim-{run}-99", D + timedelta(days=1, hours=5), ip_bot, "RU", 12389,
     "Python-urllib 3", "Linux", "unknown", ok=True, rtt=50))

w = max(len(r[0]) for r in rows)
print(f"{'step'.ljust(w)}  tier    ip_risk  takeover_risk  would-be action     reasons")
for lbl, tier, ip, tk, act, why, warm in rows:
    print(f"{lbl.ljust(w)}  {tier:<7} {ip:>7}  {('-' if tk is None else tk):>13}  {act:<18}  {why}")
if blocked:
    print(f"\nNOTE: {len(blocked)} normal login(s) scored High, so they were NOT added to the profile and the takeover score stays '-'.\n"
          "This is what a freshly started system with almost no traffic history does (its network features see no recent traffic).\n"
          "Run the script again after the service has handled more logins, or replay more background traffic first.")
print("\nwarming_up =", rows[-1][-1], "(true = the service only LOGS these decisions until it has seen enough traffic)")