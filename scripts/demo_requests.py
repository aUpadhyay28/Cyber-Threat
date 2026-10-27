"""Runs a demo scenario against the running scoring service and prints one table.
   (service must be running)   python scripts/demo_requests.py --key my-secret-123
Scenario: a normal user builds a profile -> logs in normally -> a takeover-like login (new country, network,
browser, OS, odd hour) -> credential stuffing (40 failed logins from one IP on 40 accounts) -> one more from that IP.
Uses only the standard library. Every run uses fresh user ids / IPs, so you can repeat it."""
import argparse, json, os, random, urllib.request
from datetime import datetime, timedelta, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--key", default=os.getenv("RISK_API_KEY", ""))
ap.add_argument("--background", type=int, default=300,
                help="ordinary logins to replay first, so the recent-network features are not all zero")
a = ap.parse_args()
run = random.randint(1000, 9999)
ip_home = f"203.0.{run % 200}.7"; ip_odd = f"198.51.{run % 200}.20"; ip_bot = f"192.0.{run % 200}.9"
# Every scripted time is in the PAST. The service refuses future-dated logins, because a timestamp
# beyond what it has already seen would clear the recent-activity windows an attacking IP is judged
# by (riskengine/features.py: TimestampOutOfRange).
D = (datetime.now(timezone.utc) - timedelta(days=8)).replace(hour=0, minute=0, second=0, microsecond=0)


def get(path):
    req = urllib.request.Request(a.url + path, headers={"X-API-Key": a.key})
    return json.load(urllib.request.urlopen(req))  # nosec B310


health = get("/health")
if health.get("timestamp_source") == "server":
    print("NOTE: the service is running with RISK_TIMESTAMP_SOURCE=server, so it timestamps every\n"
          "      login with its own clock and the scripted times below are ignored - the whole\n"
          "      scenario collapses into a few seconds. The tiers still demonstrate both heads, but\n"
          "      for the intended 'odd hour' and 'days of history' effects restart the service with\n"
          "      RISK_TIMESTAMP_SOURCE=client (replay mode).\n")


def post(body):
    req = urllib.request.Request(a.url + "/v1/score", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "X-API-Key": a.key})
    return json.load(urllib.request.urlopen(req))  # nosec B310


def confirm(event_id):                      # the user passed the OTP -> login joins their normal profile
    req = urllib.request.Request(f"{a.url}/v1/confirm/{event_id}", b"", {"X-API-Key": a.key}, method="POST")
    return json.load(urllib.request.urlopen(req))  # nosec B310


def login(user, when, ip, country, asn, browser, os_, device, ok=True, rtt=80):
    return post({"user_id": user, "timestamp": when.isoformat().replace("+00:00", "Z"), "ip": ip, "country": country,
                 "asn": asn, "browser": browser, "os": os_, "device": device, "rtt_ms": rtt, "success": ok})


# Background traffic FIRST. The attack-IP head judges a login by how much recent activity its IP and
# network have produced, so against a service with no live traffic every login looks like it came
# from a dead network and scores High - the warm-up problem in the README, and the reason the old
# demo reported "block" for perfectly normal logins. Replaying ordinary logins from many accounts on
# the home network gives those features something real to measure.
if a.background:
    print(f"replaying {a.background} ordinary background logins so the network features mean something ...")
    for i in range(a.background):
        login(f"bg-{run}-{i}", D + timedelta(days=5, minutes=i % 600),
              f"203.0.{run % 200}.{20 + i % 200}", "IN", 9829, "Chrome 120", "Windows 11", "desktop")
    print("done\n")

user = f"demo-user-{run}"; rows = []
home = dict(ip=ip_home, country="IN", asn=9829, browser="Chrome 120", os_="Windows 11", device="desktop")
blocked = []
def note(label, r, trusted_flow=False):
    if trusted_flow and r["tier"] == "Medium":
        confirm(r["event_id"]); label += "  (OTP passed)"
    if trusted_flow and r["tier"] == "High":
        blocked.append(label)
    # suggested_action is only present while the service is warming up; once it is past warm-up the
    # action it reports IS the action it would take.
    rows.append((label, r["tier"], r["ip_risk"], r["takeover_risk"], r.get("suggested_action", r["action"]),
                 "; ".join(r["reasons"])[:70], r["warming_up"]))

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
for lbl, tier, ip, tk, act, why, _warm in rows:
    print(f"{lbl.ljust(w)}  {tier:<7} {ip:>7}  {('-' if tk is None else tk):>13}  {act:<18}  {why}")
if blocked:
    print(f"\nNOTE: {len(blocked)} normal login(s) scored High, so they were NOT added to the profile and the takeover score stays '-'.\n"
          "This is what a freshly started system with almost no traffic history does (its network features see no recent traffic).\n"
          "Run the script again after the service has handled more logins, or replay more background traffic first.")
print("\nwarming_up =", rows[-1][-1], "(true = the service only LOGS these decisions until it has seen enough traffic)")
