# Taking this to production

This document exists because "is it production ready?" is not a yes/no question about code. The
engineering in this repository is finished. What remains is a rollout, and a rollout has an order
you cannot shuffle: you need your own traffic before you can calibrate, and you need calibration
before you can enforce.

Read the threat model in `SECURITY.md` alongside this.

---

## The part that is not code

Three things block enforcement and none of them can be written in Python.

**1. Your own logins.** The shipped model is fitted on the RBA dataset: 31 million synthetic logins
from a Norwegian university service in 2020–21, whose authors warn against production use. Its
notion of a normal browser string, a normal network and a normal hour comes from that population.
Point it at a different one and it flags almost everything — not a bug, a mismatch. You need weeks
of your own traffic, which is what shadow mode is for.

**2. A lawful basis and a privacy notice.** You are profiling login behaviour: IP addresses,
devices, approximate location, timing. Under India's DPDP Act that needs notice and a stated
purpose; if any users are in the EU it also needs a GDPR Article 35 assessment. The mechanisms are
here — `/v1/admin/forget-user`, `/v1/admin/purge-raw`, a configurable retention period — but the
policy behind them is a document you write and someone signs.

**3. GeoLite2 databases.** Until `RISK_GEOIP_MODE=maxmind` with real `.mmdb` files, `country` and
`asn` come from the caller, which means an attacker picks their own. `/health` tells you which mode
you are in; check it, do not assume.

---

## Rollout, in order

### Stage 0 — before any real user sees it

- [ ] privacy notice published, lawful basis recorded, retention period decided
- [ ] GeoLite2 Country + ASN databases installed; `/health` shows `geoip_source: maxmind`
- [ ] `RISK_API_KEY` from a secrets manager, not a file or a shell history
- [ ] the service is unreachable from the internet (only your login backend can call it)
- [ ] `RISK_TIMESTAMP_SOURCE=server` (the default) — never `client` in production
- [ ] model pinned: `scripts/pin_model.py`, digest shipped with the release
- [ ] `scripts/loadtest.py` run on your real hardware; caller timeout set above the measured p99
- [ ] `/metrics` scraped, alerts wired for the flag rates below
- [ ] decide fail-open vs fail-closed per flow and write it down (see `riskengine/client.py`)

### Stage 1 — shadow mode (weeks, not days)

Deploy with enforcement off and collect. This needs **no Redis and no second worker**: one process
is plenty, and the engine is log-only, so the fail-open question is moot.

```bash
RISK_WARMUP_MIN_EVENTS=1000000   # effectively "never enforce"
```

Wire `RiskClient` into your login flow, log `suggested_action`, and change nothing about what users
experience. Watch:

- the Medium+ and High rates against their 2% / 0.5% budgets — on the shipped model, expect these
  to be wildly high, because the model does not know your traffic yet. That is the measurement, not
  a failure.
- `risk_score_latency_seconds` p99 against your client timeout
- the `degraded` counter on the client — if it is non-zero you have an integration problem to fix
  before any of this matters

Stop when you have enough logins that your own attack and takeover cases are not single digits.
Thousands of accounts and several weeks is a reasonable floor; the takeover head needed 51 usable
cases to produce confidence intervals, and 51 was not enough to beat a one-line rule.

### Stage 2 — retrain on your own data

```bash
python scripts/import_history.py --csv your_logins.csv --state data/state.json.gz
python scripts/prepare_data.py --csv your_logins.csv --out data/prepared
python scripts/train.py --data data/prepared --out models --feedback-db data/risk.db
python scripts/pin_model.py --model models/latest.joblib
```

You will not have `Is Attack IP` or `Is Account Takeover` labels. That is the real problem with
stage 2, and it is why the analyst queue exists: `/v1/queue` and `/v1/verdict` are how you create
labels, at 20× weight in retraining. Budget analyst time for this; a risk engine with nobody
reviewing its queue decays into a random number generator.

Check in the new report that `legit_flagged_in_test_pct` lands near the budgets. If it does not,
the thresholds in `config.py` need moving, not the model.

### Stage 3 — enforce, narrowly

Turn on the home-country policy first, because it does not depend on the model being well
calibrated and it is explainable to a support team:

```bash
RISK_HOME_COUNTRIES=IN
```

Then enforce Medium (step-up) only. Watch the OTP rate by country for a week. `model_tier` versus
`tier` in the response tells you how many challenges came from the rule rather than the model —
without that split you cannot tell which one to tune.

Only then enable High (block). Before you do, be sure a blocked user has a route back in that does
not involve your support team guessing.

### Stage 4 — scale, if you need to

Measured on 2 vCPUs, one worker: **~130–155 req/s, p50 6 ms**. A site doing a million logins a day
averages about 12 req/s, so one worker is very likely enough. Do not add machinery you cannot
measure a need for.

When you do need it:

```bash
RISK_REDIS_URL=redis://your-redis:6379/0
uvicorn riskengine.api:create_app --factory --workers 4
```

Redis costs about **1 ms per login** and is what makes several workers correct rather than merely
parallel — without it each worker keeps its own private counters and an IP spread across four
workers looks a quarter as busy as it is.

One measured warning: on a 2-vCPU box, going from 1 worker to 2 made single-request latency **eight
times worse** (6 ms → 48 ms) while peak throughput rose only ~20%. Workers compete for cores. Set
workers to roughly the core count, and re-run `scripts/loadtest.py` after changing it — the number
that matters is p99, because that is what your client timeout has to clear.

---

## What to watch once it is live

Run `scripts/monitor_drift.py` on a schedule; it exits non-zero when any check fires.

| Signal | Healthy | What it means when it is not |
|---|---|---|
| Medium+ flag rate | near 2% | 10× means an attack or stale thresholds |
| High flag rate | near 0.5% | same |
| client `degraded` count | 0 | the engine is failing open and protecting nothing |
| `risk_warming_up` | 0 | stuck in warm-up: enforcing nothing at all |
| score latency p99 | below your client timeout | you are about to start failing open under load |
| feature PSI | below 0.25 | a feature has stopped meaning what it meant |
| `geoip_source` | `maxmind` | `client-supplied` means attackers choose their own country |
| analyst queue depth | draining | nobody is reviewing: no new labels, no learning |

---

## Failure modes, and what they do

| What breaks | What happens | What to do |
|---|---|---|
| Risk service down | `RiskClient` fails open after its timeout; logins proceed unprotected | alert on `degraded`; the breaker stops every login paying the timeout |
| Redis down | scoring fails; client fails open | run Redis with AOF; it holds behavioural state, not the audit trail, so a short loss costs accuracy not records |
| Model file replaced | startup refuses on the digest check | that is the feature working; re-pin deliberately |
| Feature list changed without retraining | startup refuses on the feature contract | retrain; never force it |
| Disk full | state snapshot fails; the writer logs and keeps serving | alert on disk; the audit DB is the thing you cannot lose |
| Clock skew between boxes | in `server` mode each box uses its own clock; windows get noisy | run NTP; this is why `client` mode is not the default |

---

## What you can honestly claim

> Production-grade engineering, validated on a 31-million-login public benchmark: 0.856 ROC-AUC on
> a held-out future period, thresholds calibrated to within 0.02 percentage points of their budget
> on data the model never saw, and a takeover head reported as *not* beating a one-line baseline
> because at a matched false-alarm rate it did not. Proven train/serve parity, server-side geo
> resolution, model integrity pinning, shared state for horizontal scaling, a fail-open client, and
> a documented shadow-mode rollout — because the benchmark data is not fit for production and
> pretending otherwise would be the actual mistake.

Do not shorten that to "production ready" while the model is trained on synthetic data. The longer
claim is stronger, and it survives someone who knows the dataset.