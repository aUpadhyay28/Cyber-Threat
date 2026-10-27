![tests](https://github.com/aUpadhyay28/Cyber-Threat/actions/workflows/ci.yml/badge.svg)
# Adaptive Login Risk Engine

Scores every login **Low / Medium / High** and learns from analyst-validated feedback. One codebase for
training and for the live site.

```
login ──► features from the user's / IP's history (riskengine/features.py)
              ├─► IP / network head   (small boosted model, attack-IP label)  ─► risk 0-100 ─┐
              └─► Takeover head       (logistic regression on "does this login deviate from THIS user's    ─► risk 0-100 ─┤
                  own history", needs >=3 trusted prior logins)                                                      │
                                                                  final risk = max(heads) ◄──────────────────────────┘
Low -> allow   Medium -> step-up (OTP)   High -> block + analyst review      (only while NOT warming_up, see below)
```
Each head's score is the **percentile among normal logins** (reference saved at training time), so one pair of
thresholds works for both: Medium = top 2% of normal logins per head (risk >= 98), High = top 0.5% (>= 99.5).

## 1. Install and test
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install fakeredis                                     # optional: lets the Redis tests run without a server
pytest -q                                                 # 186 tests, ~2 min
```

## 2. Train
```bash
python scripts/prepare_data.py --csv rba-dataset.csv --out data/prepared      # once; streams the 8 GB CSV
python scripts/train.py --data data/prepared --out models                     # full data
python scripts/pin_model.py --model models/latest.joblib                      # pin what you will deploy
```
`--max-rows N` limits both steps for a trial. `--bulk-threshold` (default 20,000) sets what counts as a
pseudo-account. Read `models/latest_report.json`: the attack-IP head is tested on a **future period**; the
takeover head is **cross-validated by user** (takeover logins are too few and too early for a time split) and is
always compared with the one-line rule `new_country`.
A model records the feature list it was fitted on, and `RiskModel.load` refuses one that does not match the
current code, because features are located by position and a mismatch would silently score the wrong columns.

### Developing on the 11 MB sample instead of the 31 M rows
```bash
python scripts/export_sample.py --data data/prepared --out data/sample.parquet   # once
python scripts/train_from_sample.py --sample data/sample.parquet --out models_sample
```
The sample is **stratified**: attacks are over-represented on purpose and each row carries a `sample_weight`
saying how many real logins it stands for (they sum back to 31,269,264). Everything in `train_from_sample.py`
passes those weights into every fit, every metric and both calibration reference distributions, so the thresholds
mean the same thing they would on the full data. Skipping the weights is not a rounding error — on this dataset
it reports **PR-AUC 0.77 instead of 0.34** and a test attack rate of 40% instead of 10.7%, because the sample's
base rate is three times the population's. `--unweighted` exists only to show that gap.

What the sample still cannot do: it holds features, not raw columns, so feature definitions cannot be changed or
re-derived from it and `tests/test_parity.py` needs the prepared data. Develop on the sample; retrain on the full
file before shipping.

## 3. Run the scoring service
```bash
export RISK_API_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"   # >= 24 chars, enforced
export RISK_ANALYST_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"  # review console
export RISK_ADMIN_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"    # erasure, model reload
export RISK_HASH_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"  # keyed hashing of user IDs / IPs
export RISK_MODEL_SHA256=$(cat models/latest.joblib.sha256)                            # refuse a swapped model
uvicorn riskengine.api:create_app --factory --port 8000       # then open http://127.0.0.1:8000/docs
```
`POST /v1/score` returns `risk_score`, `tier`, `action`, `ip_risk`, `takeover_risk` (null for accounts with too
little history), `model_tier` (the model's own verdict, before any policy rule), `reasons`, `event_id`,
`warming_up`, `model_version`.

| Endpoint | Key | What it does |
|---|---|---|
| `POST /v1/score` | `RISK_API_KEY` | score one login |
| `POST /v1/confirm/{id}` | `RISK_API_KEY` | user passed the OTP; the login joins their normal profile (a *failed* login cannot be confirmed) |
| `GET /v1/queue`, `GET /v1/event/{id}`, `POST /v1/verdict`, `GET /v1/stats` | `RISK_ANALYST_KEY` (or admin) | review queue and labelling |
| `POST /v1/admin/reload-model`, `forget-user`, `purge-raw`, `prune-state` | `RISK_ADMIN_KEY` | model swap, erasure, retention |
| `GET /console` | analyst key, typed into the page | the review console |
| `GET /health`, `GET /ready`, `GET /metrics` | none (`/metrics` optionally `RISK_METRICS_TOKEN`) | status, readiness probe, Prometheus metrics |

A key used outside its role gets **403** and is logged. In development, an unset analyst or admin key falls back to
`RISK_API_KEY` so the demo runs with one secret; production mode refuses that. Any key can instead be read from a
file (`RISK_API_KEY_FILE=/run/secrets/...`), and each variable accepts several comma-separated keys for rotation.

### Production mode

```bash
export RISK_ENV=production
```

With `RISK_ENV=production` the service **refuses to start** unless every setting in `PRODUCTION.md` stage 0 that can
be checked from inside the process is on: server timestamps, a `RISK_HASH_KEY`, a pinned model,
`RISK_GEOIP_MODE=maxmind`, three different role keys and a password on Redis. It lists every missing item at once.
Production mode also turns off `/docs` and `/openapi.json` and reduces the unauthenticated `/health` to
`{"status": "ok"}` (the full view needs an analyst or admin key). In development none of this is enforced, so the
demo and the tests still run on a laptop. See `SECURITY.md` for the full threat model.

### Running it for real: what else is in the box

* **`python scripts/generate_secrets.py`** – writes a `.env` with fresh, distinct random keys (owner-only file).
* **`docker compose up`** – the service plus a password-protected Redis, bound to 127.0.0.1 only, non-root,
  read-only filesystem, no Linux capabilities, model mounted read-only, shadow mode by default. Copy `.env.example`
  to `.env` and fill it in first.
* **Shared state** – set `RISK_REDIS_URL` and several workers share one set of counters (`riskengine/redis_state.py`).
  Without it, each worker would see only its own share of an attacking IP's traffic.
* **Server-side geolocation** – `RISK_GEOIP_MODE=maxmind` with the free GeoLite2 Country and ASN databases
  (`riskengine/geoip.py`). Without them the caller supplies country and ASN, which means an attacker does.
* **Home-country rule** – `RISK_HOME_COUNTRIES=IN` raises a login from a country this account has never used to at
  least Medium (step-up), never to a block (`riskengine/policy.py`).
* **Client library** – `riskengine/client.py`: timeout, fail-open by default, circuit breaker and a `degraded` counter.
  Use it from your login backend rather than calling the API by hand.
* **Analyst console** – `/console` in a browser: the review queue, event detail and verdict buttons.
* **Assistant tools** – `python -m riskengine.mcp_server` exposes the queue to an LLM assistant with defences against
  tool-output poisoning; the assistant can propose a verdict but never record one.
* **Load test** – `scripts/loadtest.py` measures throughput and p99 on your hardware. Set the client timeout above p99.

### Timestamps are not taken from the caller
By default (`RISK_TIMESTAMP_SOURCE=server`) the service timestamps each login with its own clock. The 1 h / 24 h
network features expire relative to the newest timestamp seen, so one future-dated login would clear an attacking
IP's recent-failure history and let it score as a quiet network. `RISK_TIMESTAMP_SOURCE=client` trusts the
caller's time for replay, backfill and the tests, and then `FeatureState` refuses any event more than 7 days
beyond what it has already recorded. See `SECURITY.md` and
`tests/test_security.py::test_future_timestamp_cannot_wipe_the_window`.

### Live state is not a pickle
State is gzipped JSON (`RISK_STATE_PATH=data/state.json.gz`). Loading a pickle executes code inside it, so an
old `state.pkl` is refused; convert it once with `scripts/migrate_state_pickle.py --i-trust-this-file`.

### Monitoring
```bash
python scripts/monitor_drift.py --db data/risk.db --model models/latest.joblib     # exits 1 on any alert
```
Checks the realised Medium+/High flag rates against their budgets, that the model on disk is the one making
decisions, and per-feature population-stability (PSI) between the recent window and the previous one. Run it on
a schedule: this project's own finding was that features drift, so something has to watch for it.

### Retention
`/v1/admin/purge-raw` strips raw IPs and user agents from old events, keeping features and verdicts.
`/v1/admin/prune-state` drops per-IP/ASN state idle past the retention period, which also bounds memory growth.
Pruning an idle IP also drops its distinct-account set, so `ip_distinct_users_log` and `asn_distinct_users_log`
restart for an IP that has been silent that long — a deliberate consequence of deleting old data. Both features
are excluded from the IP head (`IP_HEAD_EXCLUDE`) and only feed `explain()`, so scoring is unaffected.
`/v1/admin/forget-user` erases one account: profile, per-category entries and logged events. Honour erasure
*before* purging raw payloads — the per-category tables are keyed by a non-invertible `mix(user, value)`, so the
values have to be read back from the event log.

### Warm-up (important)
The IP/network head uses how much recent traffic each IP and network has produced. A new or small site has
almost none, so a model trained on a 31M-login service would flag nearly everything. Until the system has seen
`RISK_WARMUP_MIN_EVENTS` logins (default 10,000) the API reports `warming_up: true`, forces `action: allow`
(log-only) and gives the would-be action in `suggested_action`. Run in shadow mode, then retrain on your own data.

## 4. Learning loop and historic data
`scripts/import_history.py` loads your site's old logins into the live profiles. Analysts review `/v1/queue` and
send `/v1/verdict`; `python scripts/train.py --feedback-db data/risk.db` retrains with those rows (weight x20).
Only `benign` and `confirmed_malicious` train the model. A login only joins a user's "normal" profile when it was
Low risk and successful, or the user (OTP) or an analyst confirmed it, so an attacker cannot teach the system.

## What we learned from the RBA dataset (put these in your report)
* The dataset is **synthetic** and its authors warn against production use (CC BY 4.0: cite Wiefling et al. 2022).
  Use it to build and benchmark the pipeline, then train the deployed model on your own site's logins.
* One user ID owns 45% of all rows and every login failed: a pseudo-account, excluded from training.
* **Capacity hurts**: big models memorise the training period and get *worse* on the future (test AUC 0.86 for a
  20-tree model, 0.47 for a 1000-tree one). All-time running counters drift and were removed from the IP head.
* The IP head also ignores `log_rtt` (randomly assigned in this data; its presence was used as a signal) and the
  1-hour count (no signal here) - see `IP_HEAD_EXCLUDE` in `config.py`. Dropping them cost nothing in accuracy.
* Almost all attack-IP skill comes from **24-hour window counts** per IP/network (timestamps are partly
  randomised in this dataset, so treat that gain with care). The 1-hour window added nothing here.
* Takeover (141 cases, 51 on accounts with history): "login from a country this user never used" catches ~61%
  at ~0.4% false alarms; the fitted model is not clearly better (CIs overlap). 64% of takeover logins hit
  accounts with <3 prior logins and cannot be caught by profile deviation.
* Network-rarity features separate takeovers almost perfectly (AUC ~0.97) because those logins come from
  networks with zero recent traffic and near-unique browser strings. That looks like a **generator artifact**;
  it is left out of the default model and should be tested on real data before use.

* The takeover logins are not only few, they are **early**: with the 60/80 time split, 130 of the 141 fall in
  the training period, 11 in validation and **none in the test period**. That is why this head is cross-validated
  by user rather than time-split — the time split has no positives to score.

## Known limits

* **Not validated on real traffic.** The shipped model is fitted on synthetic benchmark data. A real deployment must
  run in shadow mode on its own logins and retrain before enforcing anything – see `PRODUCTION.md`.
* **No labels on a real site.** Real logins do not arrive marked "attack" or "takeover"; the analyst queue is how
  labels get made, so the system only improves if someone reviews it.
* **Low-and-slow attacks.** One attempt per IP per day across a large botnet produces no window signal. That is a
  limit of the features, not a bug.
* **The API key is the whole trust boundary.** Anything holding it can assert any user and outcome. Keep the service
  off the internet (the compose file binds it to 127.0.0.1).
* **Profile building.** An attacker with valid credentials who logs in quietly for a while becomes "normal" for that
  account. Step-up on sensitive actions, not only at login.
* `SECURITY.md` records the full threat model: what is defended, what is not, and what an attacker would try next.

Fixed since earlier versions: the scoring lock no longer covers disk writes (they run on a background thread);
several workers can share state through Redis; identifiers can be hashed with a secret key (`RISK_HASH_KEY`).
