# Adaptive Login Risk Engine

Scores every login **Low / Medium / High** from the user's history, trains on the RBA dataset,
and learns from analyst-validated feedback. One codebase for training and for the live site.

```
rba-dataset.csv ──prepare_data──► parquet ──train──► models/latest.joblib ──┐
                                                                            ▼
your site ──POST /v1/score──► features from live history ──► risk 0-100 ► Low / Medium / High
   ▲                              (riskengine/features.py)                    │ allow / OTP / block+review
   │ OTP passed → /v1/confirm                                                 ▼
   └──────────────────────────  analyst /v1/verdict ──► SQLite ──► scripts/train.py --feedback-db
```

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                                 # 14 tests, ~1 min
```

## 2. Smoke-test on mock data (2 minutes)
```bash
python scripts/make_mock_data.py --out data/mock.csv --n 200000
python scripts/prepare_data.py   --csv data/mock.csv --out data/prepared_mock
python scripts/train.py          --data data/prepared_mock --out models_mock
```
Mock attackers are deliberately obvious, so its scores are near-perfect. They prove the pipeline
works, **not** how good the model is.

## 3. Train on the FULL RBA dataset
```bash
python scripts/prepare_data.py --csv rba-dataset.csv --inspect          # check column names first
python scripts/prepare_data.py --csv rba-dataset.csv --out data/prepared # streams in 1M-row chunks
python scripts/train.py        --data data/prepared --out models
```
- If columns differ, edit `RAW_COLUMNS` in `riskengine/config.py` (the script tells you what's missing).
- Use `--max-rows 3000000` on both scripts for a trial run on a smaller machine.
- Full data (~31M rows) needs roughly **16 GB RAM**; prepared parquet is a few GB. Training takes minutes to
  tens of minutes depending on the machine.
- Output: `models/latest.joblib` and `models/latest_report.json`. **Read the report**: it compares a simple
  heuristic baseline vs booster vs novelty detector vs the blend on a held-out *future* period, and shows what
  share of legitimate logins each tier would challenge.

## 4. Run the scoring service
```bash
export RISK_API_KEY='a-long-random-secret'
uvicorn riskengine.api:create_app --factory --port 8000
```
Your web backend calls it after checking the password:
```bash
curl -X POST localhost:8000/v1/score -H "X-API-Key: $RISK_API_KEY" -H 'Content-Type: application/json' -d '{
  "user_id":"42","timestamp":"2026-10-11T09:30:00Z","ip":"203.0.113.7","country":"IN","asn":9829,
  "browser":"Chrome 120","os":"Windows 11","device":"desktop","rtt_ms":80,"success":true}'
```
Returns `risk_score`, `tier`, `action` (`allow` / `step_up_auth` / `block_and_review`), `reasons`, `event_id`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/score` | score a login and record it |
| `POST /v1/confirm/{event_id}` | user passed OTP → login joins their normal profile (not allowed for High) |
| `GET /v1/queue` | unreviewed Medium/High events for analysts, riskiest first |
| `POST /v1/verdict` | analyst: `benign` / `suspicious` / `confirmed_malicious` |
| `POST /v1/admin/reload-model` | load a freshly trained model without restart |
| `GET /health` | status + model version |

## 5. Historic data and the learning loop
- **Your own old logs:** `python scripts/import_history.py --csv old_logins.csv` builds each user's normal profile
  (attack-flagged rows are excluded from "normal").
- **Learn:** analysts review `/v1/queue` → `/v1/verdict`. Then
  `python scripts/train.py --data data/prepared --feedback-db data/risk.db` retrains with those rows
  (weight ×20). Only `benign` and `confirmed_malicious` train the model; `suspicious` and unreviewed never do.
  The live model never changes by itself, so an attacker can't poison it silently.

## Design decisions (what changed from the demo, and why)
| Demo weakness | Fix here |
|---|---|
| Score normalised per batch (a lone login scored 0) | Fixed reference distributions saved at training time; tested |
| Baselines built from ground-truth labels | Features use only a user's/IP's **past**; leakage test |
| Evaluated on the data it learned from | Time-based train / validation / test split |
| Linear hour distance (23:00 ≠ 01:00) | Circular hour deviation + how regular the user is |
| Hand-set risk weights | Learned booster + novelty detector, tiers calibrated to a target false-challenge rate |
| Training code ≠ live code | One feature definition, two implementations, parity-tested |
| Attacker could teach "normal" | Only clean, successful Low-risk (or user/analyst-confirmed) logins update a profile |

## Known limits (be honest about these in your report)
- Not run on the real dataset by the author of this code: verify the report numbers yourself.
- RBA's "Is Attack IP" label partly encodes IP reputation; account-takeover labels are extremely rare, so
  evaluate on both and report the positive counts.
- The dataset is synthesised from real traffic; your site's traffic will differ → run in **shadow mode** first
  (log scores, don't enforce) and recalibrate tiers on your own data.
- Live state is a pickle file + SQLite: fine for one server. For several servers move state to Redis/Postgres.
- Country/ASN come from the caller; derive them server-side (MaxMind GeoLite2) before production.
- IPs, devices and user IDs are personal data: add consent, a retention period and deletion on request.
