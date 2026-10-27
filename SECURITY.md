# Threat model

This service decides whether a login is allowed, challenged or blocked. Two things can go wrong:
it can be made to **crash**, or it can be made to **return the wrong answer**. The second is worse,
because nothing looks broken. This file lists what the system defends against, what it does not, and
what an attacker who has read this repository would try.

## What the service trusts

| Input | Trusted? | Why |
|---|---|---|
| `user_id`, `success` | Yes | Only your own backend knows whether the password matched. |
| `ip` | Validated, then trusted | Must be the socket address your edge saw, never a client-supplied `X-Forwarded-For` the user can set. |
| `country`, `asn` | Should be derived server-side | Resolve from the IP with MaxMind GeoLite2. A client-supplied country is an attacker-chosen feature value. |
| `browser`, `os`, `device` | Trusted as *strings*, not as facts | They come from the User-Agent, which an attacker controls. They are only used as "has this account used this value before", so lying makes a login look *more* unusual, not less. |
| `timestamp` | **No** (by default) | See below. |

## Defences in place

**Timestamp poisoning (the most serious one).** The 1 h / 24 h network features expire relative to
the newest timestamp the state has seen. A single login dated far in the future therefore evicted
every earlier event from an IP's window — so an IP with 60 recent failed logins could be made to look
brand new, which is exactly the signal the attack-IP head relies on most. Two layers now:

* the service timestamps events with **its own clock** by default (`RISK_TIMESTAMP_SOURCE=server`);
* `FeatureState` refuses any event more than `MAX_FORWARD_JUMP_MS` (7 days) beyond what it has
  already recorded, which is what protects the replay mode offline imports and tests use.

Reproduced and locked down in `tests/test_security.py::test_future_timestamp_cannot_wipe_the_window`.

**Code execution through the state file.** The live state used to be a pickle, and `pickle.load`
executes code embedded in the file. Anything able to write `state.pkl` — a loose volume mount, a
restored backup, a path traversal elsewhere on the host — owned the risk service. State is now
gzipped JSON with an explicit version (`riskengine/persist.py`), and a file that is still a pickle is
refused with a pointer to `scripts/migrate_state_pickle.py` (the one place that imports pickle, and
it demands `--i-trust-this-file`).

**Code execution and silent swaps through the model file.** The model is still a joblib, because
scikit-learn estimators have no portable plain-data form. Instead its SHA-256 is verified before it
is deserialised (`RISK_MODEL_SHA256`, or a `.sha256` file written by `scripts/pin_model.py`). Whoever
can replace that file decides which logins get blocked, so pin it in your release.

**Feature-contract mismatch.** `heads()` locates each feature by its *position*. A model fitted
before a feature was added or reordered would have every column read off by one and would score
confidently on the wrong numbers, with no error. `RiskModel.load` now refuses any model whose
recorded feature list differs from what the code builds.

**Profile poisoning.** A login only joins a user's "normal" profile when it was Low risk and
successful, or the user passed step-up, or an analyst called it benign. Additionally a **failed**
login can no longer be confirmed: replaying confirmations for failed attempts was a way to teach the
system that an attacker's country and device were normal.

**Reversible identifier hashes.** User IDs, IPs, devices and browsers are stored as 64-bit SipHash
values. With pandas' public default key those are deterministic but not secret: anyone holding a
state file or a database dump could hash every plausible email address or account number and match
them back. Set `RISK_HASH_KEY` and the hashes are keyed with a secret derived from it. The key's
fingerprint (never the key) is recorded in the prepared data, the state file and Redis, and a
mismatch is refused, because scoring under the wrong key would make every returning user look new.
`RISK_ENV=production` will not start without it. Tests: `tests/test_hashing.py`.

**Misconfiguration.** `RISK_ENV=production` refuses to start without server timestamps, keyed
hashing, a pinned model and server-side GeoIP, listing every missing item.

**Audit trail.** Every decision records the model version that made it, so "what scored this login?"
remains answerable after a retrain.

**One key opening everything.** There are three keys for three roles: `RISK_API_KEY` (score and
confirm - the only one a web server holds), `RISK_ANALYST_KEY` (review queue and verdicts) and
`RISK_ADMIN_KEY` (erasure, retention, model reload). A leaked login-backend key can no longer read
queued logins, record verdicts that retrain the model, or erase accounts; a valid key used outside
its role gets 403 and a log line. Keys can be read from files (`*_FILE`) so they stay out of process
environments, several keys per role allow rotation without downtime, and production mode requires
all three to be set and different. Tests: `tests/test_hardening.py`.

**Brute force and abuse.** Fixed-window rate limiting per caller and, separately, per source address
on authentication failures (shared across workers through Redis when it is configured). API keys must
be at least 24 characters. Every key is compared in constant time, and all of them are checked on
each request, so timing does not reveal which key or role a guess came close to.

**Reconnaissance.** In production `/docs`, `/redoc` and `/openapi.json` are off and the
unauthenticated `/health` says only `{"status": "ok"}`. The `Server` header is removed. `/metrics`
can require a bearer token (`RISK_METRICS_TOKEN`).

**Oversized requests.** Bodies over 64 KB (`RISK_MAX_BODY_BYTES`) are refused with 413 before any
parsing, and a body without a Content-Length is refused with 411.

**Script injection in the analyst console.** Attacker-controlled strings (user agents, user IDs) are
written with `textContent`, never as markup. Behind that, the console is served with a
Content-Security-Policy that allows only its own inline script, by hash: no external scripts, no
inline event handlers, no requests to other origins, no framing. API responses carry
`default-src 'none'`.

**Personal data in logs.** Logs are copied to weaker places than the database and are not touched
by erasure, so user IDs and IPs never appear in them in clear: log lines carry a short keyed tag
instead (`u:3f9a...`), which still lets you follow one account through the logs.

**Data at rest.** The audit database and the state file are created owner-only (0600) in an
owner-only directory. Redis requires a password, has `FLUSHALL`, `FLUSHDB`, `CONFIG` and `DEBUG`
disabled, and has no published port.

**Container.** Non-root user, read-only root filesystem, all Linux capabilities dropped,
`no-new-privileges`, memory and process limits, service bound to 127.0.0.1 on the host, model
mounted read-only.

**Supply chain.** CI runs `pip-audit` (known CVEs in dependencies), `bandit` (static analysis) and
`detect-secrets` (committed keys) on every push and weekly; Dependabot opens update PRs for Python
packages, GitHub Actions and the base image; the workflow token is read-only.

**Erasure and retention.** `/v1/admin/forget-user` removes an account from the live profiles and
deletes its events; `/v1/admin/purge-raw` strips raw IPs and user agents from old events while
keeping the derived features and verdicts; `/v1/admin/prune-state` drops per-IP/ASN state that has
been idle past the retention period, which also bounds memory growth.

## What is NOT defended, and what an attacker would try next

* **A compromised login backend.** Anything holding `RISK_API_KEY` can assert any `user_id` and
  `success`, and can confirm any Medium event. Role separation limits it to that, but within it the
  key is the trust boundary - this service must not be reachable from the internet.
* **Self-asserted analyst names.** The analyst key proves a request came from the console, not which
  analyst sent it; the `analyst` field is what the person typed. Put SSO in front of `/console` if
  verdicts need to be attributable to individuals.
* **No TLS of its own.** The service speaks plain HTTP and expects to sit on a private network or
  behind a TLS-terminating proxy. If the login backend reaches it over any network you do not control,
  terminate TLS in front of it.
* **Low-and-slow credential stuffing.** One attempt per IP per day produces no window signal at all.
  The 24 h counts are the main detector, so spreading attempts across a large botnet defeats them.
  This is a limitation of the features, not a bug, and is why the takeover head exists.
* **Warm-up as a window.** Below `RISK_WARMUP_MIN_EVENTS` everything is log-only, so a new
  deployment enforces nothing. Run shadow mode deliberately, not by accident.
* **Profile building before an attack.** An attacker with valid credentials can log in normally
  enough times to make their device and country part of the account's normal profile, after which
  deviation scoring sees nothing unusual. Step-up on sensitive actions, not just at login.
* **The per-category tables cannot be enumerated per user.** They are keyed by a non-invertible
  `mix(user, value)`, so erasure needs the values read back from the event log. If you purge raw
  payloads before honouring an erasure request, those rows cannot be found — purge order matters.
* **Per-worker rate limiter.** Behavioural state can be shared through Redis (`RISK_REDIS_URL`), but
  the request rate limiter is still per worker, so with N workers the effective limit is N times the
  configured one. Put a rate limit at your reverse proxy as well.
* **Model inversion.** `explain()` output tells an attacker precisely which signal fired. It is for
  analysts and the audit log; never return it, or the score, to the end user.

## Reporting

This is a student project and not validated on real traffic. If you find a flaw in it, open an issue.
