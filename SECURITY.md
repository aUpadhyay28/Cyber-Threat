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

**Audit trail.** Every decision records the model version that made it, so "what scored this login?"
remains answerable after a retrain.

**Brute force and abuse.** Fixed-window rate limiting per caller and, separately, per source address
on authentication failures. API keys must be at least 24 characters. Constant-time key comparison.

**Erasure and retention.** `/v1/admin/forget-user` removes an account from the live profiles and
deletes its events; `/v1/admin/purge-raw` strips raw IPs and user agents from old events while
keeping the derived features and verdicts; `/v1/admin/prune-state` drops per-IP/ASN state that has
been idle past the retention period, which also bounds memory growth.

## What is NOT defended, and what an attacker would try next

* **A compromised caller.** Anything holding the API key can assert any `user_id` and `success`, and
  can confirm any Medium event. The key is the whole trust boundary — this service must not be
  reachable from the internet.
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
* **Unsalted hashing of identifiers.** `pd.util.hash_array` is deterministic and unkeyed. That is
  fine for the public RBA dataset, but on real data keyed by email or sequential account IDs the
  hashes are reversible by enumeration. Use a keyed HMAC before exporting any sample of real logins.
* **Single-process state.** The rate limiter and the in-memory state are per worker. With several
  workers you need shared state (Redis) for both to mean what they say.
* **Model inversion.** `explain()` output tells an attacker precisely which signal fired. It is for
  analysts and the audit log; never return it, or the score, to the end user.

## Reporting

This is a student project and not validated on real traffic. If you find a flaw in it, open an issue.
