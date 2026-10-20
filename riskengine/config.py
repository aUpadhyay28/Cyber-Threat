"""Central configuration. Edit RAW_COLUMNS if your copy of the dataset names columns differently."""
# logical name -> column name in the RBA dataset CSV
RAW_COLUMNS = {
    "ts": "Login Timestamp",
    "user": "User ID",
    "ip": "IP Address",
    "country": "Country",
    "asn": "ASN",
    "browser": "Browser Name and Version",
    "os": "OS Name and Version",
    "device": "Device Type",
    "rtt": "Round-Trip Time [ms]",
    "success": "Login Successful",
    "attack_ip": "Is Attack IP",
    "ato": "Is Account Takeover",
}
CATS = ["ip", "country", "asn", "browser", "os", "device"]   # "have we seen this before?" signals

TIERS = ("Low", "Medium", "High")
TIER_ACTIONS = {"Low": "allow", "Medium": "step_up_auth", "High": "block_and_review"}
VERDICTS = ("benign", "suspicious", "confirmed_malicious")
LEARNABLE_VERDICTS = {"benign": 0, "confirmed_malicious": 1}   # 'suspicious' never trains the model

# --- two-head design (see README) -------------------------------------------------------------
# Each head turns its raw score into 0-100 = percentile among NORMAL logins, so one pair of thresholds
# works for both. Budget = share of normal logins that head may flag.
HEAD_MEDIUM_BUDGET = 0.02      # Medium: top 2% of normal logins per head  (risk >= 98)
HEAD_HIGH_BUDGET = 0.005       # High:   top 0.5%                          (risk >= 99.5)
WARM_MIN_HISTORY = 3           # takeover head needs >= 3 prior trusted logins by the user
# Running all-time counters drift (their meaning changes as data accumulates) and made big models
# memorise the past; the attack-IP head therefore does not use them.
ALLTIME_FEATURES = ["ip_distinct_users_log", "asn_distinct_users_log", "history_count_log",
                    "user_fail_prior_log", "browser_global_freq", "os_global_freq"]
# Excluded from the IP/network head (still computed, so you can switch them back on for your own data):
#  * log_rtt     - round-trip time is randomly assigned in the RBA data and mostly missing; "RTT present" was
#                  being used as a signal, which is a dataset artifact
#  * ip_n_1h_log - 1-hour count: no signal on this data (ablation) and the model fit it backwards
IP_HEAD_EXCLUDE = ALLTIME_FEATURES + ["log_rtt", "ip_n_1h_log"]
# Takeover head: deviation of THIS login from THIS user's own history ('success' is excluded: a takeover is
# a successful login by definition, so it would be a label artifact).
DEV_FEATURES = ["new_country", "new_asn", "new_ip", "new_device", "new_browser", "new_os",
                "freq_country", "freq_asn", "freq_ip", "freq_device",
                "hour_dev", "hour_conc", "since_last_log", "history_count_log", "user_fail_prior_log"]
# Accounts with an absurd number of logins are pseudo-accounts (in the RBA data one ID owns 45% of rows,
# all failed). They are excluded from training/calibration. Real sites normally have none.
BULK_USER_THRESHOLD = 20_000
FEEDBACK_WEIGHT = 20.0         # an analyst-validated row counts this many times in retraining

# --- security -------------------------------------------------------------------------------------
# A login's timestamp drives the 1 h / 24 h network-window features. A caller-supplied timestamp far in
# the future would evict every earlier event from those windows, so a busy attacking IP could be made to
# look idle (see tests/test_security.py::test_future_timestamp_cannot_wipe_the_window). FeatureState
# therefore refuses an event more than this far beyond the latest timestamp it has already seen.
# None = no limit, and is ONLY correct for trusted offline replay (scripts/import_history.py).
MAX_FORWARD_JUMP_MS = 7 * 86_400_000
# How the live service decides a login's timestamp:
#   "server" (default, secure) - the service's own clock; the client's value is recorded but not used
#   "client"                   - trust the caller's timestamp (replay / backfill / tests only)
TIMESTAMP_SOURCE = "server"
MAX_CLOCK_SKEW_MS = 300_000            # in "client" mode: how far ahead of the server clock is tolerated
# Retention. Dropping an idle IP/ASN also drops the distinct-account sets attached to it, so those two
# explanation features restart for an IP that has been silent this long. That is the intended
# consequence of deleting old data, not an accident - see README "Retention".
STATE_RETENTION_MS = 90 * 86_400_000
RAW_EVENT_RETENTION_DAYS = 90          # raw IPs / user agents are deleted from the event log after this
