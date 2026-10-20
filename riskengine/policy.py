"""Deterministic rules applied on top of the model's score.

Why a separate layer, rather than a feature
-------------------------------------------
"A login from outside India should never be waved through" is a business decision, not something to
be learned. Putting it in the model would make it negotiable - the model could decide a foreign
login looks fine - and it would need a retrain to change. As a policy layer it is explicit,
auditable, testable, adjustable without touching the model, and it keeps working while the model is
still warming up on a new deployment.

The model's own verdict is preserved in the response as `model_tier`, so you can always separate
"the model flagged this" from "a rule flagged this". Without that separation you cannot measure how
many step-ups the rule is costing you, and the rule becomes impossible to tune.

What the home-country rule does, and does not do
------------------------------------------------
An IP tells you where a connection came from. It does not tell you anyone's nationality, and nothing
in this system could. So this is a rule about *connection country*, and it is deliberately narrow:

  * it raises risk to at least Medium (step-up / OTP), it does not block - a blocked traveller is
    locked out of their own account with no way back in
  * it only fires for a country this account has NOT used before, so a user who regularly logs in
    from abroad stops being challenged once that country is part of their own normal pattern
  * it never lowers a score; the model can still escalate above it

The cost is real and you should measure it: every genuine user whose first trip abroad coincides
with a login gets an OTP. That is the intended trade, but watch the step-up rate by country after
you turn it on.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from .config import TIER_ACTIONS, TIERS

log = logging.getLogger("riskengine.policy")

_TIER_RANK = {t: i for i, t in enumerate(TIERS)}            # Low < Medium < High


def _parse_countries(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    return frozenset(c.strip().upper() for c in raw.split(",") if c.strip())


@dataclass
class HomeCountryPolicy:
    """Raise unfamiliar foreign logins to at least `min_tier`.

    home_countries: ISO-3166-1 alpha-2 codes that count as domestic. EMPTY DISABLES THE RULE, so
        the default behaviour of the library is unchanged and turning this on is a deployment
        decision (RISK_HOME_COUNTRIES=IN).
    only_unfamiliar: skip the rule when this account has used the country before.
    apply_while_warming: the rule does not depend on traffic volume - unlike the IP head, which is
        why warm-up exists - so it is safe to enforce from the first login. This is the one control
        that works on day one of a deployment.
    """

    home_countries: frozenset[str] = field(default_factory=frozenset)
    min_tier: str = "Medium"
    only_unfamiliar: bool = True
    apply_while_warming: bool = True

    @classmethod
    def from_env(cls) -> "HomeCountryPolicy":
        countries = _parse_countries(os.getenv("RISK_HOME_COUNTRIES"))
        min_tier = os.getenv("RISK_FOREIGN_MIN_TIER", "Medium").title()
        if min_tier not in _TIER_RANK:
            raise RuntimeError(f"RISK_FOREIGN_MIN_TIER must be one of {TIERS}, got {min_tier!r}")
        pol = cls(home_countries=countries, min_tier=min_tier,
                  only_unfamiliar=os.getenv("RISK_FOREIGN_ONLY_UNFAMILIAR", "1") not in ("0", "false", "False"),
                  apply_while_warming=os.getenv("RISK_FOREIGN_IN_WARMUP", "1") not in ("0", "false", "False"))
        if pol.enabled:
            log.info("home-country policy active: home=%s, unfamiliar foreign logins raised to %s "
                     "(only_unfamiliar=%s, during_warmup=%s)",
                     sorted(pol.home_countries), pol.min_tier, pol.only_unfamiliar, pol.apply_while_warming)
        return pol

    @property
    def enabled(self) -> bool:
        return bool(self.home_countries)

    def apply(self, result: dict, country, feats: dict, warming: bool) -> dict:
        """Return a new result dict, with `model_tier` and `policy` recorded.

        `result` is what RiskModel.score_features produced. `country` must be the SERVER-RESOLVED
        country - passing the caller's value makes the rule trivially bypassable.
        """
        out = {**result, "model_tier": result["tier"], "policy": []}
        if not self.enabled or (warming and not self.apply_while_warming):
            return out
        if country is None or str(country).strip() == "" or str(country).upper() == "NA":
            # Unknown origin: no claim either way. Private ranges and unmapped IPs land here, and
            # escalating every one of them would make staging unusable.
            return out
        if str(country).upper() in self.home_countries:
            return out
        if self.only_unfamiliar and not feats.get("new_country", 1.0):
            out["policy"].append("foreign_country_known_to_user_no_action")
            return out
        if _TIER_RANK[result["tier"]] >= _TIER_RANK[self.min_tier]:
            out["policy"].append("foreign_country_already_above_floor")
            return out

        out["tier"] = self.min_tier
        out["action"] = TIER_ACTIONS[self.min_tier]
        out["policy"].append("foreign_country_step_up")
        return out
