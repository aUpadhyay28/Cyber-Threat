"""Work out where a login came from, server-side.

Why this exists
---------------
`country` and `asn` are two of the strongest signals in the engine: the takeover head is largely
"has this account used this country before", and the home-country policy below it refuses to treat
an unfamiliar country as routine. Until now both arrived in the request body, which means the
caller chose them - and anything an attacker can choose is not a signal, it is an input they
control. A login claiming `country: "IN"` from a hosting provider in another continent would have
scored as a quiet domestic login.

So in production the service resolves them from the socket address itself, with MaxMind's GeoLite2
databases, and ignores whatever the caller said. Where the two disagree, the disagreement is logged:
a backend that keeps sending a country different from the one the IP resolves to is either reading
a spoofable `X-Forwarded-For` or has a bug, and either way you want to know.

Getting the databases
---------------------
GeoLite2 is free but requires an account (MaxMind stopped serving them anonymously in 2019):
  1. sign up at https://www.maxmind.com/en/geolite2/signup
  2. download GeoLite2-Country.mmdb and GeoLite2-ASN.mmdb
  3. point RISK_GEOIP_COUNTRY_DB and RISK_GEOIP_ASN_DB at them
The databases go stale - IP ranges are reassigned constantly - so refresh them monthly. MaxMind
publish updates twice a week and their licence requires you to keep reasonably current.
"""
from __future__ import annotations

import ipaddress
import logging
import os
from pathlib import Path
from typing import Optional, Protocol, Tuple

log = logging.getLogger("riskengine.geoip")

Resolved = Tuple[Optional[str], Optional[int]]      # (ISO-3166-1 alpha-2 country, ASN)


class GeoResolver(Protocol):
    """Anything that can turn an IP into (country, asn)."""

    def resolve(self, ip: str) -> Resolved: ...

    @property
    def name(self) -> str: ...


def is_private(ip: str) -> bool:
    """True for addresses that carry no geographic meaning (RFC1918, loopback, CGNAT, link-local)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (addr.is_private or addr.is_loopback or addr.is_link_local
            or addr.is_multicast or addr.is_reserved or addr.is_unspecified)


class ClientSuppliedResolver:
    """Keep trusting the caller's country/asn. The original behaviour, and not safe for production.

    Kept as an explicit, named choice so that running without geo-IP is a decision somebody made
    rather than something that happened by default.
    """

    name = "client-supplied"

    def resolve(self, ip: str) -> Resolved:
        return (None, None)                           # None means "use what the caller sent"


class StaticResolver:
    """A fixed table, for tests and for pinning behaviour in a demo."""

    name = "static"

    def __init__(self, table: dict[str, Resolved], default: Resolved = (None, None)):
        self.table, self.default = dict(table), default

    def resolve(self, ip: str) -> Resolved:
        return self.table.get(ip, self.default)


class MaxMindResolver:
    """GeoLite2 lookups. Opens the databases once and reuses the readers (they are thread-safe)."""

    name = "maxmind"

    def __init__(self, country_db: str | os.PathLike, asn_db: str | os.PathLike | None = None):
        try:
            import geoip2.database                    # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "geo-IP resolution needs the geoip2 package: pip install geoip2"
            ) from exc
        country_db = Path(country_db)
        if not country_db.exists():
            raise RuntimeError(
                f"GeoLite2 country database not found at {country_db}. Download GeoLite2-Country.mmdb "
                f"from MaxMind (free account) and set RISK_GEOIP_COUNTRY_DB, or set "
                f"RISK_GEOIP_MODE=client to keep trusting the caller's country - which means an "
                f"attacker picks their own country, so do not do that in production.")
        self._country = geoip2.database.Reader(str(country_db))
        self._asn = None
        if asn_db:
            asn_db = Path(asn_db)
            if not asn_db.exists():
                raise RuntimeError(f"GeoLite2 ASN database not found at {asn_db}")
            self._asn = geoip2.database.Reader(str(asn_db))
        else:
            log.warning("no GeoLite2 ASN database configured; ASN will fall back to the caller's value")

    def resolve(self, ip: str) -> Resolved:
        if is_private(ip):
            return (None, None)
        country = asn = None
        try:
            country = self._country.country(ip).country.iso_code
        except Exception:                             # noqa: BLE001 - AddressNotFoundError and friends
            country = None
        if self._asn is not None:
            try:
                asn = self._asn.asn(ip).autonomous_system_number
            except Exception:                         # noqa: BLE001
                asn = None
        return (country, asn)

    def close(self) -> None:
        self._country.close()
        if self._asn is not None:
            self._asn.close()


def build_resolver(mode: str | None = None, country_db: str | None = None,
                   asn_db: str | None = None) -> GeoResolver:
    """Pick a resolver from configuration. 'maxmind' in production, 'client' only knowingly."""
    mode = (mode or os.getenv("RISK_GEOIP_MODE", "client")).lower()
    if mode == "client":
        log.warning("geo-IP mode is 'client': country and ASN come from the caller and can be chosen "
                    "by an attacker. Set RISK_GEOIP_MODE=maxmind with GeoLite2 databases in production.")
        return ClientSuppliedResolver()
    if mode == "maxmind":
        return MaxMindResolver(country_db or os.getenv("RISK_GEOIP_COUNTRY_DB", ""),
                               asn_db or os.getenv("RISK_GEOIP_ASN_DB") or None)
    raise RuntimeError(f"RISK_GEOIP_MODE must be 'maxmind' or 'client', got {mode!r}")


def apply_resolution(resolver: GeoResolver, ip: str, claimed_country, claimed_asn) -> dict:
    """Resolve, and report what happened so the decision is auditable.

    Returns {country, asn, geo_source, geo_mismatch}. A resolver that returns None for a value
    leaves the caller's value in place, which is what makes private IPs and unknown ranges work in
    staging without special-casing them at every call site.
    """
    country, asn = resolver.resolve(ip)
    mismatch = None
    if country is not None and claimed_country and str(claimed_country).upper() != country.upper():
        mismatch = {"claimed": str(claimed_country).upper(), "resolved": country.upper()}
    return {
        "country": country if country is not None else claimed_country,
        "asn": asn if asn is not None else claimed_asn,
        "geo_source": resolver.name if country is not None else "caller",
        "geo_mismatch": mismatch,
    }
