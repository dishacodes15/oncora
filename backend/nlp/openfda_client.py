"""
openfda_client.py
Thin wrapper around the OpenFDA REST API.

Centralizes: api_key injection, retry/backoff on 429 and 5xx, and
timeout handling. Every other module talks to OpenFDA only through
this file — nothing else should call `requests.get` on api.fda.gov
directly. That keeps rate-limit and error handling in one place.
"""

import os
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger("oncora.nlp.openfda")

BASE_URL = "https://api.fda.gov"
DEFAULT_TIMEOUT = 10          # seconds — OpenFDA can be slow; don't hang the request thread forever
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1.5    # exponential backoff: 1.5s, 3s, 6s


class OpenFDAError(Exception):
    """Raised when OpenFDA can't be reached or returns something unusable
    after retries. Callers should catch this and degrade gracefully
    (e.g. skip the interaction lookup) rather than crash the request."""


def _get_api_key() -> Optional[str]:
    key = os.environ.get("OPENFDA_API_KEY")
    if not key:
        # Not fatal — OpenFDA works without a key, just at a much lower
        # rate limit (40/min, 1000/day vs 240/min, 120,000/day with a key).
        logger.warning(
            "OPENFDA_API_KEY not set — using unauthenticated OpenFDA rate limits."
        )
    return key


def _request(path: str, params: dict) -> dict:
    """
    GET against OpenFDA with retry/backoff.
    Returns the parsed JSON body. Raises OpenFDAError on unrecoverable
    failure (including a clean "no results" 404, which callers should
    treat as an empty result, not an exception — see get_label()).
    """
    key = _get_api_key()
    if key:
        params = {**params, "api_key": key}

    url = f"{BASE_URL}{path}"
    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        except (requests.RequestException, OSError) as exc:
            # OSError catches raw socket/SSL-level drops (connection reset,
            # SSL EOF, etc.) that occasionally reach us unwrapped instead of
            # as a requests.RequestException — seen in practice on flaky
            # networks / when something (AV, proxy) interferes mid-response.
            last_exc = exc
            logger.warning("OpenFDA request failed (attempt %d/%d): %s",
                            attempt, MAX_RETRIES, exc)
            time.sleep(BACKOFF_BASE_SECONDS * attempt)
            continue

        if resp.status_code == 404:
            # OpenFDA returns 404 for "search matched nothing" — this is a
            # normal, expected outcome (e.g. drug has no label on file),
            # not an error. Return an empty result shape instead of raising.
            return {"results": []}

        if resp.status_code == 429:
            # Rate limited. Respect Retry-After if present, else backoff.
            wait = float(resp.headers.get("Retry-After", BACKOFF_BASE_SECONDS * attempt))
            logger.warning("OpenFDA rate limit hit, backing off %.1fs", wait)
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            logger.warning("OpenFDA server error %s (attempt %d/%d)",
                            resp.status_code, attempt, MAX_RETRIES)
            time.sleep(BACKOFF_BASE_SECONDS * attempt)
            continue

        if not resp.ok:
            # 4xx other than 404/429 (e.g. malformed query) — don't retry,
            # this won't succeed on a second attempt.
            raise OpenFDAError(f"OpenFDA {resp.status_code}: {resp.text[:200]}")

        return resp.json()

    raise OpenFDAError(f"OpenFDA request failed after {MAX_RETRIES} attempts: {last_exc}")


def search_ndc(limit: int = 1000, skip: int = 0, search: Optional[str] = None) -> dict:
    """Page through /drug/ndc.json. Used only by the offline vocab builder.

    `search` (optional): an OpenFDA search expression, e.g. 'generic_name:a*'.
    Needed because OpenFDA hard-caps `skip` at 25000 regardless of total
    result count — the vocab builder partitions the full directory into
    per-letter slices via this parameter so each slice's own result count
    stays under that ceiling. See vocab_builder.py.
    """
    params = {"limit": limit, "skip": skip}
    if search:
        params["search"] = search
    return _request("/drug/ndc.json", params)


def get_label_by_generic_name(generic_name: str, limit: int = 1) -> dict:
    """Look up a structured product label by generic name (exact match on
    the openfda.generic_name facet). Returns {"results": []} on no match —
    callers must handle that, not every drug has a full label on file."""
    query = f'openfda.generic_name:"{generic_name}"'
    return _request("/drug/label.json", {"search": query, "limit": limit})


def get_label_by_brand_name(brand_name: str, limit: int = 1) -> dict:
    """Same as above but matched on brand name. Used as a fallback when
    the generic-name lookup comes back empty — OpenFDA's generic_name
    facet doesn't cover every product, but brand_name sometimes does."""
    query = f'openfda.brand_name:"{brand_name}"'
    return _request("/drug/label.json", {"search": query, "limit": limit})
