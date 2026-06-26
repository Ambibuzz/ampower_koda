# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Live AI model pricing fetcher with in-process caching and a static fallback.

import json
import time
import urllib.request
from decimal import Decimal, ROUND_HALF_UP

import frappe

# Cache for remotely-fetched pricing data: {"data": {...}, "fetched_at": <epoch>}
_PRICING_CACHE = {"data": None, "fetched_at": 0}
_PRICING_CACHE_TTL = 6 * 60 * 60  # refresh every 6 hours
_PRICING_SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

# Static fallback rates (USD per 1,000,000 tokens, blended 3:1 input/output).
# Used only when the remote source is unreachable or doesn't list a model.
STATIC_RATES = {
    # OpenAI
    "gpt-4o-mini": 0.25,
    "gpt-4o": 5.00,
    "gpt-5-mini": 0.25,
    "o3-mini": 2.00,
    # Gemini
    "gemini-2.0-flash": 0.15,
    "gemini-2.5-pro": 2.50,
    # Claude
    "claude-sonnet-4-20250514": 6.00,
    "claude-3-5-sonnet-20241022": 6.00,
    "claude-3-5-haiku-20241022": 1.00,
}

DEFAULT_FALLBACK_MODEL = "gpt-4o-mini"


def _fetch_remote_pricing() -> dict:
    """Fetch current per-model pricing from LiteLLM's maintained pricing JSON.
    Returns {} on any failure so callers can fall back to static rates.
    Cached in-process for _PRICING_CACHE_TTL seconds to avoid hitting the
    network on every cost calculation.
    """
    now = time.time()
    if _PRICING_CACHE["data"] is not None and (now - _PRICING_CACHE["fetched_at"]) < _PRICING_CACHE_TTL:
        return _PRICING_CACHE["data"]

    try:
        req = urllib.request.Request(_PRICING_SOURCE_URL, headers={"User-Agent": "ampower-koda"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        frappe.log_error(
            f"Could not fetch remote pricing, will use static fallback: {e}",
            "Agent Pricing Remote Fetch",
        )
        # Keep serving a stale cache rather than nothing, if we have one
        return _PRICING_CACHE["data"] or {}

    _PRICING_CACHE["data"] = raw
    _PRICING_CACHE["fetched_at"] = now
    return raw


def _blended_rate_from_remote(normalized_model: str) -> float | None:
    """Look up a model in the remote pricing data and compute a blended
    3:1 input/output rate per 1,000,000 tokens. Returns None if not found
    or if the entry is missing usable price fields.
    """
    pricing = _fetch_remote_pricing()
    if not pricing:
        return None

    entry = pricing.get(normalized_model)
    if entry is None:
        for key, val in pricing.items():
            if normalized_model.startswith(key.lower()):
                entry = val
                break
    if not entry:
        return None

    input_cost = entry.get("input_cost_per_token")
    output_cost = entry.get("output_cost_per_token")
    if input_cost is None or output_cost is None:
        return None

    input_rate = input_cost * 1_000_000
    output_rate = output_cost * 1_000_000
    return float((input_rate * 3 + output_rate) / 4)


def get_blended_rate(provider: str, model: str) -> tuple[float, str]:
    """Resolve a blended USD-per-1M-token rate for a given provider/model.
    Tries remote pricing first, then static exact match, then static
    prefix match, then a hard default. Returns (rate, source) where source
    is one of: 'remote', 'static-exact', 'static-prefix', 'default-fallback'.
    """
    normalized_model = (model or "").strip().lower()

    rate = _blended_rate_from_remote(normalized_model)
    if rate is not None:
        return rate, "remote"

    rate = STATIC_RATES.get(normalized_model)
    if rate is not None:
        return rate, "static-exact"

    for key, key_rate in STATIC_RATES.items():
        if normalized_model.startswith(key.lower()):
            return key_rate, "static-prefix"

    rate = STATIC_RATES[DEFAULT_FALLBACK_MODEL]
    frappe.log_error(
        f"Unknown model '{model}' for provider '{provider}', "
        f"falling back to default rate ${rate}/1M tokens",
        "Agent Cost Estimate Fallback",
    )
    return rate, "default-fallback"


def calculate_cost_estimate(provider: str, model: str, tokens: int) -> float:
    """Estimate USD cost based on total tokens used and a per-model blended rate
    (assumes roughly 3:1 input/output token ratio at typical pricing tiers).
    """
    if not tokens:
        return 0.0

    rate, _source = get_blended_rate(provider, model)
    cost = (Decimal(tokens) / Decimal(1_000_000)) * Decimal(str(rate))
    return float(cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))