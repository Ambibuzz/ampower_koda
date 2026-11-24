"""Persist prompt-cache usage on an Agent Request without double-counting input."""

from __future__ import annotations

import re

import frappe

from . import reasoning_catalog
from .errors import log_agent_error
from .run_control import set_request_value

DOCTYPE = "Agent Request"
_FIELDS = (
    "cache_input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cost_estimate",
)


def persist_usage(
    request_name: str,
    total_tokens: int,
    *,
    input_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost_delta: float = 0,
) -> None:
    """Add one billed model response to the request's cache ledger.

    Provider ``input_tokens`` already includes cache reads and writes, so the
    whole-run hit rate is ``sum(read) / sum(input)``.
    """
    if not request_name:
        return
    try:
        # Keep the long-standing total usable even before a site has migrated
        # the new cache fields.
        set_request_value(request_name, "tokens_used", max(0, int(total_tokens or 0)))
        input_delta = max(0, int(input_tokens or 0))
        read_delta = max(0, min(input_delta, int(cache_read_tokens or 0)))
        write_delta = max(0, min(input_delta - read_delta, int(cache_write_tokens or 0)))
        cost = max(0.0, float(cost_delta or 0))
        if input_delta or cost:
            current = frappe.db.get_value(DOCTYPE, request_name, list(_FIELDS), as_dict=True) or {}
            set_request_value(request_name, {
                "cache_input_tokens": _value(current, "cache_input_tokens") + input_delta,
                "cache_read_tokens": _value(current, "cache_read_tokens") + read_delta,
                "cache_write_tokens": _value(current, "cache_write_tokens") + write_delta,
                "cost_estimate": round(_number(current, "cost_estimate") + cost, 8),
            })
        frappe.db.commit()
    except Exception:  # noqa: BLE001 -- optional metrics must not fail a run before migration.
        log_agent_error(
            "Agent Cache Usage Persist",
            f"request={request_name}\n{frappe.get_traceback()}",
        )


def _value(row, key: str) -> int:
    if isinstance(row, dict):
        value = row.get(key)
    else:
        value = getattr(row, key, 0)
    return max(0, int(value or 0))


def _number(row, key: str) -> float:
    if isinstance(row, dict):
        value = row.get(key)
    else:
        value = getattr(row, key, 0)
    return max(0.0, float(value or 0))


# (input, cache read, output, cache write). A model not listed here is priced from OpenRouter's list.
MODEL_PRICES_PER_MILLION = {
    "gpt-6-luna": (0.10, 0.01, 0.50, 0.125),
    "gpt-5": (1.25, 0.125, 10.00, 1.25),
    "gpt-5-codex": (1.25, 0.125, 10.00, 1.25),
    "gpt-5-mini": (0.25, 0.025, 2.00, 0.25),
    "gpt-5-nano": (0.05, 0.005, 0.40, 0.05),
    "gpt-4.1": (2.00, 0.50, 8.00, 2.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60, 0.40),
    "gpt-4o": (2.50, 1.25, 10.00, 2.50),
    "gpt-4o-mini": (0.15, 0.075, 0.60, 0.15),
    "claude-sonnet-4": (3.00, 0.30, 15.00, 3.75),
    "claude-sonnet-4-5": (3.00, 0.30, 15.00, 3.75),
    "claude-sonnet-4-6": (3.00, 0.30, 15.00, 3.75),
    "claude-sonnet-5": (2.00, 0.20, 10.00, 2.50),
    "claude-opus-4-5": (5.00, 0.50, 25.00, 6.25),
    "claude-haiku-4-5": (1.00, 0.10, 5.00, 1.25),
    "claude-opus-5-5": (4.00, 0.20, 20.00, 5.00),
    "claude-sonnet-5-5": (2.00, 0.10, 10.00, 2.50),
    "claude-haiku-5-5": (0.10, 0.01, 0.50, 0.125),
}

# Models whose whole request is billed at a higher tier past an input size: (threshold, prices).
LONG_CONTEXT_PRICES_PER_MILLION = {
    "gpt-6-luna": (272_000, (0.20, 0.02, 0.75, 0.25)),
    "claude-haiku-5-5": (100_000, (0.50, 0.05, 2.50, 0.625)),
}


HOUR_WRITE_KEY = "ephemeral_1h_input_tokens"
WRITE_LIFETIME_KEYS = ("ephemeral_5m_input_tokens", HOUR_WRITE_KEY)
# The price of a one-hour write, as a multiple of the input price; listed write prices are five-minute ones.
HOUR_WRITE_PRICE_FACTOR = 2


def cache_write_tokens(details: dict) -> int:
    """Prompt tokens written to the cache, from LangChain's ``input_token_details``.

    LangChain's Anthropic adapter reports a write as ``cache_creation`` until Anthropic splits
    it by lifetime; then the counts move to ``ephemeral_5m_input_tokens`` and
    ``ephemeral_1h_input_tokens`` and ``cache_creation`` reads 0. Either form is a total of
    the same writes, so the larger is taken.
    """
    if not details:
        return 0
    by_lifetime = sum(int(details.get(key) or 0) for key in WRITE_LIFETIME_KEYS)
    return max(0, int(details.get("cache_creation") or 0), by_lifetime)


def provider_cost(response) -> float:
    """The billed cost of one response: OpenRouter's reported cost, else estimated from its usage.

    Only OpenRouter reports a cost; for other providers the response's token usage
    is priced, so the request's cost and spend limit are not stuck at zero.
    """
    metadata = getattr(response, "response_metadata", None) or {}
    usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    if isinstance(usage, dict) and usage.get("cost") is not None:
        try:
            return max(0.0, float(usage.get("cost") or 0))
        except (TypeError, ValueError):
            pass
    model = (metadata.get("model_name") or metadata.get("model") or "") if isinstance(metadata, dict) else ""
    return estimated_cost(str(model), getattr(response, "usage_metadata", None) or {})


def _model_price_tiers(model: str) -> list:
    """The model's prices in the form of ``reasoning_catalog.model_price_tiers``; [] when unknown."""
    if not model:
        return []
    if model in MODEL_PRICES_PER_MILLION:
        tiers = [(0, MODEL_PRICES_PER_MILLION[model])]
        if model in LONG_CONTEXT_PRICES_PER_MILLION:
            tiers.append(LONG_CONTEXT_PRICES_PER_MILLION[model])
        return tiers
    try:
        return reasoning_catalog.model_price_tiers(model)
    except Exception:  # noqa: BLE001 -- an unpriced response must not fail the run that paid for it.
        log_agent_error("Agent Cost Estimate", f"model={model}\n{frappe.get_traceback()}")
        return []


def estimated_cost(model: str, usage: dict) -> float:
    """Price LangChain usage metadata at list prices; 0.0 for an unknown model or no usage."""
    name = model.strip().lower().rsplit("/", 1)[-1]  # "openai/gpt-6-luna" -> "gpt-6-luna"
    base = re.sub(r"-\d{4}-?\d{2}-?\d{2}$", "", name)  # a dated snapshot is priced as its model
    if not isinstance(usage, dict):
        return 0.0
    tiers = _model_price_tiers(base)
    if not tiers:
        return 0.0
    details = usage.get("input_token_details") or {}
    try:
        input_tokens = max(0, int(usage.get("input_tokens") or 0))
        read = min(input_tokens, max(0, int(details.get("cache_read") or 0)))
        write = min(input_tokens - read, cache_write_tokens(details))
        hour_write = min(write, max(0, int(details.get(HOUR_WRITE_KEY) or 0)))
        output = max(0, int(usage.get("output_tokens") or 0))
    except (TypeError, ValueError):
        return 0.0
    prices = tiers[0][1]
    for threshold, tier_prices in tiers[1:]:
        if input_tokens > threshold:
            prices = tier_prices
    uncached, read_price, output_price, write_price = prices
    fresh = input_tokens - read - write
    write_cost = (write - hour_write) * write_price + hour_write * uncached * HOUR_WRITE_PRICE_FACTOR
    return round((fresh * uncached + read * read_price + write_cost + output * output_price)
                 / 1_000_000, 8)
