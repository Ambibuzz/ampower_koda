"""Persist prompt-cache usage on an Agent Request without double-counting input."""

from __future__ import annotations

import frappe

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


def provider_cost(response) -> float:
    """Return an OpenRouter-reported billed cost, or zero when unavailable."""
    metadata = getattr(response, "response_metadata", None) or {}
    usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return 0.0
    try:
        return max(0.0, float(usage.get("cost") or 0))
    except (TypeError, ValueError):
        return 0.0
