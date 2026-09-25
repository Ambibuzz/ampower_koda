"""Budget the complete model-facing request, independently of cache hits."""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..tokens import estimate_tokens

DEFAULT_INPUT_TOKENS = 150_000
MAX_WINDOW_SHARE = 0.9
MESSAGE_OVERHEAD = 12


def input_limit(window: int, output_tokens: int, ceiling: int = DEFAULT_INPUT_TOKENS) -> int:
    """Cap working input by the configured ceiling and usable model capacity."""
    margin = min(2_048, max(256, window // 20))
    return max(0, min(ceiling, int(window * MAX_WINDOW_SHARE), window - output_tokens - margin))


def cleanup_target(limit: int) -> int:
    """Free one third of the input allowance per cleanup, not a few tokens."""
    return max(0, limit * 2 // 3)


def serialized_tokens(value: object) -> int:
    if isinstance(value, str):
        return estimate_tokens(value)
    return estimate_tokens(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def message_tokens(message: object) -> int:
    """Count content, tool arguments and replayed reasoning, not tracing metadata."""
    size = MESSAGE_OVERHEAD + serialized_tokens(getattr(message, "content", ""))
    calls = getattr(message, "tool_calls", None)
    if calls:
        size += serialized_tokens(calls)
    extra = getattr(message, "additional_kwargs", None) or {}
    for key in ("reasoning", "reasoning_content", "reasoning_details"):
        if extra.get(key):
            details = extra[key]
            usage = getattr(message, "usage_metadata", None) or {}
            reported = (usage.get("output_token_details") or {}).get("reasoning")
            if (key == "reasoning_details" and isinstance(details, list)
                    and isinstance(reported, int) and not isinstance(reported, bool) and reported > 0
                    and any(isinstance(part, dict) and part.get("type") == "reasoning.encrypted"
                            for part in details)):
                # Encrypted reasoning is not tokenized as text: count the
                # provider-reported reasoning tokens plus the visible parts.
                # Without reported usage, fall back to the serialized size.
                visible = [{k: v for k, v in part.items() if k != "data"}
                           if isinstance(part, dict) and part.get("type") == "reasoning.encrypted"
                           else part for part in details]
                size += reported + serialized_tokens(visible)
            else:
                size += serialized_tokens(details)
    return size


def estimate_messages(messages: Sequence[object], schemas: Sequence[object] = ()) -> int:
    return sum(message_tokens(message) for message in messages) + serialized_tokens(schemas)
