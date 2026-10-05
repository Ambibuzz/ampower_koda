"""One allocator, constructed once per session, owning every context budget."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import (
    BUDGET_CEILED_REGIONS,
    BUDGET_FLOOR_CEILING,
    BUDGET_FLOORS,
    BUDGET_SHARES,
    BUDGET_TOTAL_CEILING,
    MAX_TURN_TOKENS_MARGINAL,
    MAX_TURN_TOKENS_OBSERVED,
    MEMORY_MAX_TOKENS,
    OBSERVED_WINDOW_MULTIPLE,
)
from ..errors import ConfigError
from .request import DEFAULT_INPUT_TOKENS

# More history capacity must not inflate automatically injected context.
AUTO_CONTEXT_WINDOW = 32_000


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Every ceiling one session runs under, derived from its window."""

    window: int

    ledger: int
    working_set: int
    fold: int

    memory: int

    marginal_turn: int
    observed_turn: int
    input_tokens: int = DEFAULT_INPUT_TOKENS


def allocate(
    window: int,
    *,
    ledger_override: int = 0,
    memory_tokens: int = MEMORY_MAX_TOKENS,
    input_tokens: int = DEFAULT_INPUT_TOKENS,
) -> ContextBudget:
    """Derive every context budget from the window size."""
    if window <= 0:
        raise ConfigError("context.window_tokens", "must be positive")
    if input_tokens <= 0:
        raise ConfigError("context.input_tokens", "must be positive")
    active = min(window, input_tokens)

    automatic = min(active, AUTO_CONTEXT_WINDOW)
    memory = min(memory_tokens, int(automatic * BUDGET_FLOOR_CEILING))
    regions = _fit(
        {
            "ledger": ledger_override or _at_least("ledger", automatic),
            "working_set": _at_least("working_set", automatic),
            "fold": _at_least("fold", automatic),
            "memory": memory,
        },
        active,
        protected="ledger" if ledger_override else "",
    )

    return ContextBudget(
        window=window,
        ledger=regions["ledger"],
        working_set=regions["working_set"],
        fold=regions["fold"],
        memory=regions["memory"],
        marginal_turn=max(MAX_TURN_TOKENS_MARGINAL, 2 * active),
        observed_turn=min(MAX_TURN_TOKENS_OBSERVED, int(window * OBSERVED_WINDOW_MULTIPLE)),
        input_tokens=active,
    )


def _fit(regions: dict[str, int], window: int, *, protected: str = "") -> dict[str, int]:
    """Scale the out-of-transcript regions down together if they overrun."""
    ceiling = int(window * BUDGET_TOTAL_CEILING)
    governed = {name: regions[name] for name in BUDGET_CEILED_REGIONS if name in regions}
    total = sum(governed.values())
    if total <= ceiling:
        return regions

    fixed = governed.get(protected, 0)
    scalable = total - fixed
    room = max(0, ceiling - fixed)
    if scalable <= 0:
        return regions

    factor = room / scalable
    return {
        name: value
        if name not in governed or name == protected
        else max(1, int(value * factor))
        for name, value in regions.items()
    }


def _at_least(region: str, window: int) -> int:
    """``max(share × window, min(floor, ¼ × window))``."""
    share = BUDGET_SHARES.get(region, 0.0) * window
    floor = min(BUDGET_FLOORS.get(region, 0), BUDGET_FLOOR_CEILING * window)
    return round(max(share, floor))
