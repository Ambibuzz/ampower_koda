"""One allocator, constructed once per session, owning every context budget."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import (
    BUDGET_CEILED_REGIONS,
    BUDGET_FLOOR_CEILING,
    BUDGET_FLOORS,
    BUDGET_SHARES,
    BUDGET_TOKENS_PER_HOT_RESULT,
    BUDGET_TOTAL_CEILING,
    COMPACTION_TRIGGER_FRACTION,
    MAP_MAX_TOKENS,
    MAX_TURN_TOKENS_MARGINAL,
    MAX_TURN_TOKENS_OBSERVED,
    MEMORY_MAX_TOKENS,
    OBSERVED_WINDOW_MULTIPLE,
    REPLY_HEADROOM_TOKENS,
)
from ..errors import ConfigError


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Every ceiling one session runs under, derived from its window."""

    window: int

    ledger: int
    working_set: int
    hot_results: int
    hot_count: int
    fold: int

    repo_map: int
    memory: int

    compaction_trigger: int
    reply_headroom: int
    marginal_turn: int
    observed_turn: int

    @property
    def claimed(self) -> int:
        """What the allocator has spoken for, map and memory included."""
        return (
            self.ledger
            + self.working_set
            + self.hot_results
            + self.fold
            + self.repo_map
            + self.memory
        )

    @property
    def claimed_share(self) -> float:
        return self.claimed / self.window if self.window else 0.0

    def summary(self) -> str:
        return (
            f"{self.window:,}-token window: "
            f"ledger {self.ledger:,}, working set {self.working_set:,}, "
            f"hot results {self.hot_results:,} ({self.hot_count}), "
            f"map {self.repo_map:,}, memory {self.memory:,} "
            f"— {self.claimed_share:.0%} claimed, compaction at {self.compaction_trigger:,}"
        )


def allocate(
    window: int,
    *,
    ledger_override: int = 0,
    map_tokens: int = MAP_MAX_TOKENS,
    memory_tokens: int = MEMORY_MAX_TOKENS,
) -> ContextBudget:
    """Derive every context budget from the window size."""
    if window <= 0:
        raise ConfigError("context.window_tokens", "must be positive")

    prefix = _prefix_blocks(window, map_tokens, memory_tokens)
    regions = _fit(
        {
            "ledger": ledger_override or _at_least("ledger", window),
            "working_set": _at_least("working_set", window),
            "hot_results": _at_least("hot_results", window),
            "fold": _at_least("fold", window),
            "repo_map": prefix.repo_map,
            "memory": prefix.memory,
        },
        window,
        protected="ledger" if ledger_override else "",
    )

    return ContextBudget(
        window=window,
        ledger=regions["ledger"],
        working_set=regions["working_set"],
        hot_results=regions["hot_results"],
        hot_count=max(BUDGET_FLOORS["hot_count"], round(window / BUDGET_TOKENS_PER_HOT_RESULT)),
        fold=regions["fold"],
        repo_map=regions["repo_map"],
        memory=regions["memory"],
        compaction_trigger=int(window * COMPACTION_TRIGGER_FRACTION),
        reply_headroom=min(REPLY_HEADROOM_TOKENS, window // 2),
        marginal_turn=MAX_TURN_TOKENS_MARGINAL,
        observed_turn=min(MAX_TURN_TOKENS_OBSERVED, int(window * OBSERVED_WINDOW_MULTIPLE)),
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


@dataclass(frozen=True, slots=True)
class _PrefixBlocks:
    repo_map: int
    memory: int


def _prefix_blocks(window: int, map_tokens: int, memory_tokens: int) -> _PrefixBlocks:
    """Split one quarter-window ceiling between the map and the memory files."""
    ceiling = int(window * BUDGET_FLOOR_CEILING)
    memory = min(memory_tokens, ceiling)
    return _PrefixBlocks(repo_map=min(map_tokens, max(0, ceiling - memory)), memory=memory)


def _at_least(region: str, window: int) -> int:
    """``max(share × window, min(floor, ¼ × window))``."""
    share = BUDGET_SHARES.get(region, 0.0) * window
    floor = min(BUDGET_FLOORS.get(region, 0), BUDGET_FLOOR_CEILING * window)
    return round(max(share, floor))
