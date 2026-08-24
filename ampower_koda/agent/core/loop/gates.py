"""Every way a turn ends, as pure functions of what the turn has spent."""

from __future__ import annotations

from collections.abc import Container, Sequence
from dataclasses import dataclass, replace

from ..constants import MAX_TURN_TOKENS_MARGINAL, MAX_TURN_TOKENS_OBSERVED
from . import nudges

DRY_ROUNDS_LIMIT = 3

ROUNDS_RESERVE = 2

EVIDENCE_MIN_LINE = 24

EVIDENCE_MIN_NOVEL = 240

EVIDENCE_NOVEL_SHARE = 0.5


@dataclass(frozen=True, slots=True)
class TurnMeters:
    """Everything the gates read, and nothing else."""

    round_index: int = 0
    max_rounds: int = 60

    processed: int = 0
    """Uncached input + cache writes + output. Charged against ``max_turn``."""

    observed: int = 0
    """The above plus cache reads. Charged against ``max_observed``."""

    max_turn: int = MAX_TURN_TOKENS_MARGINAL
    max_observed: int = MAX_TURN_TOKENS_OBSERVED

    dry_streak: int = 0
    continuations: int = 0
    coverage_fired: bool = False
    """Latched, so the coverage gate can fire at most once per turn. Latched
    *before* the check runs upstream — a gate that latches after an await can
    fire twice on a fast second round."""

    @property
    def rounds_left(self) -> int:
        return max(0, self.max_rounds - self.round_index)

    def charged(self, *, processed: int, observed: int) -> TurnMeters:
        return replace(
            self,
            processed=self.processed + processed,
            observed=self.observed + observed,
        )

    def next_round(self, *, dry: bool) -> TurnMeters:
        return replace(
            self,
            round_index=self.round_index + 1,
            dry_streak=self.dry_streak + 1 if dry else 0,
        )


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether to keep going, and what to say if not."""

    stop: bool = False
    force_terminal: bool = False
    nudge: nudges.Nudge | None = None
    reason: str = ""

    @property
    def carry_on(self) -> bool:
        return not self.stop and not self.force_terminal


def check(meters: TurnMeters) -> Decision:
    """The gates, in the order they are allowed to fire."""
    if meters.processed >= meters.max_turn:
        return Decision(force_terminal=True, nudge=nudges.budget(), reason="marginal budget")

    if meters.observed >= meters.max_observed:
        return Decision(
            force_terminal=True,
            nudge=nudges.budget(),
            reason="observed budget — cache reads must not buy infinite rounds",
        )

    if meters.rounds_left <= ROUNDS_RESERVE:
        return Decision(force_terminal=True, nudge=nudges.rounds(), reason="rounds reserve")

    if meters.dry_streak >= DRY_ROUNDS_LIMIT:
        return Decision(force_terminal=True, nudge=nudges.dry(), reason="dry rounds")

    return Decision()


def after_max_tokens(meters: TurnMeters) -> Decision:
    """The model's output hit the limit. Continue, or say plainly that it did."""
    if meters.continuations < nudges.MAX_CONTINUATIONS:
        return Decision(nudge=nudges.continuation(), reason="output limit")
    return Decision(stop=True, reason=nudges.CUT_OFF)


COVERAGE_MIN_CALLS = 3

COVERAGE_MAX_UNOPENED = 3


def coverage_gate(
    meters: TurnMeters,
    *,
    central: Sequence[str],
    opened: Container[str],
    discovery_calls: int,
) -> Decision:
    """Before the first answer of a turn: did it look at the obvious files?"""
    if meters.coverage_fired or discovery_calls < COVERAGE_MIN_CALLS:
        return Decision()

    missed = tuple(path for path in central if path not in opened)
    if not missed or len(missed) > COVERAGE_MAX_UNOPENED:
        return Decision()

    return Decision(nudge=nudges.coverage(missed), reason="coverage")


def late_tool_call() -> Decision:
    """A tool call arrived after the budget closed."""
    return Decision(stop=True, reason="tool call after the budget closed")


def evidence_yield(text: str, seen: set[str]) -> int:
    """Characters on substantial lines this turn has not seen before."""
    novel = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) < EVIDENCE_MIN_LINE or stripped in seen:
            continue
        seen.add(stripped)
        novel += len(stripped)
    return novel


def is_dry(novel: int, total: int) -> bool:
    """Both conditions. Either alone misclassifies a whole class of round."""
    return novel < EVIDENCE_MIN_NOVEL and novel <= EVIDENCE_NOVEL_SHARE * max(1, total)
