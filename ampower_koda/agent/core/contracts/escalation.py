"""What escalation is allowed to do, and what it must report having done."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from .retrieval import SearchResult

Rung = Literal["none", "translate", "expand", "explore"]

FanoutAngle = Literal["symbol", "path", "behavior", "data_flow"]


@dataclass(frozen=True, slots=True)
class SideUsage:
    """Tokens spent outside the main loop, so the loop can charge them back."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def plus(self, other: SideUsage) -> SideUsage:
        return SideUsage(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class Rewrite:
    """One reformulation of the question, and the angle it was written from."""

    text: str
    angle: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class Formulation:
    """What one utility call produced: rewrites, cost, and whether it worked."""

    rewrites: tuple[Rewrite, ...] = ()
    usage: SideUsage = SideUsage()
    failed: bool = False
    detail: str = ""
    """Why it failed, in the words the caller will read in ``notes``. Empty on
    success."""

    @property
    def is_empty(self) -> bool:
        return not self.rewrites


@runtime_checkable
class Formulator(Protocol):
    """The one seam through which this package can cause a model call."""

    def translate(self, question: str, repo_map: str) -> Formulation:
        """Question → the identifiers this repository actually uses."""

    def fan_out(self, question: str, repo_map: str) -> Formulation:
        """Question → up to four rewrites along four distinct angles."""


@dataclass(frozen=True, slots=True)
class Escalation:
    """One escalated search: the result, the path taken, and the bill."""

    result: SearchResult
    rungs: tuple[Rung, ...] = ()
    """Every rung actually attempted, in order. Empty means the ladder declined
    to climb — a confident result, or escalation switched off."""

    usage: SideUsage = SideUsage()
    """Only what was actually spent. A rung served from the memo costs nothing
    and is charged nothing; ``rungs`` will still name it, because it was still
    climbed."""

    readings: tuple[str, ...] = ()
    """Every substitution the answer was retrieved under, in the order the rungs
    produced them — rendered as ``[read as: …]`` so a reader is never shown
    results for a question they did not ask without being told.

    A tuple rather than one string because two rungs can both contribute, and
    collapsing them lost which hits came from which. The authoritative label is
    the one on each :class:`~…contracts.retrieval.Hit`; this is the summary."""

    original_confidence: float = 0.0
    """The unescalated result's confidence — the question's own number.

    Carried separately because ``result.confidence`` after a translation is
    measured against *the substitution*, and the two are not comparable: a
    retry whose query is one rare identifier carries all of its own IDF mass by
    construction and scores ~1.0 whether or not it answered anything. Reporting
    that as the question's confidence would be the thermometer measuring
    itself."""

    notes: tuple[str, ...] = ()

    @property
    def escalated(self) -> bool:
        return bool(self.rungs)

    @property
    def reading(self) -> str:
        """Every substitution, joined. Convenience for a one-line header."""
        return " | ".join(self.readings)

    @property
    def improved(self) -> bool:
        """Whether climbing actually changed the answer."""
        return bool(self.readings)
