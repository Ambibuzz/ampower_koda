"""What a search returns, and what each leg contributes to it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

from ..errors import CoreError
from .chunks import Chunk

LegName = Literal["lexical", "dense", "fanout", "graph", "structure", "history"]

LEG_TRUST: Mapping[str, float] = MappingProxyType(
    {
        "lexical": 1.0,
        "dense": 0.8,
        "fanout": 0.6,
        "graph": 0.5,
        "structure": 0.4,
        "history": 0.3,
    }
)

UNKNOWN_LEG_TRUST = 0.5


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved chunk, with everything ranking needs to order it."""

    chunk: Chunk
    score: float = 0.0

    sources: Mapping[str, int] = field(default_factory=dict)
    """Leg name → the rank that leg gave this hit, 0-based. Empty for a hit that
    never went through fusion — a plain single-leg lexical result — which is
    exactly the case the reranker checks for before declining to run."""

    note: str = ""
    """A fact about *this hit* that the ranked list would otherwise destroy —
    "4 more definitions of this symbol", "read as: AnchorPolicy". Never
    decoration; if it is not something a reader would act on, it is not here."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))

    @property
    def path(self) -> str:
        return self.chunk.path

    @property
    def symbol(self) -> str:
        return self.chunk.identity

    @property
    def location(self) -> str:
        return self.chunk.location

    @property
    def leg_count(self) -> int:
        return len(self.sources)

    def with_score(self, score: float) -> Hit:
        return replace(self, score=score)

    def claimed_by(self, leg: str, rank: int) -> Hit:
        """Return this hit with one more leg's claim recorded."""
        sources = dict(self.sources)
        existing = sources.get(leg)
        sources[leg] = rank if existing is None else min(existing, rank)
        return replace(self, sources=sources)


@dataclass(frozen=True, slots=True)
class LegResult:
    """What one retrieval leg produced, before fusion."""

    leg: str
    hits: tuple[Hit, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.leg:
            raise CoreError("a leg result must name its leg")

    @property
    def is_empty(self) -> bool:
        return not self.hits


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The public shape of a search."""

    hits: tuple[Hit, ...] = ()

    confidence: float = 0.0
    """How much of the query's IDF mass the top document carries, lifted by
    agreement between independent legs. Read by the escalation ladder to decide
    whether a model call on *vocabulary* is worth making.

    Structure, graph and history deliberately do not vote. They are derived from
    the lexical hits, so letting them raise confidence would be measuring the
    thermometer with itself."""

    margin: float = 0.0
    """``(top − third) / top``. The *third* hit, not the second: a definition
    and its call site are the normal shape of two near-identical top hits, and
    comparing against the second would report a tie on every healthy query."""

    notes: tuple[str, ...] = ()
    legs_run: tuple[str, ...] = ()
    """Which legs actually ran. A query that reached one leg and a query that
    reached five can produce the same number of hits, and the difference is the
    single most useful thing to know when a result looks thin."""

    @property
    def is_empty(self) -> bool:
        return not self.hits

    @property
    def top(self) -> Hit | None:
        return self.hits[0] if self.hits else None
