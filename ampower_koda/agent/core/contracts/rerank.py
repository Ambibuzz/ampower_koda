"""The network-free boundary to a dedicated relevance scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RerankScores:
    """One score per supplied document, in input order; an error means fallback."""

    scores: tuple[float, ...] = ()
    error: str = ""


class Reranker(Protocol):
    def score(self, query: str, documents: tuple[str, ...]) -> RerankScores:
        """Score bounded query/document pairs without generating prose."""
        ...
