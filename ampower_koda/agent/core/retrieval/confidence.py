"""How much of the question the top result actually answered."""

from __future__ import annotations

from collections.abc import Sequence

from ..constants import AGREEMENT_LIFT, DENSE_CONFIDENCE_WEIGHT
from ..contracts.retrieval import Hit
from .bm25 import LexicalIndex, ScoredDocument
from .tokenize import tokenize


def compute_confidence(
    index: LexicalIndex,
    query: str,
    scored: Sequence[ScoredDocument],
    *,
    dense_confidence: float = 0.0,
    agreement: float = 0.0,
) -> float:
    """The share of the query's IDF mass the top document carries."""
    terms = tokenize(query, is_query=True)
    if not terms or not scored:
        return 0.0

    total = sum(index.idf(term) for term in set(terms))
    if total <= 0:
        return 0.0

    carried = sum(index.idf(term) for term in set(scored[0].matched))
    base = max(carried / total, dense_confidence * DENSE_CONFIDENCE_WEIGHT)
    return max(0.0, min(1.0, base + agreement * AGREEMENT_LIFT))


def margin_of(hits: Sequence[Hit]) -> float:
    """``(top − third) / top``, and ``0`` when there is no third."""
    if len(hits) < 3 or hits[0].score <= 0:
        return 0.0
    return max(0.0, (hits[0].score - hits[2].score) / hits[0].score)
