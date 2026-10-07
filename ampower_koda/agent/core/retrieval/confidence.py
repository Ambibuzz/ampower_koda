"""How much of the question the top result actually answered."""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts.retrieval import Hit
from .bm25 import LexicalIndex
from .tokenize import tokenize


def margin_of(hits: Sequence[Hit]) -> float:
    """``(top − third) / top``, and ``0`` when there is no third."""
    if len(hits) < 3 or hits[0].score <= 0:
        return 0.0
    return max(0.0, (hits[0].score - hits[2].score) / hits[0].score)


def hit_coverage(index: LexicalIndex, query: str, hits: Sequence[Hit]) -> float:
    """Coverage of the selected top hit, not an earlier discarded BM25 winner."""
    terms = set(tokenize(query, is_query=True))
    if not terms or not hits:
        return 0.0
    top = hits[0]
    present = set(tokenize(f"{top.path} {top.symbol} {top.chunk.body}"))
    total = sum(index.idf(term) for term in terms)
    return sum(index.idf(term) for term in terms & present) / total if total > 0 else 0.0
