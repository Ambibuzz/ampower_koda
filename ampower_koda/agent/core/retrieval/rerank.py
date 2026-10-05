"""Bound a candidate batch, then use dedicated relevance scores unchanged."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from ..config.schema import RerankConfig
from ..contracts.rerank import Reranker
from ..contracts.repository import RepositoryIndex
from ..contracts.retrieval import Hit
from ..graph.edges import CodeGraph
from .excerpts import excerpt
from .fusion import FusedHit
from .select import decay_same_file, diversify, penalise_prose

_PURPOSE = "Locate implementation code and registration or configuration needed to investigate this request:\n"


@dataclass(frozen=True, slots=True)
class Ranking:
    hits: tuple[Hit, ...]
    reranked: bool = False
    notes: tuple[str, ...] = ()


def rerank(fused: Sequence[FusedHit], query: str, *, model: Reranker | None,
           config: RerankConfig, prose: frozenset[str] = frozenset(),
           index: RepositoryIndex | None = None, graph: CodeGraph | None = None) -> Ranking:
    """Never mix provider scores with retrieval priors or resurrect rejected hits."""
    candidates = diversify(tuple(entry.hit for entry in fused
                                 if any(char.isalnum() for char in entry.hit.chunk.body)),
                           limit=config.candidates, per_file=config.per_file)
    if not candidates:
        return Ranking(())

    def fallback(note: str = "") -> Ranking:
        hits = decay_same_file(penalise_prose(candidates, prose))
        return Ranking(hits, notes=(note,) if note else ())

    if not config.enabled or model is None:
        return fallback()  # Deliberately local: nothing failed, so nothing to note.

    bounded_query = query.strip()
    notes = ()
    query_limit = config.query_chars - len(_PURPOSE)
    if len(bounded_query) > query_limit:
        head = query_limit * 3 // 4
        tail = query_limit - head - 5
        bounded_query = bounded_query[:head] + "\n...\n" + bounded_query[-tail:]
        notes = ("reranker query shortened to its beginning and end",)
    documents = tuple(_document(hit, bounded_query, config.document_chars, index, graph)
                      for hit in candidates)
    try:
        result = model.score(_PURPOSE + bounded_query, documents)
        if result.error:
            return fallback("dedicated reranker failed; using local retrieval (" + result.error + ")")
        if len(result.scores) != len(candidates) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or not 0 <= value <= 1 for value in result.scores
        ):
            return fallback("invalid reranker scores; using local retrieval")
    except Exception:  # The injected service must not take down local search.
        return fallback("dedicated reranker failed; using local retrieval")

    hits = [hit.with_score(float(value)) for hit, value in zip(candidates, result.scores)
            if value >= config.min_score]
    hits.sort(key=lambda hit: (-hit.score, hit.location))
    if not hits:
        notes += ("no candidates met the reranker relevance floor",)
    return Ranking(tuple(hits), reranked=True, notes=notes)


def _document(hit: Hit, query: str, limit: int, index: RepositoryIndex | None,
              graph: CodeGraph | None) -> str:
    header = f"File: {hit.path}\nSymbol: {hit.symbol}\nLines: {hit.chunk.span}\n"
    context = []
    analysis = index.files.get(hit.path) if index is not None else None
    if analysis is not None and analysis.chunks:
        first = min(analysis.chunks, key=lambda chunk: chunk.span.start)
        if first.digest != hit.chunk.digest:
            context.append("File context:\n" + excerpt(first.body, max_chars=400))
    if graph is not None:
        related = set()
        for edge in (*graph.out_edges(hit.path), *graph.in_edges(hit.path)):
            if edge.kind in {"feature", "rpc", "registration", "test"}:
                other = edge.target if edge.source == hit.path else edge.source
                related.add(f"{edge.kind}: {other} ({edge.symbol})")
        context.extend(sorted(related)[:4])
    # Keep at least half the document for the actual selected source span.
    header = (header + "\n".join(context))[:limit // 2 - 9].rstrip() + "\nSource:\n"
    return header + excerpt(hit.chunk.body, query, max_chars=limit - len(header))
