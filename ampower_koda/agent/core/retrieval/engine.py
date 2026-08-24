"""One entry point, six stages, everything off-prompt."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..constants import (
    BRIDGE_DEFINITION_BONUS,
    BRIDGE_MAX_SYMBOLS,
    BRIDGE_SCORE_PER_TERM,
    BRIDGE_TERM_WEIGHT,
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    SEED_LIMIT,
    SOURCE_LIMIT,
    SYMBOL_EXPANSION_DEFINITION_BONUS,
    SYMBOL_EXPANSION_MAX,
    SYMBOL_EXPANSION_WEIGHT,
    VIEW_RANK_DECAY,
)
from ..contracts.repo_map import FileRanks, MirrorSet
from ..contracts.repository import RepositoryIndex
from ..contracts.retrieval import Hit, LegResult, SearchResult
from ..contracts.session import CoChangeMemory
from ..graph.edges import CodeGraph
from . import select
from .bm25 import LexicalIndex, ScoredDocument, build_lexical_index, score
from .confidence import compute_confidence, margin_of
from .fusion import fuse
from .legs import graph_leg, history_leg, seeds_from, structural_leg
from .query import QueryPlan, merged_weight, plan_query
from .rerank import RerankContext, rerank
from .tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Retriever:
    """Everything a search needs, built once at cold start."""

    index: RepositoryIndex
    lexical: LexicalIndex
    graph: CodeGraph
    ranks: FileRanks
    mirrors: MirrorSet = field(default_factory=MirrorSet)
    cochange: CoChangeMemory = field(default_factory=CoChangeMemory)

    prose: frozenset[str] = frozenset()
    """Digests of mostly-comment chunks, precomputed from the one definition in
    :mod:`bm25` so selection and reranking cannot disagree about what prose is."""


def build_retriever(
    index: RepositoryIndex,
    graph: CodeGraph,
    ranks: FileRanks,
    *,
    mirrors: MirrorSet | None = None,
    cochange: CoChangeMemory | None = None,
) -> Retriever:
    """Build the retriever. The expensive half of cold start after indexing."""
    lexical = build_lexical_index(index)
    return Retriever(
        index=index,
        lexical=lexical,
        graph=graph,
        ranks=ranks,
        mirrors=mirrors or MirrorSet(),
        cochange=cochange or CoChangeMemory(),
        prose=frozenset(
            document.chunk.digest for document in lexical.documents if document.prose
        ),
    )


def search(
    retriever: Retriever,
    query: str,
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> SearchResult:
    """Run the whole pipeline and return the visible list."""
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))
    plan = plan_query(query, retriever.lexical)

    lexical, scored = _lexical_leg(retriever, plan)
    if lexical.is_empty:
        return SearchResult(notes=("no lexical match",), legs_run=("lexical",))

    legs: list[LegResult] = [lexical]
    if plan.anchored:
        seeds = seeds_from(lexical.hits, limit=SEED_LIMIT)
        legs.append(structural_leg(seeds, retriever.index, retriever.graph))
        legs.append(graph_leg(seeds, retriever.index, retriever.graph))
        legs.append(history_leg(seeds, retriever.index, retriever.cochange))

    ran = tuple(result.leg for result in legs if not result.is_empty)
    notes = tuple(note for result in legs for note in result.notes)

    if len(ran) < 2:
        hits = _finish(retriever, lexical.hits, lexical.hits, limit)
        return SearchResult(
            hits=hits,
            confidence=compute_confidence(retriever.lexical, plan.original, scored),
            margin=margin_of(hits),
            notes=notes,
            legs_run=ran,
        )

    fused = fuse(legs)
    context = RerankContext(
        query=plan.original,
        ranks=retriever.ranks,
        mirrors=retriever.mirrors,
        index=retriever.lexical,
        prose=retriever.prose,
    )
    hits = _finish(retriever, rerank(fused, context), lexical.hits, limit)

    return SearchResult(
        hits=hits,
        confidence=compute_confidence(retriever.lexical, plan.original, scored),
        margin=margin_of(hits),
        notes=notes,
        legs_run=ran,
    )


def _lexical_leg(
    retriever: Retriever,
    plan: QueryPlan,
) -> tuple[LegResult, tuple[ScoredDocument, ...]]:
    """BM25 across every view, merged by weighted rank, expanded if weak."""
    merged: dict[int, float] = {}
    primary: tuple[ScoredDocument, ...] = ()

    for view in plan.views:
        results = score(retriever.lexical, view.text, limit=SOURCE_LIMIT)
        if view.kind == "original":
            primary = results
        for rank, result in enumerate(results):
            contribution = merged_weight(view.weight, rank, VIEW_RANK_DECAY)
            merged[result.position] = merged.get(result.position, 0.0) + contribution

    if _looks_weak(retriever.lexical, plan, primary):
        for position, value in _expand(retriever, plan).items():
            merged[position] = merged.get(position, 0.0) + value

    ordered = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:SOURCE_LIMIT]
    hits = tuple(
        Hit(chunk=retriever.lexical.documents[position].chunk, score=value)
        for position, value in ordered
    )
    return LegResult(leg="lexical", hits=hits), primary


def _looks_weak(index: LexicalIndex, plan: QueryPlan, primary: Sequence[ScoredDocument]) -> bool:
    """Whether the first pass justifies spending a second one."""
    if not primary:
        return True
    if plan.route == "exact":
        return False

    terms = max(1, len(tokenize(plan.original, is_query=True)))
    if primary[0].score / terms < BRIDGE_SCORE_PER_TERM:
        return True
    return index.documents[primary[0].position].prose


def _expand(retriever: Retriever, plan: QueryPlan) -> dict[int, float]:
    """Symbol expansion and the prose bridge, merged."""
    contributions: dict[int, float] = {}
    query_terms = frozenset(tokenize(plan.original, is_query=True))
    if not query_terms:
        return contributions

    for symbol in _overlapping_symbols(retriever.lexical, query_terms):
        _accumulate(
            retriever,
            contributions,
            symbol,
            weight=SYMBOL_EXPANSION_WEIGHT,
            definition_bonus=SYMBOL_EXPANSION_DEFINITION_BONUS,
        )

    for identifier in _bridge_identifiers(retriever, plan):
        _accumulate(
            retriever,
            contributions,
            identifier,
            weight=BRIDGE_TERM_WEIGHT,
            definition_bonus=BRIDGE_DEFINITION_BONUS,
        )

    return contributions


def _accumulate(
    retriever: Retriever,
    contributions: dict[int, float],
    term: str,
    *,
    weight: float,
    definition_bonus: float,
) -> None:
    """Add one expansion term's results, decayed by *their own* rank."""
    for rank, result in enumerate(score(retriever.lexical, term, limit=BRIDGE_MAX_SYMBOLS)):
        bonus = (
            definition_bonus
            if retriever.lexical.documents[result.position].chunk.identity == term
            else 1.0
        )
        contributions[result.position] = contributions.get(result.position, 0.0) + (
            result.score * weight * bonus / (1.0 + rank)
        )


def _overlapping_symbols(index: LexicalIndex, terms: frozenset[str]) -> tuple[str, ...]:
    """Identifiers whose own tokens overlap the query, best overlap first."""
    scored: dict[str, float] = {}
    for document in index.documents:
        symbol = document.chunk.identity
        if not symbol or symbol in scored:
            continue
        parts = frozenset(tokenize(symbol.replace(".", " ")))
        overlap = len(parts & terms)
        if overlap:
            scored[symbol] = overlap / (1.0 + 0.5 * (len(parts) - overlap))

    return tuple(sorted(scored, key=lambda name: (-scored[name], name))[:SYMBOL_EXPANSION_MAX])


def _bridge_identifiers(retriever: Retriever, plan: QueryPlan) -> tuple[str, ...]:
    """Identifiers harvested from the top results of the first pass."""
    top = score(retriever.lexical, plan.original, limit=BRIDGE_MAX_SYMBOLS)
    if not top:
        return ()

    seen: dict[str, int] = {}
    for result in top:
        symbol = retriever.lexical.documents[result.position].chunk.identity
        if symbol:
            bare = symbol.rsplit(".", 1)[-1]
            seen[bare] = seen.get(bare, 0) + 1

    total = len(top)
    scored = {
        name: (hits / total) * _spread_bonus(retriever.lexical, name)
        for name, hits in seen.items()
    }
    return tuple(sorted(scored, key=lambda name: (-scored[name], name))[:BRIDGE_MAX_SYMBOLS])


def _spread_bonus(index: LexicalIndex, name: str) -> float:
    """``ln(1 + N / spread)`` — how concentrated this name is in the corpus."""
    from math import log

    spread = max(1, index.document_frequency.get(name.lower(), 1))
    return log(1.0 + max(index.counted, 1) / spread)


def _finish(
    retriever: Retriever,
    ranked: Sequence[Hit],
    original: Sequence[Hit],
    limit: int,
) -> tuple[Hit, ...]:
    """The floors and the diversity cap, in the one order that is correct."""
    hits = select.penalise_prose(ranked, retriever.prose)
    hits = select.decay_same_file(hits)
    hits = select.preserve_original_window(hits, original)
    hits = select.diversify(hits, limit=limit + 1)
    hits = select.add_supplemental(hits, ranked)
    return select.preserve_direct_files(hits, original[:3], limit=limit)
