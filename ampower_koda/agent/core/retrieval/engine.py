"""One entry point, six stages, everything off-prompt."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..constants import (
    BRIDGE_DEFINITION_BONUS,
    BRIDGE_MAX_SYMBOLS,
    BRIDGE_SCORE_PER_TERM,
    BRIDGE_TERM_WEIGHT,
    MAX_SEARCH_LIMIT,
    SEED_LIMIT,
    SOURCE_LIMIT,
    SYMBOL_EXPANSION_DEFINITION_BONUS,
    SYMBOL_EXPANSION_MAX,
    SYMBOL_EXPANSION_WEIGHT,
    VIEW_RANK_DECAY,
)
from ..config.schema import RerankConfig, RetrievalConfig
from ..contracts.mirrors import MirrorSet
from ..contracts.rerank import Reranker
from ..contracts.repository import RepositoryIndex
from ..contracts.retrieval import Hit, LegResult, SearchResult
from ..contracts.session import CoChangeMemory
from ..graph.edges import CodeGraph
from . import select
from .bm25 import LexicalIndex, ScoredDocument, build_lexical_index, score
from .confidence import hit_coverage, margin_of
from .excerpts import best_chunks
from .fusion import fuse
from .legs import graph_leg, history_leg, related_leg, seeds_from, structural_leg
from .query import QueryPlan, merged_weight, named_paths, plan_query
from .rerank import rerank
from .tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Retriever:
    """Everything a search needs, built once at cold start."""

    index: RepositoryIndex
    lexical: LexicalIndex
    graph: CodeGraph
    cochange: CoChangeMemory = field(default_factory=CoChangeMemory)

    prose: frozenset[str] = frozenset()
    """Digests of mostly-comment chunks, precomputed from the one definition in
    :mod:`bm25` so selection and reranking cannot disagree about what prose is."""

    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank_config: RerankConfig = field(default_factory=RerankConfig)
    reranker: Reranker | None = None

    mirrors: MirrorSet = field(default_factory=MirrorSet)


def build_retriever(
    index: RepositoryIndex,
    graph: CodeGraph,
    *,
    cochange: CoChangeMemory | None = None,
    config: RetrievalConfig | None = None,
    rerank_config: RerankConfig | None = None,
    reranker: Reranker | None = None,
    mirrors: MirrorSet | None = None,
) -> Retriever:
    """Build the retriever. The expensive half of cold start after indexing."""
    lexical = build_lexical_index(index)
    return Retriever(
        index=index,
        lexical=lexical,
        graph=graph,
        cochange=cochange or CoChangeMemory(),
        config=config or RetrievalConfig(),
        rerank_config=rerank_config or RerankConfig(),
        reranker=reranker,
        mirrors=mirrors or MirrorSet(),
        prose=frozenset(
            document.chunk.digest for document in lexical.documents if document.prose
        ),
    )


def search(
    retriever: Retriever,
    query: str,
    *,
    limit: int | None = None,
) -> SearchResult:
    """Run the whole pipeline and return the visible list."""
    limit = max(1, min(retriever.config.limit if limit is None else limit, MAX_SEARCH_LIMIT))
    plan = plan_query(query, retriever.lexical)

    lexical = _lexical_leg(retriever, plan)
    named = _named_leg(retriever, plan.original)
    if lexical.is_empty and named.is_empty:
        return SearchResult(notes=("no lexical match",), legs_run=("lexical",))

    legs: list[LegResult] = [lexical, named]
    if retriever.config.expand:
        seeds = seeds_from((*named.hits, *lexical.hits), limit=SEED_LIMIT)
        legs.append(related_leg(seeds, retriever.index, retriever.graph, query=plan.original))
        legs.append(structural_leg(seeds, retriever.index, retriever.graph))
        legs.append(graph_leg(seeds, retriever.index, retriever.graph, query=plan.original))
        legs.append(history_leg(seeds, retriever.index, retriever.cochange, query=plan.original))

    ran = tuple(result.leg for result in legs if not result.is_empty)
    notes = tuple(note for result in legs for note in result.notes)

    # Apply file diversity before the candidate cutoff, so one large file
    # cannot consume the union and hide companions from the reranker.
    fused = fuse(legs, limit=sum(len(leg.hits) for leg in legs))
    ranking = rerank(fused, plan.original, model=retriever.reranker,
                     config=retriever.rerank_config, prose=retriever.prose,
                     index=retriever.index, graph=retriever.graph)
    hits = select.diversify(ranking.hits, limit=limit)

    return SearchResult(
        hits=hits,
        confidence=hit_coverage(retriever.lexical, plan.original, hits),
        margin=margin_of(hits),
        notes=(*notes, *ranking.notes),
        legs_run=ran,
        reranked=ranking.reranked,
    )


def _named_leg(retriever: Retriever, query: str) -> LegResult:
    """A named file participates before ranking, even outside BM25's top forty."""
    hits = []
    for path in named_paths(query, retriever.index.paths):
        hits.extend(Hit(chunk=chunk, score=1.0) for chunk in
                    best_chunks(retriever.index, path, query))
    return LegResult(leg="named", hits=tuple(hits))


def _lexical_leg(
    retriever: Retriever,
    plan: QueryPlan,
) -> LegResult:
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
        # Expansion hands back raw BM25 scores, in the tens; the merge above
        # is in view weights, around one. Scale by the best direct score so a
        # chunk found only through a symbol's name can add to, but never
        # outvote, the chunks the query itself matched.
        scale = primary[0].score if primary else 1.0
        for position, value in _expand(retriever, plan).items():
            merged[position] = merged.get(position, 0.0) + min(0.5, value / scale)

    ordered = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:SOURCE_LIMIT]
    hits = tuple(
        Hit(chunk=retriever.lexical.documents[position].chunk, score=value)
        for position, value in ordered
    )
    return LegResult(leg="lexical", hits=hits)


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
