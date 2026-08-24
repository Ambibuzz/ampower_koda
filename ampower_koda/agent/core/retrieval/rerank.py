"""Ten features, one linear model, no model call and no I/O."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log

from ..constants import RERANK_WEIGHTS
from ..contracts.repo_map import FileRanks, MirrorSet
from ..contracts.retrieval import Hit
from .bm25 import LexicalIndex
from .fusion import FusedHit, leg_trust, normalise
from .tokenize import tokenize


@dataclass(frozen=True, slots=True)
class RerankContext:
    """Everything the features need that is not the hit itself."""

    query: str
    ranks: FileRanks
    mirrors: MirrorSet
    index: LexicalIndex
    prose: frozenset[str] = frozenset()
    """Chunk digests that are mostly comment. Precomputed from the one
    definition in :mod:`bm25`, so the selection multiplier and this penalty can
    never disagree about what prose is."""


def rerank(
    fused: Sequence[FusedHit],
    context: RerankContext,
    *,
    weights: Mapping[str, float] = RERANK_WEIGHTS,
) -> tuple[Hit, ...]:
    """Reorder ``fused``, or return it untouched."""
    if len(fused) < 2:
        return tuple(entry.hit for entry in fused)

    legs = {leg for entry in fused for leg in entry.hit.sources}
    if len(legs) < 2:
        return tuple(entry.hit for entry in fused)

    priors = normalise({entry.hit.chunk.digest: entry.prior for entry in fused})
    terms = frozenset(tokenize(context.query, is_query=True))
    highest = max((context.ranks.of(entry.hit.path) for entry in fused), default=0.0) or 1.0

    scored = [
        (
            _score(entry, context, priors, terms, highest, weights),
            entry.hit,
        )
        for entry in fused
    ]
    scored.sort(key=lambda item: (-item[0], item[1].chunk.location))
    return tuple(hit.with_score(value) for value, hit in scored)


def features(
    entry: FusedHit,
    context: RerankContext,
    *,
    prior: float,
    terms: frozenset[str],
    max_centrality: float,
) -> dict[str, float]:
    """The ten features for one hit, each in ``[0, 1]``."""
    hit = entry.hit
    chunk = hit.chunk
    body_terms = frozenset(tokenize(chunk.body))
    symbol = chunk.identity

    return {
        "prior": prior,
        "centrality": context.ranks.of(hit.path) / max_centrality if max_centrality else 0.0,
        "leg_trust": max((leg_trust(leg) for leg in hit.sources), default=0.0),
        "term_coverage": (len(terms & body_terms) / len(terms)) if terms else 0.0,
        "symbol_match": _symbol_match(symbol, context.query, terms),
        "leg_agreement": _agreement(len(hit.sources)),
        "rarity": _rarity(context.index, symbol),
        "definitionness": 1.0 if chunk.kind == "symbol" else 0.0,
        "prose_penalty": 1.0 if chunk.digest in context.prose else 0.0,
        "vendored_copy": 1.0 if context.mirrors.contains(hit.path) else 0.0,
    }


def _score(
    entry: FusedHit,
    context: RerankContext,
    priors: Mapping[str, float],
    terms: frozenset[str],
    max_centrality: float,
    weights: Mapping[str, float],
) -> float:
    computed = features(
        entry,
        context,
        prior=priors.get(entry.hit.chunk.digest, 0.0),
        terms=terms,
        max_centrality=max_centrality,
    )
    return sum(weights.get(name, 0.0) * value for name, value in computed.items())


def _symbol_match(symbol: str, query: str, terms: frozenset[str]) -> float:
    """1.0 when the symbol *is* the query, else the share of it the query covers."""
    if not symbol:
        return 0.0
    if symbol.lower() == query.strip().lower():
        return 1.0

    parts = frozenset(tokenize(symbol.replace(".", " ")))
    if not parts:
        return 0.0
    return len(parts & terms) / len(parts)


def _agreement(leg_count: int) -> float:
    """One leg says nothing; two is corroboration; three or more is consensus."""
    if leg_count <= 1:
        return 0.0
    return 0.5 if leg_count == 2 else 1.0


def _rarity(index: LexicalIndex, symbol: str) -> float:
    """``1 / (1 + ln(1 + sites))`` over the symbol's own token."""
    if not symbol:
        return 0.0
    bare = symbol.rsplit(".", 1)[-1].lower()
    sites = index.document_frequency.get(bare, 0)
    return 1.0 / (1.0 + log(1.0 + sites))
