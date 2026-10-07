"""The expansion legs: related, structure, graph, and history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log

from ..constants import (
    GRAPH_DEPTH,
    GRAPH_EXPAND_LIMIT,
    GRAPH_HOP_DECAY,
    GRAPH_KEEP,
    HISTORY_NEIGHBOURS,
    STRUCTURAL_DEFINITION_WEIGHT,
    STRUCTURAL_LIMIT,
    STRUCTURAL_MAX_SEEDS,
    STRUCTURAL_PER_SYMBOL_CAP,
    STRUCTURAL_REFERENCE_WEIGHT,
)
from ..contracts.chunks import Chunk
from ..contracts.repository import RepositoryIndex
from ..contracts.retrieval import Hit, LegResult
from ..contracts.session import CoChangeMemory
from ..graph.edges import CodeGraph
from .excerpts import best_chunks


@dataclass(frozen=True, slots=True)
class Seed:
    """One lexical hit, as the expansion legs see it."""

    path: str
    symbol: str
    rank: int

    @property
    def weight(self) -> float:
        """``1 / (rank + 1)``. The first seed is worth twice the second."""
        return 1.0 / (self.rank + 1)


def seeds_from(hits: Sequence[Hit], limit: int = STRUCTURAL_MAX_SEEDS) -> tuple[Seed, ...]:
    """Represent distinct files first, then additional symbols if space remains."""
    seen: set[tuple[str, str]] = set()
    seeds: list[Seed] = []
    first: dict[str, Hit] = {}
    for hit in hits:
        first.setdefault(hit.path, hit)
    for hit in (*first.values(), *hits):
        key = (hit.path, hit.symbol)
        if key in seen:
            continue
        seen.add(key)
        seeds.append(Seed(path=hit.path, symbol=hit.symbol, rank=len(seeds)))
        if len(seeds) >= limit:
            break
    return tuple(seeds)


def related_leg(seeds: Sequence[Seed], index: RepositoryIndex, graph: CodeGraph,
                *, query: str = "", limit: int = 24) -> LegResult:
    """Reserve candidate opportunities for explicit feature/test relationships."""
    scored: dict[str, tuple[float, Chunk]] = {}
    kinds = {"feature", "rpc", "registration", "route", "test"}
    for seed in seeds:
        for edge in (*graph.out_edges(seed.path), *graph.in_edges(seed.path)):
            if edge.kind not in kinds:
                continue
            target = edge.target if edge.source == seed.path else edge.source
            for rank, chunk in enumerate(best_chunks(index, target, query, symbol=edge.symbol)):
                _keep(scored, chunk, seed.weight * edge.weight / (rank + 1))
    return LegResult(leg="related", hits=_rank(scored, limit))


def structural_leg(
    seeds: Sequence[Seed],
    index: RepositoryIndex,
    graph: CodeGraph,
    *,
    limit: int = STRUCTURAL_LIMIT,
) -> LegResult:
    """Definitions and references of the symbols in the seeds."""
    scored: dict[str, tuple[float, Chunk]] = {}
    notes: list[str] = []

    for seed in seeds:
        if not seed.symbol:
            continue
        bare = seed.symbol.rsplit(".", 1)[-1]
        definitions = graph.defines(seed.symbol) or graph.defines(bare)
        if not definitions:
            continue

        rarity = 1.0 / (1.0 + log(1.0 + len(definitions)))
        if len(definitions) > STRUCTURAL_PER_SYMBOL_CAP:
            notes.append(
                f"{bare} has {len(definitions)} definitions; "
                f"showing {STRUCTURAL_PER_SYMBOL_CAP} — read the rest by name"
            )

        for path in definitions[:STRUCTURAL_PER_SYMBOL_CAP]:
            for chunk in _chunks_for(index, path, seed.symbol, bare):
                _keep(scored, chunk, seed.weight * STRUCTURAL_DEFINITION_WEIGHT * rarity)

        for path in _referencing(index, bare)[:STRUCTURAL_PER_SYMBOL_CAP]:
            for chunk in _chunks_for(index, path, seed.symbol, bare):
                _keep(scored, chunk, seed.weight * STRUCTURAL_REFERENCE_WEIGHT * rarity)

    return LegResult(leg="structure", hits=_rank(scored, limit), notes=tuple(notes))


def _chunks_for(index: RepositoryIndex, path: str, symbol: str, bare: str) -> tuple[Chunk, ...]:
    """The chunks in ``path`` that carry this symbol, or the file's first chunk."""
    analysis = index.files.get(path)
    if analysis is None:
        return ()
    matching = tuple(
        chunk
        for chunk in analysis.chunks
        if chunk.identity in (symbol, bare) or chunk.identity.endswith(f".{bare}")
    )
    return matching or best_chunks(index, path, bare, symbol=bare)


def _referencing(index: RepositoryIndex, name: str) -> tuple[str, ...]:
    return tuple(
        path
        for path in index.paths
        if any(reference.name == name for reference in index.files[path].references)
    )


def graph_leg(
    seeds: Sequence[Seed],
    index: RepositoryIndex,
    graph: CodeGraph,
    *,
    depth: int = GRAPH_DEPTH,
    keep: int = GRAPH_KEEP,
    query: str = "",
) -> LegResult:
    """A weighted walk over typed edges, both directions, from the seed files."""
    best: dict[str, float] = {}
    frontier: dict[str, float] = {}
    for seed in seeds:
        frontier[seed.path] = max(frontier.get(seed.path, 0.0), seed.weight)
    best.update(frontier)
    symbols: dict[str, str] = {}

    for hop in range(depth):
        decay = 1.0 if hop == 0 else GRAPH_HOP_DECAY
        following: dict[str, float] = {}

        for path in sorted(frontier):
            parent = frontier[path]
            for edge in (*graph.out_edges(path), *graph.in_edges(path)):
                neighbour = edge.target if edge.source == path else edge.source
                if neighbour == path:
                    continue
                score = parent * edge.weight * decay
                if score > best.get(neighbour, 0.0):
                    best[neighbour] = score
                    following[neighbour] = score
                    symbols[neighbour] = edge.symbol

        widest = sorted(following.items(), key=lambda item: (-item[1], item[0]))
        frontier = dict(widest[:GRAPH_EXPAND_LIMIT])
        if not frontier:
            break

    seeded = {seed.path for seed in seeds}
    scored: dict[str, tuple[float, Chunk]] = {}
    for path, value in best.items():
        if path in seeded:
            continue
        for rank, chunk in enumerate(best_chunks(index, path, query, symbol=symbols.get(path, ""))):
            _keep(scored, chunk, value / (rank + 1))

    return LegResult(leg="graph", hits=_rank(scored, keep))


def history_leg(
    seeds: Sequence[Seed],
    index: RepositoryIndex,
    cochange: CoChangeMemory,
    *,
    limit: int = HISTORY_NEIGHBOURS,
    query: str = "",
) -> LegResult:
    """Files that this repository's own commits say belong with the seeds."""
    if not cochange.neighbours:
        return LegResult(leg="history")

    seeded = {seed.path for seed in seeds}
    scored: dict[str, tuple[float, Chunk]] = {}

    for seed in seeds:
        for neighbour, weight in cochange.for_file(seed.path, limit=limit):
            if neighbour in seeded:
                continue
            for rank, chunk in enumerate(best_chunks(index, neighbour, query)):
                _keep(scored, chunk, seed.weight * weight / (rank + 1))

    return LegResult(leg="history", hits=_rank(scored, limit))


def _keep(scored: dict[str, tuple[float, Chunk]], chunk: Chunk, value: float) -> None:
    """Record a chunk at its best score. One entry per digest."""
    existing = scored.get(chunk.digest)
    if existing is None or value > existing[0]:
        scored[chunk.digest] = (value, chunk)


def _rank(scored: Mapping[str, tuple[float, Chunk]], limit: int) -> tuple[Hit, ...]:
    """Best first, ties broken by location so the order is total."""
    ordered = sorted(scored.values(), key=lambda item: (-item[0], item[1].location))
    return tuple(Hit(chunk=chunk, score=value) for value, chunk in ordered[:limit])
