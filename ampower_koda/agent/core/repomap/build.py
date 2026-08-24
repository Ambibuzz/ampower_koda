"""Building the map: rank, demote, render — and the one rewrite it is allowed."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..constants import MAP_MAX_TOKENS, MIRROR_RANK_FACTOR
from ..contracts.repo_map import FileRanks, MirrorSet, RepoMap
from ..contracts.repository import RepositoryIndex
from ..graph.edges import CodeGraph, build_graph
from ..graph.mirrors import detect_mirrors
from ..graph.pagerank import pagerank
from .personalize import demote_mirrors, personalization
from .render import render_repo_map


@dataclass(frozen=True, slots=True)
class MapBuild:
    """A rendered map plus the ranking machinery behind it."""

    map: RepoMap
    graph: CodeGraph
    ranks: FileRanks
    """Unpersonalized. The reranker's ``centrality`` feature must be a property
    of the repository, not of the question — a personalized rank would let a
    query raise the centrality of the files it already matched."""

    mirrors: MirrorSet
    personalized: bool = False

    rewritten: bool = False
    """Whether the one permitted re-personalization has been spent.

    Separate from ``personalized`` because they answer different questions.
    ``personalized`` says the ranking carries a bias; ``rewritten`` says the
    session has already paid for the rewrite. Gating the latch on the first one
    means a re-rank that finds no boosts leaves it unset, and the map is then
    rebuilt on every turn forever — one cache write per turn, which is the exact
    cost the latch exists to prevent.
    """


def build_map(
    index: RepositoryIndex,
    *,
    query: str = "",
    established: Sequence[str] = (),
    max_tokens: int = MAP_MAX_TOKENS,
    graph: CodeGraph | None = None,
) -> MapBuild:
    """Rank the repository and render its map."""
    graph = graph if graph is not None else build_graph(index)
    mirrors = detect_mirrors(index.paths)

    neutral = pagerank(graph, nodes=index.paths)
    boosts = personalization(index.paths, query=query, established=established)
    biased = pagerank(graph, nodes=index.paths, personalization=boosts) if boosts else neutral

    demoted = FileRanks(
        scores=demote_mirrors(biased.scores, mirrors.roots, MIRROR_RANK_FACTOR),
        personalized=biased.personalized,
        iterations=biased.iterations,
    )

    return MapBuild(
        map=render_repo_map(index, demoted, max_tokens=max_tokens),
        graph=graph,
        ranks=neutral,
        mirrors=mirrors,
        personalized=bool(boosts),
    )


def repersonalize_once(
    build: MapBuild,
    index: RepositoryIndex,
    *,
    query: str = "",
    established: Sequence[str] = (),
    max_tokens: int = MAP_MAX_TOKENS,
) -> MapBuild:
    """Re-rank the map once, now that the session knows what it is about."""
    if build.rewritten or build.personalized or not (query or established):
        return build

    return replace(
        build_map(
            index,
            query=query,
            established=established,
            max_tokens=max_tokens,
            graph=build.graph,
        ),
        rewritten=True,
    )


def with_map(build: MapBuild, rendered: RepoMap) -> MapBuild:
    """Replace only the rendered map, keeping the ranking machinery."""
    return replace(build, map=rendered)
