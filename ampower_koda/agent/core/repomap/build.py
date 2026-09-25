"""Building the map: rank, demote, render."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import MAP_MAX_TOKENS, MIRROR_RANK_FACTOR
from ..contracts.repo_map import FileRanks, MirrorSet, RepoMap
from ..contracts.repository import RepositoryIndex
from ..graph.edges import CodeGraph, build_graph
from ..graph.mirrors import detect_mirrors
from ..graph.pagerank import pagerank
from .personalize import demote_mirrors
from .render import render_repo_map


@dataclass(frozen=True, slots=True)
class MapBuild:
    """A rendered map plus the ranking machinery behind it."""

    map: RepoMap
    graph: CodeGraph
    mirrors: MirrorSet


def build_map(
    index: RepositoryIndex,
    *,
    max_tokens: int = MAP_MAX_TOKENS,
) -> MapBuild:
    """Rank the repository and render its map."""
    graph = build_graph(index)
    mirrors = detect_mirrors(index.paths)

    neutral = pagerank(graph, nodes=index.paths)
    demoted = FileRanks(
        scores=demote_mirrors(neutral.scores, mirrors.roots, MIRROR_RANK_FACTOR),
    )

    return MapBuild(
        map=render_repo_map(index, demoted, max_tokens=max_tokens),
        graph=graph,
        mirrors=mirrors,
    )


