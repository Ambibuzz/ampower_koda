"""Personalized PageRank, run for determinism rather than for convergence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..constants import PAGERANK_ALPHA, PAGERANK_ITERATIONS
from ..contracts.repo_map import FileRanks
from .edges import CodeGraph


def pagerank(
    graph: CodeGraph,
    *,
    nodes: Sequence[str] | None = None,
    personalization: Mapping[str, float] | None = None,
    alpha: float = PAGERANK_ALPHA,
    iterations: int = PAGERANK_ITERATIONS,
) -> FileRanks:
    """Rank ``nodes`` over ``graph``."""
    ordered = tuple(sorted(nodes)) if nodes is not None else graph.nodes
    if not ordered:
        return FileRanks(personalized=False, iterations=0)

    teleport = _normalise(ordered, personalization)
    adjacency = _adjacency(graph, ordered)

    count = len(ordered)
    scores = dict.fromkeys(ordered, 1.0 / count)

    for _ in range(iterations):
        incoming = dict.fromkeys(ordered, 0.0)

        for node in ordered:
            targets = adjacency[node]
            if not targets:
                incoming[node] += scores[node]
                continue
            share = scores[node]
            for target, weight in targets:
                incoming[target] += share * weight

        scores = _rescale(
            {
                node: alpha * incoming[node] + (1.0 - alpha) * teleport[node]
                for node in ordered
            }
        )

    return FileRanks(
        scores=scores,
        personalized=any(
            personalization.get(node, 0.0) > 0 for node in ordered
        )
        if personalization
        else False,
        iterations=iterations,
    )


def _adjacency(graph: CodeGraph, nodes: Sequence[str]) -> dict[str, tuple[tuple[str, float], ...]]:
    """Out-edges as ``(target, normalised_weight)``, restricted to ``nodes``."""
    present = set(nodes)
    adjacency: dict[str, tuple[tuple[str, float], ...]] = {}

    for node in nodes:
        totals: dict[str, float] = {}
        for edge in graph.out_edges(node):
            if edge.target in present and edge.target != node:
                totals[edge.target] = totals.get(edge.target, 0.0) + edge.weight

        total = sum(totals.values())
        adjacency[node] = (
            tuple((target, totals[target] / total) for target in sorted(totals))
            if total > 0.0
            else ()
        )

    return adjacency


def _normalise(
    nodes: Sequence[str],
    personalization: Mapping[str, float] | None,
) -> dict[str, float]:
    """Turn relative interest into a probability vector over ``nodes``."""
    if not personalization:
        return dict.fromkeys(nodes, 1.0 / len(nodes))

    raw = {node: max(0.0, personalization.get(node, 0.0)) for node in nodes}
    total = sum(raw.values())
    if total <= 0.0:
        return dict.fromkeys(nodes, 1.0 / len(nodes))
    return {node: value / total for node, value in raw.items()}


def _rescale(scores: dict[str, float]) -> dict[str, float]:
    """Renormalise to sum 1."""
    total = sum(scores.values())
    if total <= 0.0:
        return dict.fromkeys(scores, 1.0 / len(scores))
    return {node: value / total for node, value in scores.items()}
