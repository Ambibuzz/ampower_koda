"""Adjusting the neutral ranking before the map is rendered."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def demote_mirrors(
    scores: Mapping[str, float],
    mirror_roots: Iterable[str],
    factor: float,
) -> dict[str, float]:
    """Scale down files living under a vendored root."""
    roots = set(mirror_roots)
    if not roots:
        return dict(scores)
    return {
        path: (score * factor if path.split("/", 1)[0] in roots else score)
        for path, score in scores.items()
    }
