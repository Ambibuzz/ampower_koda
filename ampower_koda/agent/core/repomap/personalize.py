"""Deciding what this session is about, before the model has said anything."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from ..constants import (
    MAP_BOOST_BASENAME,
    MAP_BOOST_DIRECTORY,
    MAP_BOOST_ESTABLISHED,
    MAP_BOOST_PARTIAL,
)

_MIN_TOKEN = 3

_SPLIT = re.compile(r"[^0-9A-Za-z]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def query_tokens(query: str) -> tuple[str, ...]:
    """Lowercase word-ish tokens from a query, camelCase split, order preserved."""
    seen: dict[str, None] = {}
    for raw in _SPLIT.split(query):
        for part in _CAMEL.split(raw):
            token = part.lower()
            if len(token) >= _MIN_TOKEN:
                seen.setdefault(token, None)
    return tuple(seen)


def personalization(
    paths: Iterable[str],
    *,
    query: str = "",
    established: Sequence[str] = (),
) -> Mapping[str, float]:
    """Relative interest per file. Sparse: only boosted files appear."""
    tokens = query_tokens(query)
    established_paths = set(established)
    boosts: dict[str, float] = {}

    for path in paths:
        weight = 1.0
        if path in established_paths:
            weight *= MAP_BOOST_ESTABLISHED

        basename = path.rsplit("/", 1)[-1]
        stem = basename.rsplit(".", 1)[0].lower()
        parts = set(query_tokens(stem))
        directories = {segment.lower() for segment in path.split("/")[:-1]}

        if tokens:
            if stem in tokens:
                weight *= MAP_BOOST_BASENAME
            elif parts & set(tokens):
                weight *= MAP_BOOST_PARTIAL

            if directories & set(tokens):
                weight *= MAP_BOOST_DIRECTORY

        if weight > 1.0:
            boosts[path] = weight

    return boosts


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
