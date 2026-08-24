"""Finding the second copy of a tree."""

from __future__ import annotations

from collections.abc import Iterable

from ..constants import (
    MIRROR_MIN_SHARED_FILES,
    MIRROR_MIN_SHARED_FRACTION,
    VENDOR_DIRECTORIES,
)
from ..contracts.repo_map import MirrorSet


def detect_mirrors(paths: Iterable[str]) -> MirrorSet:
    """Return the top-level directories that hold a copy of another tree."""
    by_root: dict[str, set[str]] = {}
    for path in paths:
        root, separator, tail = path.partition("/")
        if separator and tail:
            by_root.setdefault(root, set()).add(tail)

    if len(by_root) < 2:
        return MirrorSet()

    mirrors: set[str] = set()
    for root, tails in by_root.items():
        if root not in VENDOR_DIRECTORIES:
            continue
        elsewhere = {tail for other, others in by_root.items() if other != root for tail in others}
        shared = tails & elsewhere
        if len(shared) < MIRROR_MIN_SHARED_FILES:
            continue
        if len(shared) / len(tails) < MIRROR_MIN_SHARED_FRACTION:
            continue
        mirrors.add(root)

    return MirrorSet(roots=frozenset(mirrors))
