"""Vendored copies of a tree, detected once at cold start."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MirrorSet:
    """Directory roots that hold a second copy of another tree."""

    roots: frozenset[str] = frozenset()

    def contains(self, path: str) -> bool:
        root = path.split("/", 1)[0]
        return root in self.roots

    @property
    def is_empty(self) -> bool:
        return not self.roots
