"""The repo map, and the ranking that produces it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class FileRanks:
    """PageRank over the reference graph, one score per file."""

    scores: Mapping[str, float] = field(default_factory=dict)
    personalized: bool = False
    iterations: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    def of(self, path: str) -> float:
        return self.scores.get(path, 0.0)

    def ordered(self) -> tuple[str, ...]:
        """Paths by descending rank, ties broken by path."""
        return tuple(sorted(self.scores, key=lambda path: (-self.scores[path], path)))


@dataclass(frozen=True, slots=True)
class RepoMap:
    """A rendered, budgeted view of the repository's definitions."""

    text: str = ""
    files_shown: int = 0
    files_total: int = 0
    tokens: int = 0

    degraded: bool = False
    """True when the workspace held no parseable definitions and the map fell
    back to a directory tree. Recorded rather than hidden: a model shown a
    directory listing needs to know it is not looking at a symbol map, and the
    rendered text says so too."""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def truncated(self) -> bool:
        return self.files_shown < self.files_total


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
