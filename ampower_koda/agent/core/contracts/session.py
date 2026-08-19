"""The world, built once per session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..config.schema import CoreConfig
from .repo_map import RepoMap
from .repository import RepositoryIndex


@dataclass(frozen=True, slots=True)
class RepoMemory:
    """The hand-written instructions a repository carries for its agent."""

    text: str = ""
    sources: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class CoChangeMemory:
    """Which files change together, weighted by recency."""

    neighbours: Mapping[str, tuple[tuple[str, float], ...]] = field(default_factory=dict)
    """Path → ``((neighbour, weight), …)``, strongest first."""

    commits_read: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "neighbours", MappingProxyType(dict(self.neighbours)))

    def for_file(self, path: str, limit: int | None = None) -> tuple[tuple[str, float], ...]:
        entries = self.neighbours.get(path, ())
        return entries if limit is None else entries[:limit]


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Everything cold start produces. Built once; replaced, never mutated."""

    root: str
    config: CoreConfig
    index: RepositoryIndex
    memory: RepoMemory = field(default_factory=RepoMemory)
    cochange: CoChangeMemory = field(default_factory=CoChangeMemory)

    repo_map: RepoMap = field(default_factory=RepoMap)
    """Frozen for the session, and first in the cached prefix. Held here rather
    than rebuilt per turn because rewriting it invalidates every byte
    downstream of it — see
    :func:`~ampower_koda.agent.core.repomap.build.repersonalize_once` for the
    single rewrite that is permitted."""

    overlaid: tuple[str, ...] = ()
    """Paths whose analysis came from an overlay rather than from disk. Kept so
    a later stage can tell the developer *why* the index disagrees with their
    working tree, instead of leaving them to discover it."""

    @property
    def file_count(self) -> int:
        return len(self.index)

    @property
    def skipped_count(self) -> int:
        return len(self.index.skipped)
