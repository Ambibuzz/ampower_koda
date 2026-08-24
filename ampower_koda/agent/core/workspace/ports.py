"""The boundary between the core and the machine."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from ..contracts.source import FileStat


@runtime_checkable
class Workspace(Protocol):
    """Read-only access to a repository, plus a scratch space for the cache."""

    @property
    def root(self) -> str:
        """Absolute path of the workspace root, for display and for git."""

    def list_files(self) -> Iterable[str]:
        """Yield every candidate file as a workspace-relative, POSIX-style path."""

    def read_bytes(self, path: str) -> bytes:
        """Return a file's raw bytes."""

    def stat(self, path: str) -> FileStat | None:
        """Return size and mtime, or ``None`` if the file is gone."""

    def read_cache(self, key: str) -> bytes | None:
        """Return a cache entry, or ``None`` for a miss or an unreadable entry."""

    def write_cache(self, key: str, payload: bytes) -> None:
        """Store a cache entry. Failures are swallowed — the cache is optional."""

    def run_git(self, args: Sequence[str]) -> str | None:
        """Run a read-only ``git`` command and return stdout."""


@runtime_checkable
class Clock(Protocol):
    """Current time as a POSIX timestamp."""

    def now(self) -> float: ...
