"""Two implementations of the :class:`~.ports.Workspace` port."""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..constants import CACHE_DIRECTORY, EXCLUDED_DIRECTORIES
from ..contracts.source import FileStat
from ..errors import WorkspaceError

_ALLOWED_GIT_SUBCOMMANDS = frozenset({"log"})

_GIT_TIMEOUT_SECONDS = 20


@dataclass(frozen=True, slots=True)
class LocalWorkspace:
    """A repository checkout on the local filesystem."""

    root_path: Path
    cache_directory: str = CACHE_DIRECTORY

    @property
    def root(self) -> str:
        return str(self.root_path)

    def list_files(self) -> Iterable[str]:
        """Walk the tree, pruning excluded directories as it descends."""
        root = self.root_path
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(name for name in dirnames if name not in EXCLUDED_DIRECTORIES)
            base = Path(dirpath)
            for filename in sorted(filenames):
                yield (base / filename).relative_to(root).as_posix()

    def read_bytes(self, path: str) -> bytes:
        target = self._resolve(path)
        try:
            return target.read_bytes()
        except OSError as exc:
            raise WorkspaceError(path, str(exc)) from exc

    def stat(self, path: str) -> FileStat | None:
        try:
            info = self._resolve(path).stat()
        except (OSError, WorkspaceError):
            return None
        return FileStat(size=info.st_size, mtime_ns=info.st_mtime_ns)

    def read_cache(self, key: str) -> bytes | None:
        try:
            return (self.root_path / self.cache_directory / f"{key}.json").read_bytes()
        except OSError:
            return None

    def write_cache(self, key: str, payload: bytes) -> None:
        """Write a cache entry atomically, or give up quietly."""
        directory = self.root_path / self.cache_directory
        target = directory / f"{key}.json"
        temporary = directory / f"{key}.{os.getpid()}.tmp"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(payload)
            os.replace(temporary, target)
        except OSError:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)

    def run_git(self, args: Sequence[str]) -> str | None:
        """Run a read-only git command, or return ``None`` if it cannot be run."""
        if not args or args[0] not in _ALLOWED_GIT_SUBCOMMANDS:
            raise WorkspaceError(self.root, f"refusing git command {list(args)}")
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.root_path), *args],
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout if completed.returncode == 0 else None

    def _resolve(self, path: str) -> Path:
        """Resolve a relative path, refusing anything that escapes the root."""
        candidate = (self.root_path / path).resolve()
        root = self.root_path.resolve()
        if candidate != root and root not in candidate.parents:
            raise WorkspaceError(path, "resolves outside the workspace root")
        return candidate


@dataclass(frozen=True, slots=True)
class MemoryWorkspace:
    """An in-memory workspace: a path → bytes mapping and nothing else."""

    files: Mapping[str, bytes]
    root_label: str = "/memory"
    git_output: Mapping[str, str] = field(default_factory=dict)
    """Maps a joined git argument list to canned stdout, so co-change parsing
    can be exercised without a repository."""

    cache: dict[str, bytes] = field(default_factory=dict)
    now_ns: int = 0

    @property
    def root(self) -> str:
        return self.root_label

    def list_files(self) -> Iterable[str]:
        return sorted(self.files)

    def read_bytes(self, path: str) -> bytes:
        try:
            return self.files[path]
        except KeyError as exc:
            raise WorkspaceError(path, "not found") from exc

    def stat(self, path: str) -> FileStat | None:
        content = self.files.get(path)
        if content is None:
            return None
        return FileStat(size=len(content), mtime_ns=self.now_ns)

    def read_cache(self, key: str) -> bytes | None:
        return self.cache.get(key)

    def write_cache(self, key: str, payload: bytes) -> None:
        self.cache[key] = payload

    def run_git(self, args: Sequence[str]) -> str | None:
        return self.git_output.get(" ".join(args))


@dataclass(frozen=True, slots=True)
class SystemClock:
    """The real clock. Injected, never reached for directly."""

    def now(self) -> float:
        return time.time()


@dataclass(frozen=True, slots=True)
class FixedClock:
    """A clock that does not move, so decay curves are testable."""

    timestamp: float

    def now(self) -> float:
        return self.timestamp


def text_workspace(files: Mapping[str, str], **kwargs: object) -> MemoryWorkspace:
    """Build a :class:`MemoryWorkspace` from text, so fixtures read as source."""
    return MemoryWorkspace(
        files={path: text.encode("utf-8") for path, text in files.items()},
        **kwargs,  # type: ignore[arg-type]
    )
