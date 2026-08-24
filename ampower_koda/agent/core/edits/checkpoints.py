"""Content-addressed pre-edit bytes, and the two ways they are captured."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from ..errors import CoreError
from ..workspace.ports import Workspace

ABSENT = ""


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One restore point: which files, at which content."""

    patch_id: str
    files: Mapping[str, str] = field(default_factory=dict)
    sequence: int = 0
    """Monotone within a session. Revert resolves "the earliest checkpoint later
    than this point" and needs a total order to do it; comparing patch ids would
    make that order depend on how ids happen to sort."""

    def __post_init__(self) -> None:
        if not self.patch_id:
            raise CoreError("a checkpoint must name its patch")
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.files))

    def sha_for(self, path: str) -> str | None:
        return self.files.get(path)


@dataclass(frozen=True, slots=True)
class Store:
    """Checkpoints in order, plus the content-addressed blobs they name."""

    checkpoints: tuple[Checkpoint, ...] = ()
    blobs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blobs", MappingProxyType(dict(self.blobs)))

    def content(self, sha: str) -> str:
        """The bytes behind a sha, or ``""``. See :meth:`holds` before trusting it."""
        return self.blobs.get(sha, "") if sha else ""

    def holds(self, sha: str) -> bool:
        """Whether this store actually has the bytes behind ``sha``."""
        return sha == ABSENT or sha in self.blobs

    def for_patch(self, patch_id: str) -> Checkpoint | None:
        return next((point for point in self.checkpoints if point.patch_id == patch_id), None)

    def after(self, sequence: int) -> tuple[Checkpoint, ...]:
        """Checkpoints later than ``sequence``, **in sequence order**."""
        return tuple(
            sorted(
                (point for point in self.checkpoints if point.sequence > sequence),
                key=lambda point: point.sequence,
            )
        )


def blob_id(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def snapshot(
    store: Store,
    patch_id: str,
    paths: Iterable[str],
    workspace: Workspace,
) -> Store:
    """Capture a known path list up front, before an attempt runs."""
    return _capture(store, patch_id, dict.fromkeys(paths), workspace)


def capture_before(store: Store, patch_id: str, path: str, workspace: Workspace) -> Store:
    """Capture one file's pre-edit bytes, lazily, on the way to writing it."""
    return _capture(store, patch_id, {path: None}, workspace)


def _capture(
    store: Store,
    patch_id: str,
    paths: Mapping[str, None],
    workspace: Workspace,
) -> Store:
    """Record pre-edit bytes for ``paths``, never overwriting what is recorded."""
    existing = store.for_patch(patch_id)
    files = dict(existing.files) if existing else {}
    blobs = dict(store.blobs)

    for path in paths:
        if path in files:
            continue
        content = _read(workspace, path)
        if content is None:
            files[path] = ABSENT
            continue
        sha = blob_id(content)
        blobs[sha] = content
        files[path] = sha

    point = (
        replace(existing, files=files)
        if existing
        else Checkpoint(patch_id=patch_id, files=files, sequence=len(store.checkpoints) + 1)
    )
    others = tuple(p for p in store.checkpoints if p.patch_id != patch_id)
    ordered = tuple(sorted((*others, point), key=lambda p: p.sequence))
    return Store(checkpoints=ordered, blobs=blobs)


def _read(workspace: Workspace, path: str) -> str | None:
    try:
        return workspace.read_bytes(path).decode("utf-8", errors="replace")
    except (CoreError, OSError):
        return None
