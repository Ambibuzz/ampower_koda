"""Deciding which files the index is allowed to hold."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..config.schema import CoreConfig
from ..constants import EXCLUDED_DIRECTORIES
from ..contracts.analysis import SkippedFile
from ..contracts.source import SourceFile
from ..errors import WorkspaceError
from ..identity import source_hash
from .ports import Workspace
from .redaction import compile_redaction


@dataclass(frozen=True, slots=True)
class Discovery:
    """The outcome of one discovery pass."""

    candidates: tuple[SourceFile, ...]
    skipped: tuple[SkippedFile, ...]

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(source.path for source in self.candidates)


def excluded_directories(config: CoreConfig) -> frozenset[str]:
    """Directory names pruned at any depth."""
    return EXCLUDED_DIRECTORIES | frozenset(config.indexing.extra_excluded_directories)


def is_excluded_path(path: str, excluded: frozenset[str]) -> bool:
    """True when any segment of ``path`` names an excluded directory."""
    return any(segment in excluded for segment in path.split("/")[:-1])


def decode_source(content: bytes) -> str | None:
    """Decode file bytes as UTF-8, or return ``None`` if this is not text."""
    if b"\x00" in content:
        return None
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def discover(workspace: Workspace, config: CoreConfig) -> Discovery:
    """Enumerate the workspace and split it into candidates and skips."""
    excluded = excluded_directories(config)
    is_redacted = compile_redaction(config.security.redact_globs)
    max_bytes = config.indexing.max_file_bytes

    candidates: list[SourceFile] = []
    skipped: list[SkippedFile] = []

    for path in sorted(_normalise(workspace.list_files())):
        if is_excluded_path(path, excluded):
            continue

        pattern = is_redacted(path)
        if pattern is not None:
            skipped.append(SkippedFile(path, "redacted", f"matched {pattern}"))
            continue

        stat = workspace.stat(path)
        if stat is None:
            continue

        if stat.size > max_bytes:
            skipped.append(
                SkippedFile(path, "too_large", f"{stat.size} bytes exceeds {max_bytes}")
            )
            continue

        try:
            content = workspace.read_bytes(path)
        except WorkspaceError as exc:
            skipped.append(SkippedFile(path, "unreadable", exc.reason))
            continue

        text = decode_source(content)
        if text is None:
            skipped.append(SkippedFile(path, "binary", "not valid UTF-8 text"))
            continue

        candidates.append(
            SourceFile(
                path=path,
                text=text,
                source_hash=source_hash(content),
                stat=stat,
            )
        )

    return Discovery(candidates=tuple(candidates), skipped=tuple(skipped))


def _normalise(paths: Iterable[str]) -> Iterable[str]:
    """Force relative POSIX form, so Windows and Linux produce the same index."""
    for path in paths:
        cleaned = path.replace("\\", "/").lstrip("/")
        if cleaned.startswith("./"):
            cleaned = cleaned[2:]
        if cleaned:
            yield cleaned
