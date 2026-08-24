"""Keeping the index current after cold start."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from ..config.schema import CoreConfig
from ..contracts.analysis import SkippedFile
from ..contracts.repository import RepositoryIndex, with_file, with_skip, without_file
from ..contracts.session import SessionContext
from ..contracts.source import Overlay, SourceFile
from ..errors import ParseError
from ..identity import source_hash
from ..workspace.discovery import decode_source
from ..workspace.ports import Workspace
from ..workspace.redaction import compile_redaction
from .analysis import analyze
from .parsers.registry import ParserRegistry, default_registry


@dataclass(frozen=True, slots=True)
class Revisions:
    """Per-file revision counters. Immutable; every change returns a new table."""

    counters: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "counters", MappingProxyType(dict(self.counters)))

    def current(self, path: str) -> int:
        return self.counters.get(path, 0)

    def observe(self, path: str) -> tuple[Revisions, int]:
        """Record that ``path`` changed. Returns the new table and the revision."""
        revision = self.current(path) + 1
        counters = dict(self.counters)
        counters[path] = revision
        return Revisions(counters=counters), revision


def commit_if_current(
    index: RepositoryIndex,
    revisions: Revisions,
    path: str,
    revision: int,
    analysis_or_skip: object,
) -> RepositoryIndex:
    """Apply a result, unless it has already been superseded."""
    if revisions.current(path) != revision:
        return index
    if isinstance(analysis_or_skip, SkippedFile):
        return with_skip(index, analysis_or_skip)
    return with_file(index, analysis_or_skip)  # type: ignore[arg-type]


def reanalyse(
    workspace: Workspace,
    config: CoreConfig,
    path: str,
    *,
    registry: ParserRegistry | None = None,
) -> object:
    """Re-read and re-analyse one file from disk."""
    registry = registry or default_registry()

    pattern = compile_redaction(config.security.redact_globs)(path)
    if pattern is not None:
        return SkippedFile(path, "redacted", f"matched {pattern}")

    stat = workspace.stat(path)
    if stat is None:
        return None
    if stat.size > config.indexing.max_file_bytes:
        return SkippedFile(path, "too_large", f"{stat.size} bytes")

    content = workspace.read_bytes(path)
    text = decode_source(content)
    if text is None:
        return SkippedFile(path, "binary", "not valid UTF-8 text")

    source = SourceFile(path=path, text=text, source_hash=source_hash(content), stat=stat)
    try:
        return analyze(source, registry)
    except ParseError as exc:
        return SkippedFile(path, "parse_failed", exc.reason)


def forget(index: RepositoryIndex, path: str) -> RepositoryIndex:
    """Drop a deleted file from the index."""
    return without_file(index, path)


def apply_overlays(
    context: SessionContext,
    overlays: tuple[Overlay, ...],
    *,
    registry: ParserRegistry | None = None,
) -> SessionContext:
    """Replay in-memory content over the index, returning a new context."""
    registry = registry or default_registry()
    if not overlays:
        return context

    index = context.index
    applied: list[str] = []

    for overlay in _resolve_collisions(overlays):
        source = SourceFile(
            path=overlay.path,
            text=overlay.text,
            source_hash=source_hash(overlay.text.encode("utf-8")),
            stat=None,
        )
        try:
            index = with_file(index, analyze(source, registry))
        except ParseError:
            continue
        applied.append(overlay.path)

    return replace(
        context,
        index=index,
        overlaid=tuple(sorted({*context.overlaid, *applied})),
    )


def _resolve_collisions(overlays: tuple[Overlay, ...]) -> tuple[Overlay, ...]:
    """One overlay per path, editor buffers winning, in path order."""
    chosen: dict[str, Overlay] = {}
    for overlay in overlays:
        existing = chosen.get(overlay.path)
        if existing is None or (existing.origin != "buffer" and overlay.origin == "buffer"):
            chosen[overlay.path] = overlay
    return tuple(chosen[path] for path in sorted(chosen))
