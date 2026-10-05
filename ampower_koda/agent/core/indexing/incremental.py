"""Keeping the index current after cold start."""

from __future__ import annotations

from dataclasses import replace

from ..contracts.repository import with_file
from ..contracts.session import SessionContext
from ..contracts.source import Overlay, SourceFile
from ..errors import ParseError
from ..identity import source_hash
from .analysis import analyze
from .parsers.registry import ParserRegistry, default_registry


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
        try:
            # A lone surrogate from an editor buffer cannot be encoded: skipped like a parse failure.
            source = SourceFile(
                path=overlay.path,
                text=overlay.text,
                source_hash=source_hash(overlay.text.encode("utf-8")),
                stat=None,
            )
            index = with_file(index, analyze(source, registry))
        except (ParseError, UnicodeEncodeError):
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
