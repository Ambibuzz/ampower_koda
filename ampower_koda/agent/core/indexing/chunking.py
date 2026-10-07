"""Cutting a file into retrievable pieces."""

from __future__ import annotations

from collections.abc import Sequence

from ..constants import (
    CHUNK_CHARS,
    CHUNK_LINES,
    CHUNK_LONG_LINE_STRIDE,
    CHUNK_OVERLAP_LINES,
    CHUNKLESS_ROLES,
    COMMENT_MAX_LINES,
)
from ..contracts.chunks import Chunk, is_indexable_role
from ..contracts.source import Span, split_lines
from ..contracts.symbols import Definition

_COMMENT_PREFIXES = ("#", "//", "/*", "*", "--")


def build_chunks(path: str, text: str, definitions: Sequence[Definition]) -> tuple[Chunk, ...]:
    """Cut one file into chunks, in source order."""
    lines = split_lines(text)
    if not lines:
        return ()

    symbol_chunks = _symbol_chunks(path, lines, definitions)
    line_chunks = _line_chunks(path, lines, _uncovered(len(lines), symbol_chunks))

    return tuple(sorted((*symbol_chunks, *line_chunks), key=_chunk_order))


def _symbol_chunks(
    path: str,
    lines: Sequence[str],
    definitions: Sequence[Definition],
) -> tuple[Chunk, ...]:
    chunkable = [
        definition
        for definition in definitions
        if definition.role not in CHUNKLESS_ROLES and definition.extent.start <= len(lines)
    ]
    ends = sorted({definition.extent.end for definition in chunkable})

    chunks: list[Chunk] = []
    for definition in chunkable:
        extent = _clamp(definition.extent, len(lines))
        start = _absorb_comment(lines, extent.start, floor=_comment_floor(ends, extent.start))
        span = Span(start, extent.end)

        for piece in _fit_symbol(lines, span, definition, chunkable):
            for body in _split_long_lines(_slice(lines, piece)):
                chunks.append(
                    Chunk(
                        path=path,
                        kind="symbol",
                        span=piece,
                        body=body,
                        identity=definition.qualified_name,
                        role=definition.role,
                        indexable=is_indexable_role(definition.role),
                    )
                )
    return tuple(chunks)


def _fit_symbol(
    lines: Sequence[str],
    span: Span,
    definition: Definition,
    siblings: Sequence[Definition],
) -> tuple[Span, ...]:
    """Bound one definition's span, by header-trimming or by windowing."""
    if _fits(lines, span):
        return (span,)

    first_nested = _first_nested_start(definition, siblings)
    if first_nested is not None and first_nested > span.start:
        header = Span(span.start, first_nested - 1)
        return _windows(lines, header) if not _fits(lines, header) else (header,)

    return _windows(lines, span)


def _first_nested_start(definition: Definition, siblings: Sequence[Definition]) -> int | None:
    """The lowest start line of a chunk-producing definition inside this one."""
    starts = [
        other.extent.start
        for other in siblings
        if other is not definition and definition.extent.contains(other.extent)
    ]
    return min(starts) if starts else None


def _comment_floor(ends: Sequence[int], start: int) -> int:
    """The highest line a definition at ``start`` may absorb upward to."""
    previous = [end for end in ends if end < start]
    return (max(previous) + 1) if previous else 1


def _absorb_comment(lines: Sequence[str], start: int, *, floor: int) -> int:
    """Extend ``start`` upward over the comment block directly above it."""
    limit = max(floor, start - COMMENT_MAX_LINES)
    absorbed = start
    cursor = start - 1

    while cursor >= limit:
        stripped = lines[cursor - 1].strip()
        if stripped.startswith(_COMMENT_PREFIXES):
            absorbed = cursor
        elif stripped:
            break
        cursor -= 1

    return absorbed


def _uncovered(total_lines: int, chunks: Sequence[Chunk]) -> tuple[Span, ...]:
    """Contiguous line ranges no symbol chunk covers, in order."""
    covered = bytearray(total_lines + 1)
    for chunk in chunks:
        for line in range(chunk.span.start, chunk.span.end + 1):
            covered[line] = 1

    gaps: list[Span] = []
    start: int | None = None
    for line in range(1, total_lines + 1):
        if covered[line]:
            if start is not None:
                gaps.append(Span(start, line - 1))
                start = None
        elif start is None:
            start = line
    if start is not None:
        gaps.append(Span(start, total_lines))
    return tuple(gaps)


def _line_chunks(path: str, lines: Sequence[str], gaps: Sequence[Span]) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for gap in gaps:
        if _is_blank(lines, gap):
            continue
        for window in _windows(lines, gap):
            for body in _split_long_lines(_slice(lines, window)):
                chunks.append(Chunk(path=path, kind="lines", span=window, body=body))
    return tuple(chunks)


def _windows(lines: Sequence[str], span: Span) -> tuple[Span, ...]:
    """Split ``span`` into overlapping windows within the line and char caps."""
    windows: list[Span] = []
    cursor = span.start

    while cursor <= span.end:
        end = cursor
        width = 0
        while end <= span.end:
            width += len(lines[end - 1]) + 1
            if end > cursor and (width > CHUNK_CHARS or end - cursor + 1 > CHUNK_LINES):
                end -= 1
                break
            end += 1
        end = min(end, span.end)

        windows.append(Span(cursor, end))
        if end >= span.end:
            break
        cursor = max(end - CHUNK_OVERLAP_LINES + 1, cursor + 1)

    return tuple(windows)


def _split_long_lines(body: str) -> tuple[str, ...]:
    """Split a body whose text exceeds the character cap."""
    if len(body) <= CHUNK_CHARS:
        return (body,)
    return tuple(
        body[offset : offset + CHUNK_CHARS]
        for offset in range(0, len(body), CHUNK_LONG_LINE_STRIDE)
    )


def _slice(lines: Sequence[str], span: Span) -> str:
    return "\n".join(lines[span.as_slice()])


def _fits(lines: Sequence[str], span: Span) -> bool:
    return span.line_count <= CHUNK_LINES and len(_slice(lines, span)) <= CHUNK_CHARS


def _is_blank(lines: Sequence[str], span: Span) -> bool:
    return not _slice(lines, span).strip()


def _clamp(span: Span, total_lines: int) -> Span:
    """Keep a span inside the file."""
    end = min(span.end, total_lines)
    return span if end == span.end else Span(min(span.start, end), end)


def _chunk_order(chunk: Chunk) -> tuple[int, int, str, str]:
    """Source order, with a total tiebreak so output is byte-reproducible."""
    return (chunk.span.start, chunk.span.end, chunk.identity, chunk.digest)
