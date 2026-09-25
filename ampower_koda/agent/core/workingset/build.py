"""One retrieval pass per user message, spent line by line."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from ampower_koda.agent.frappe_rpc import call_options

from ..constants import (
    LEDGER_RECENT_WINDOW,
    WORKING_SET_CALLERS,
    WORKING_SET_EXCERPT_CHARS,
    WORKING_SET_FULL_SPAN_CHARS,
    WORKING_SET_FULL_SPANS,
    WORKING_SET_MAX_EDITED,
    WORKING_SET_MAX_SPANS,
    WORKING_SET_OUTLINE_CHARS,
    WORKING_SET_OUTLINE_FILES,
    WORKING_SET_RELATIVE_FLOOR,
    WORKING_SET_SEARCH_LIMIT,
    WORKING_SET_WEAK_COVERAGE,
)
from ..contracts.chunks import Chunk
from ..contracts.ledger import Ledger
from ..contracts.repository import RepositoryIndex
from ..contracts.retrieval import Hit
from ..contracts.working_set import WorkingSet, WorkingSpan
from ..identity import anchor_id
from ..retrieval.engine import Retriever, search
from ..retrieval.excerpts import _NOISE, excerpt
from ..retrieval.query import plain_query
from ..tokens import estimate_tokens

HEADER = "WORKING SET (most relevant spans for this message)"
COMPLETE_NOTE = ("Numbered spans are complete current source: cite or edit them directly, without "
                 "reading them again. Unnumbered entries are pointers, even when they carry "
                 "an excerpt.")


def working_set_for(
    message: str,
    retriever: Retriever,
    *,
    ledger: Ledger | None = None,
    edited: Sequence[str] = (),
    max_tokens: int = 0,
) -> WorkingSet:
    """Build the block from the same dedicated ranking used by search tools."""
    query = plain_query(message)
    if not query:
        return WorkingSet()

    result = search(retriever, query, limit=WORKING_SET_SEARCH_LIMIT)
    spans = _collect(_relevant(result.hits), ledger, query, retriever.index)
    if not spans:
        return WorkingSet(coverage=result.confidence, notes=result.notes)

    working = _render(spans, edited, coverage=result.confidence, max_tokens=max_tokens,
                      reranked=result.reranked, outlines=_outlines(retriever.index, spans))
    return replace(working, reranked=result.reranked and bool(result.hits), notes=result.notes)


def _relevant(hits: Sequence[Hit]) -> tuple[Hit, ...]:
    """Drop the tail far below the best hit; the ranking already ordered it."""
    if not hits:
        return ()
    floor = hits[0].score * WORKING_SET_RELATIVE_FLOOR
    return tuple(hit for index, hit in enumerate(hits) if index == 0 or hit.score >= floor)


def _collect(hits: Sequence[Hit], ledger: Ledger | None, query: str = "",
             index: RepositoryIndex | None = None) -> list[WorkingSpan]:
    """Retrieved spans (the best ones whole), their callers, then established refs."""
    spans: list[WorkingSpan] = []
    seen: set[str] = set()

    retrieved = list(_retrieved(hits, query, index))
    related = list(_related(retrieved, index)) if index is not None else []
    for span in (*retrieved, *related, *_established(ledger)):
        if span.location in seen or _overlaps(span, spans):
            continue
        seen.add(span.location)
        spans.append(span)
        if len(spans) >= WORKING_SET_MAX_SPANS:
            break

    return spans


def _retrieved(hits: Sequence[Hit], query: str = "",
               index: RepositoryIndex | None = None) -> Iterable[WorkingSpan]:
    """Tier one: scored against this message, with anchors. The first few whole."""
    whole = 0
    symbols: set[tuple[str, str]] = set()
    for hit in hits:
        key = (hit.path, hit.symbol or hit.location)
        if key in symbols:
            continue   # another window of a function already shown
        symbols.add(key)
        source = _whole(index, hit.chunk) if index is not None and whole < WORKING_SET_FULL_SPANS else None
        if source is not None:
            whole += 1
            start, end, body = source
            yield WorkingSpan(location=f"{hit.path}:{start}-{end}", symbol=hit.symbol, score=hit.score,
                              anchor=anchor_id(hit.path, body), origin="retrieved", body=body)
            continue
        text = excerpt(hit.chunk.body, query, max_chars=WORKING_SET_EXCERPT_CHARS)
        yield WorkingSpan(
            location=hit.location,
            excerpt=text,
            symbol=hit.symbol,
            score=hit.score,
            anchor=anchor_id(hit.path, text),
            origin="retrieved",
        )


def _whole(index: RepositoryIndex, chunk: Chunk) -> tuple[int, int, str] | None:
    """The complete definition a hit belongs to, numbered like a ``read`` result.

    Long functions are indexed as overlapping windows; the windows are joined
    back by line number. A gap (lines no chunk covers) falls back to the hit's
    own window rather than inventing text.
    """
    analysis = index.files.get(chunk.path)
    start, end = chunk.span.start, chunk.span.end
    if analysis is not None and chunk.identity:
        for definition in analysis.definitions:
            if (definition.qualified_name == chunk.identity
                    and definition.extent.start <= end and start <= definition.extent.end):
                start, end = min(start, definition.extent.start), max(end, definition.extent.end)
                break
    lines = _lines(analysis.chunks if analysis is not None else (chunk,), start, end)
    if any(number not in lines for number in range(start, end + 1)):
        start, end = chunk.span.start, chunk.span.end
        lines = _lines((chunk,), start, end)
        if any(number not in lines for number in range(start, end + 1)):
            return None
    while start < end and (not lines[start].strip() or lines[start].lstrip().startswith(_NOISE[:-1])):
        start += 1   # licence banners and comment runs, never a decorator
    rows, used, last = [], 0, start - 1
    for number in range(start, end + 1):
        row = f"{number:>5}  {lines[number]}"
        if rows and used + len(row) + 1 > WORKING_SET_FULL_SPAN_CHARS:
            break
        rows.append(row)
        used += len(row) + 1
        last = number
    body = "\n".join(rows)
    if last < end:
        body += f"\n[lines {last + 1}-{end} not shown; read {chunk.path} start={last + 1} end={end}]"
    return start, end, body


def _lines(chunks: Iterable[Chunk], start: int, end: int) -> dict[int, str]:
    lines: dict[int, str] = {}
    for part in chunks:
        if part.span.end < start or part.span.start > end:
            continue
        body = part.body.split("\n")
        if len(body) != part.span.line_count:
            continue   # a long-line split; its rows no longer map to line numbers
        for number, text in zip(range(part.span.start, part.span.end + 1), body):
            lines.setdefault(number, text)
    return lines


def _related(spans: Sequence[WorkingSpan], index: RepositoryIndex) -> Iterable[WorkingSpan]:
    """One hop out from the complete spans: who calls them, and which client RPC reaches them.

    The retrieved spans answer "what matches the request"; the entry point
    that runs them is often worded differently and ranks below the floor.
    """
    produced = 0
    names: list[str] = []
    for span in spans:
        if not span.body or not span.symbol:
            continue
        path, _, lines = span.location.rpartition(":")
        start, _, end = lines.partition("-")
        bare = span.symbol.rsplit(".", 1)[-1]
        names.append(bare)
        for caller in _callers(index, bare, path, int(start), int(end or start)):
            if produced >= WORKING_SET_CALLERS:
                return
            names.append(caller.symbol.rsplit(".", 1)[-1])
            produced += 1
            yield caller
    for site in _rpc_sites(index, names):
        yield site
        return


def _callers(index: RepositoryIndex, name: str, path: str, start: int, end: int) -> Iterable[WorkingSpan]:
    for file_path, analysis in index.files.items():
        for reference in analysis.references:
            if reference.name != name or (file_path == path and start <= reference.line <= end):
                continue
            owner = _enclosing(analysis.definitions, reference.line)
            if owner is None:
                continue
            yield WorkingSpan(
                location=f"{file_path}:{owner.extent.start}-{owner.extent.end}",
                symbol=owner.qualified_name,
                excerpt=_line_text(analysis.chunks, owner.name_line),
                origin="related",
            )


def _rpc_sites(index: RepositoryIndex, names: Sequence[str]) -> Iterable[WorkingSpan]:
    """Client code whose ``frappe.call`` method path ends in one of ``names``."""
    wanted = {name for name in names if name}
    for file_path, analysis in index.files.items():
        if not file_path.endswith((".js", ".ts")):
            continue
        for chunk in analysis.chunks:
            if "frappe" not in chunk.body:
                continue
            for options in call_options(chunk.body):
                method = str(options.get("method") or "")
                if method.rsplit(".", 1)[-1] in wanted:
                    yield WorkingSpan(location=chunk.location, symbol=chunk.identity,
                                      excerpt=f"frappe.call {method}", origin="related")
                    break


def _enclosing(definitions, line: int):
    """The innermost definition containing ``line``."""
    best = None
    for definition in definitions:
        if definition.extent.start <= line <= definition.extent.end:
            if best is None or definition.extent.line_count < best.extent.line_count:
                best = definition
    return best


def _line_text(chunks: Iterable[Chunk], line: int) -> str:
    text = _lines(chunks, line, line).get(line, "")
    return " ".join(text.split())[:120]


def _overlaps(span: WorkingSpan, spans: Sequence[WorkingSpan]) -> bool:
    """Whether ``span`` repeats lines already shown.

    A method inside a class shown whole, or a caller pointer touching a shown
    span, adds only a second copy of the same lines.
    """
    if span.origin == "established":
        return False
    path, start, end = _range(span)
    if start is None:
        return False
    for other in spans:
        other_path, other_start, other_end = _range(other)
        if other_path != path or other_start is None:
            continue
        if span.origin == "related" and start <= other_end and other_start <= end:
            return True
        if other.body and other_start <= start and end <= other_end:
            return True
    return False


def _range(span: WorkingSpan) -> tuple[str, int | None, int]:
    path, _, lines = span.location.rpartition(":")
    start, _, end = lines.partition("-")
    if not start.isdigit():
        return span.location, None, 0
    return path, int(start), int(end) if end.isdigit() else int(start)


def _outlines(index: RepositoryIndex, spans: Sequence[WorkingSpan]) -> list[str]:
    """Names-only outlines, ``symbol@line``, for the files the whole spans came from."""
    blocks: list[str] = []
    for span in spans:
        if len(blocks) >= WORKING_SET_OUTLINE_FILES:
            break
        path = span.location.rpartition(":")[0]
        analysis = index.files.get(path)
        if not span.body or analysis is None or len(analysis.definitions) < 3:
            continue
        if any(block.startswith(f"outline {path}:") for block in blocks):
            continue
        names = [f"{d.qualified_name}@{d.name_line}"
                 for d in sorted(analysis.definitions, key=lambda d: d.name_line)]
        text = f"outline {path}: " + ", ".join(names)
        if len(text) > WORKING_SET_OUTLINE_CHARS:
            text = text[:WORKING_SET_OUTLINE_CHARS].rsplit(", ", 1)[0] + ", …"
        blocks.append(text)
    return blocks


def _established(ledger: Ledger | None) -> Iterable[WorkingSpan]:
    """Tier two: refs from the last twelve live entries, newest first."""
    if ledger is None:
        return
    for entry in ledger.recent(LEDGER_RECENT_WINDOW):
        for ref in entry.refs:
            if ref.start:
                yield WorkingSpan(location=ref.location, origin="established")


def _render(
    spans: Sequence[WorkingSpan],
    edited: Sequence[str],
    *,
    coverage: float,
    max_tokens: int,
    reranked: bool = False,
    outlines: Sequence[str] = (),
) -> WorkingSet:
    """Header, optional warning, spans, outlines, edited clause — spent line by line."""
    lines = [HEADER]
    whole = any(span.body for span in spans)
    warning = _warning(coverage, reranked=reranked, whole=whole)
    truncated = False

    def fits(candidate: Sequence[str]) -> bool:
        return not max_tokens or estimate_tokens("\n".join(candidate)) <= max_tokens

    if warning:
        if not fits([*lines, warning]):
            return WorkingSet(coverage=coverage, truncated=True)
        lines.append(warning)
    if whole and fits([*lines, COMPLETE_NOTE]):
        lines.append(COMPLETE_NOTE)

    kept: list[WorkingSpan] = []
    for span in spans:
        if not fits([*lines, span.line()]) and span.body:
            # A whole span that no longer fits still earns a pointer.
            text = excerpt(span.body, "", max_chars=WORKING_SET_EXCERPT_CHARS).split("\n", 1)[0]
            span = replace(span, body="", excerpt=" ".join(text.split()[1:])[:120])
        line = span.line()
        if not fits([*lines, line]):
            truncated = True
            break
        lines.append(line)
        kept.append(span)

    for outline in outlines:
        if fits([*lines, outline]):
            lines.append(outline)
        else:
            truncated = True

    changed = _changed_clause(edited)
    if changed and fits([*lines, changed]):
        lines.append(changed)
    elif changed:
        truncated = True

    if not kept:
        return WorkingSet(coverage=coverage, truncated=truncated)

    text = "\n".join(lines)
    return WorkingSet(
        text=text,
        spans=tuple(kept),
        tokens=estimate_tokens(text),
        coverage=coverage,
        truncated=truncated,
        broadened=True,
    )


def _warning(coverage: float, *, reranked: bool = False, whole: bool = False) -> str:
    """The self-report when coverage is below the weak floor."""
    if coverage >= WORKING_SET_WEAK_COVERAGE:
        return ""
    if whole:
        # The numbered source is exact; only the choice of places is uncertain.
        return "[Low word overlap with the request: confirm these are the right places before building on them.]"
    if reranked:
        return "[Semantically ranked candidates with low lexical overlap; verify the source before editing.]"
    return (
        f"[weak automatic retrieval: {coverage:.0%} query coverage; "
        "verify before relying on these spans]"
    )


def _changed_clause(edited: Sequence[str]) -> str:
    if not edited:
        return ""
    names = list(dict.fromkeys(edited))[:WORKING_SET_MAX_EDITED]
    return "changed this session: " + ", ".join(names)
