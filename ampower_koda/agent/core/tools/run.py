"""The tools the read-only port can serve, and the host for the three it cannot."""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..contracts.agent import ToolHost, ToolOutcome
from ..contracts.ledger import Ledger
from ..contracts.repository import definitions_by_name, files_referencing
from ..contracts.source import split_lines
from ..errors import CoreError
from ..globs import compile_globs
from ..ledger.recall import rehydrate
from ..retrieval.engine import Retriever, search
from ..retrieval.excerpts import excerpt
from ..workspace.ports import Workspace
from ..workspace.redaction import redaction_matcher
from .results import cap_chars, cap_rows

SEARCH_HITS = 10
SEARCH_CHARS = 4_000
GREP_ROWS = 200
GREP_LINE_CHARS = 300
GLOB_PATHS = 100
OUTLINE_ROWS = 120
SYMBOL_ROWS = 150
REFS_ROWS = 100
READ_LINES = 600
READ_DEFAULT_LINES = 80
# About 4k tokens: enough that a long file takes few chained reads.
READ_CHARS = 16_000
TOOL_RESULT_CHARS = 8_000
EXPLORE_CHARS = 6_500

NOT_WIRED = "[{tool} is not wired in this host — nothing was executed]"


@dataclass(frozen=True, slots=True)
class NullHost:
    """Declines every tool, in a way the model can act on."""

    def call(self, name: str, arguments: Mapping[str, object]) -> ToolOutcome:  # noqa: ARG002
        return ToolOutcome(text=NOT_WIRED.format(tool=name), ok=False)


def run_tool(
    name: str,
    arguments: Mapping[str, object],
    *,
    retriever: Retriever,
    workspace: Workspace,
    ledger: Ledger | None = None,
    host: ToolHost | None = None,
) -> ToolOutcome:
    """Dispatch one call. Returns a value in every case, including the bad ones."""
    handler = _HANDLERS.get(name)
    try:
        outcome = (handler(arguments, retriever, workspace, ledger) if handler is not None
                   else (host or NullHost()).call(name, arguments))
        # read bounds itself to READ_CHARS; every other result is capped here.
        if name != "read" and len(outcome.text) > TOOL_RESULT_CHARS:
            capped = cap_chars(outcome.text, TOOL_RESULT_CHARS)
            return replace(outcome, text=capped.text + "\nRequest a narrower range or query for omitted details.",
                           truncated=True, dropped=capped.dropped, entry_text="")
        return outcome
    except (CoreError, OSError, ValueError, KeyError) as error:
        return ToolOutcome(text=f"[error: {name} — {error}]", ok=False)


def _search(arguments, retriever, workspace, ledger):  # noqa: ANN001, ARG001
    """Ranked hits. Has no glob and cannot be narrowed — by design."""
    query = str(arguments.get("query", "")).strip()
    if not query:
        return ToolOutcome(text="[error: search needs a query]", ok=False)

    result = search(retriever, query, limit=SEARCH_HITS)
    if result.is_empty:
        notes = "\n[" + "; ".join(result.notes) + "]" if result.notes else ""
        return ToolOutcome(text=f'search "{query}": 0 hits' + notes)

    method = "dedicated reranker" if result.reranked else "local retrieval"
    rows = [f'search "{query}": {len(result.hits)} hits ({method}), lexical coverage {result.confidence:.2f}']
    rows += [_hit_row(hit, query) for hit in result.hits]
    if result.notes:
        rows.append(f"[{'; '.join(result.notes)}]")
    return ToolOutcome(text=cap_chars("\n".join(rows), SEARCH_CHARS).text)


def _hit_row(hit, query: str = "") -> str:  # noqa: ANN001
    symbol = f" {hit.symbol}" if hit.symbol else ""
    note = f"  [{hit.note}]" if hit.note else ""
    text = excerpt(hit.chunk.body, query, max_chars=300)
    return f"{hit.location} [{hit.score:.3f}]{symbol} — {text}{note}"


def _explore(arguments, retriever, workspace, ledger):  # noqa: ANN001
    """A bounded batch: hits, then the outline of the top file."""
    query = str(arguments.get("query", "")).strip()
    sections = []
    if query:
        sections.append(_search(arguments, retriever, workspace, ledger).text)

    path = _explore_path(arguments, retriever, query)
    if path:
        sections.append(_outline({"path": path}, retriever, workspace, ledger).text)

    if not sections:
        return ToolOutcome(text="[error: explore needs a query, a path or a symbol]", ok=False)
    return ToolOutcome(text=cap_chars("\n\n".join(sections), EXPLORE_CHARS).text)


def _explore_path(arguments, retriever, query: str) -> str:  # noqa: ANN001
    """Which file to outline: the one named, the one a symbol lives in, or the
    top hit.
    """
    path = str(arguments.get("path", "")).strip()
    if path:
        return path

    symbol = str(arguments.get("symbol", "")).strip()
    if symbol:
        table = definitions_by_name(retriever.index)
        found = table.get(symbol) or table.get(symbol.rsplit(".", 1)[-1]) or ()
        if found:
            return str(found[0][0])

    if query:
        hits = search(retriever, query, limit=1).hits
        return hits[0].path if hits else ""
    return ""


def _outline(arguments, retriever, workspace, ledger):  # noqa: ANN001, ARG001
    """Definitions and signatures, no bodies. PREFER over read."""
    path = str(arguments.get("path", "")).strip()
    analysis = retriever.index.files.get(path)
    if analysis is None:
        return ToolOutcome(text=f"[error: {path or '<none>'} is not in the index]", ok=False)

    rows = [
        f"{path}:{definition.extent.start} {definition.role} {definition.qualified_name}"
        for definition in sorted(analysis.definitions, key=lambda d: d.extent.start)
    ]
    if not rows:
        return ToolOutcome(text=f"{path}: no definitions ({analysis.language})")
    header = f"{path}: {len(rows)} definition(s), {analysis.language}"
    return ToolOutcome(text=header + "\n" + cap_rows(rows, OUTLINE_ROWS).text)


def _symbols(arguments, retriever, workspace, ledger):  # noqa: ANN001, ARG001
    """defs: and refs: for one file."""
    path = str(arguments.get("path", "")).strip()
    analysis = retriever.index.files.get(path)
    if analysis is None:
        return ToolOutcome(text=f"[error: {path or '<none>'} is not in the index]", ok=False)

    defs = [f"{d.qualified_name} ({d.role}) :{d.extent.start}" for d in analysis.definitions]
    refs = [f"{r.name} ({r.kind}) :{r.line}" for r in analysis.references]
    return ToolOutcome(text="\n".join([
        f"{path} defs: {len(defs)}",
        cap_rows(defs, SYMBOL_ROWS).text,
        f"{path} refs: {len(refs)}",
        cap_rows(refs, SYMBOL_ROWS).text,
    ]))


def _refs(arguments, retriever, workspace, ledger):  # noqa: ANN001, ARG001
    """Reference sites, with the container that makes each one legible."""
    name = str(arguments.get("symbol", "")).strip()
    if not name:
        return ToolOutcome(text="[error: refs needs a symbol]", ok=False)

    rows = []
    for path in files_referencing(retriever.index, name):
        analysis = retriever.index.files[path]
        for reference in analysis.references:
            if reference.name == name:
                container = _container(analysis, reference.line)
                rows.append(f"{path}:{reference.line} {reference.kind}{container}")

    if not rows:
        return ToolOutcome(text=f'refs "{name}": 0 sites')
    head = f'refs "{name}": {len(rows)} site(s)'
    return ToolOutcome(text=head + "\n" + cap_rows(rows, REFS_ROWS).text)


def _container(analysis, line: int) -> str:  # noqa: ANN001
    """`` in Cart.total`` — the innermost definition whose span holds this line."""
    holding = [
        definition
        for definition in analysis.definitions
        if definition.extent.start <= line <= definition.extent.end
    ]
    if not holding:
        return ""
    innermost = min(holding, key=lambda d: d.extent.end - d.extent.start)
    return f" in {innermost.qualified_name}"


def _definition(arguments, retriever, workspace, ledger):  # noqa: ANN001, ARG001
    """Exact definitions. Use before read when a symbol may be shadowed."""
    name = str(arguments.get("symbol", "")).strip()
    table = definitions_by_name(retriever.index)
    found = table.get(name) or table.get(name.rsplit(".", 1)[-1]) or ()

    scope = str(arguments.get("path", "")).strip()
    if scope:
        narrowed = tuple(pair for pair in found if pair[0] == scope)
        if found and not narrowed:
            return ToolOutcome(text=f'definition "{name}": not defined in {scope}')
        found = narrowed

    if not found:
        return ToolOutcome(text=f'definition "{name}": not found')

    rows = [
        f"{path}:{definition.extent.start}-{definition.extent.end} "
        f"{definition.role} {definition.qualified_name}"
        for path, definition in found
    ]
    return ToolOutcome(text=f'definition "{name}": {len(rows)}\n' + "\n".join(rows))


def _read(arguments, retriever, workspace, ledger):  # noqa: ANN001, ARG001
    """Bounded source spans; oversized individual lines can be paged by offset."""
    path = str(arguments.get("path", "")).strip()
    symbol = str(arguments.get("symbol", "")).strip()

    if symbol:
        table = definitions_by_name(retriever.index)
        found = table.get(symbol) or ()
        if path:
            found = tuple(item for item in found if item[0] == path)
        if not found:
            scope = f" in {path}" if path else ""
            return ToolOutcome(text=f'read "{symbol}": no such symbol{scope}', ok=False)
        if len(found) > 1:
            choices = "; ".join(f"{p}:{d.extent.start}-{d.extent.end} {d.qualified_name}" for p, d in found[:10])
            return ToolOutcome(text=f'read "{symbol}": ambiguous; use a qualified symbol and path, '
                                    f'or explicit lines without symbol. Candidates: {choices}', ok=False)
        path, definition = found[0]
        span = definition.extent
        arguments = {**arguments, "start": span.start, "end": span.end}

    if not path:
        return ToolOutcome(text="[error: read needs a path or a symbol]", ok=False)

    pattern = _redacted(path, retriever, workspace)
    if pattern is not None:
        return ToolOutcome(text=f"[error: {path} matches the redaction pattern {pattern} "
                                "(secrets are never read)]", ok=False)

    lines = split_lines(workspace.read_bytes(path).decode("utf-8", errors="replace"))
    start = max(1, int(arguments.get("start", 1) or 1))
    end = int(arguments.get("end", 0) or (start + READ_DEFAULT_LINES - 1))
    end = min(end, start + READ_LINES - 1, len(lines))
    if start > len(lines):
        return ToolOutcome(text=f"[error: {path} has {len(lines)} lines]", ok=False)

    if end < start:
        return ToolOutcome(text="[error: end must be at or after start]", ok=False)
    offset = max(0, int(arguments.get("offset", 0) or 0))
    if offset >= len(lines[start - 1]) and offset:
        return ToolOutcome(text="[error: offset is past the selected line]", ok=False)
    if offset:
        end = start
    allowance = max(1, READ_CHARS - 2 * len(path) - 250)
    rows = []
    used = 0
    shown_end = start
    partial = bool(offset)
    more = ""
    for number in range(start, end + 1):
        content = lines[number - 1][offset if number == start else 0:]
        row = f"{number:>5}  {content}"
        if used + len(row) + 1 > allowance:
            if rows:
                break
            take = max(1, allowance - 8)
            rows.append(f"{number:>5}  {content[:take]}")
            partial = True
            more = (f"\n[Line {number} excerpt, character offset {offset}; truncated. Continue with "
                    f"read(path={path!r}, start={number}, end={number}, offset={offset + take}).]")
            break
        rows.append(row)
        used += len(row) + 1
        shown_end = number
    if not more and shown_end < len(lines):
        more = (f"\n[Excerpt; {len(lines) - shown_end} more lines. Continue with read(path={path!r}, "
                f"start={shown_end + 1}); request every further range you need as parallel read calls "
                "in this same turn.]")
    if partial and not more:
        more = f"\n[Line {start} excerpt from character offset {offset}.]"
    return ToolOutcome(
        text=f"{path}:{start}-{shown_end}\n" + "\n".join(rows) + more,
        entry_text="" if partial else f"{path}:{start}-{shown_end}",
        truncated=partial or shown_end < len(lines),
    )


def _redacted(path: str, retriever: Retriever, workspace: Workspace) -> str | None:
    """The pattern that redacts ``path``, or ``None``. Redaction guards reads too,
    not only the index: a file skipped as ``redacted`` could still be read by path."""
    normalised = posixpath.normpath(path.replace("\\", "/"))
    skipped = retriever.index.skipped.get(normalised)
    if skipped is not None and skipped.reason == "redacted":
        return skipped.detail.removeprefix("matched ")
    return redaction_matcher(workspace)(normalised)


def _grep(arguments, retriever, workspace, ledger):  # noqa: ANN001, ARG001
    """Literal-substring search over the indexed files."""
    pattern = str(arguments.get("regex") or arguments.get("pattern") or "").strip()
    if not pattern:
        return ToolOutcome(text="[error: grep needs a pattern]", ok=False)

    glob = str(arguments.get("glob", "")).strip()
    within = compile_globs([glob]) if glob else None
    files_only = str(arguments.get("mode", "")) == "files"
    rows: list[str] = []
    hit_files: set[str] = set()

    for path in retriever.index.paths:
        if within is not None and within(path) is None:
            continue
        analysis = retriever.index.files[path]
        seen_lines: set[int] = set()
        for chunk in analysis.chunks:
            for offset, line in enumerate(split_lines(chunk.body)):
                if pattern not in line:
                    continue
                number = chunk.span.start + offset
                if number in seen_lines:
                    continue
                seen_lines.add(number)
                hit_files.add(path)
                if files_only:
                    break
                rows.append(f"{path}:{number}: {line.strip()[:GREP_LINE_CHARS]}")
            if files_only and path in hit_files:
                break

    if files_only:
        listing = sorted(hit_files)
        return ToolOutcome(
            text=f'grep "{pattern}": {len(listing)} file(s)\n' + cap_rows(listing, GREP_ROWS).text
        )
    if not rows:
        return ToolOutcome(text=f'grep "{pattern}": 0 matches')
    head = f'grep "{pattern}": {len(rows)} matches in {len(hit_files)} file(s)'
    return ToolOutcome(text=head + "\n" + cap_rows(rows, GREP_ROWS).text)


def _glob(arguments, retriever, workspace, ledger):  # noqa: ANN001, ARG001
    """Paths by pattern. Order is mtime, not relevance — and it says so."""
    pattern = str(arguments.get("pattern", "")).strip() or "**/*"
    within = compile_globs([pattern])
    found = [path for path in retriever.index.paths if within(path) is not None]
    ordered = sorted(found, key=lambda p: -_mtime(retriever.index, p))
    head = f'glob "{pattern}": {len(ordered)} path(s) — order is mtime, not relevance'
    return ToolOutcome(text=head + "\n" + cap_rows(ordered, GLOB_PATHS, unit="paths").text)


def _mtime(index, path: str) -> int:  # noqa: ANN001
    analysis = index.files.get(path)
    return analysis.stat.mtime_ns if analysis and analysis.stat else 0


def _recall(arguments, retriever, workspace, ledger):  # noqa: ANN001, ARG001
    """Dereference a ledger id. Announces staleness rather than serving the old
    span — which is the entire reason an elided result is a pointer and not a
    hole.
    """
    entry_id = str(arguments.get("id", "")).strip()
    if ledger is None:
        return ToolOutcome(text="[error: this session has no ledger]", ok=False)

    _, recalled = rehydrate(ledger, entry_id, workspace)
    if recalled.is_empty:
        return ToolOutcome(text=f"[error: {entry_id} is not a ledger id in this session]", ok=False)
    return ToolOutcome(text=recalled.text, entry_text=f"recall {entry_id}")


_HANDLERS = {
    "search": _search,
    "explore": _explore,
    "outline": _outline,
    "symbols": _symbols,
    "refs": _refs,
    "definition": _definition,
    "read": _read,
    "grep": _grep,
    "glob": _glob,
    "recall": _recall,
}
