"""The tools the read-only port can serve, and the host for the three it cannot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..contracts.agent import ToolHost, ToolOutcome
from ..contracts.ledger import Ledger
from ..contracts.repository import definitions_by_name, files_referencing
from ..contracts.source import split_lines
from ..errors import CoreError
from ..globs import compile_globs
from ..ledger.recall import rehydrate
from ..retrieval.engine import Retriever, search
from ..workspace.ports import Workspace
from .results import cap_chars, cap_rows

SEARCH_HITS = 10
SEARCH_CHARS = 4_000
GREP_ROWS = 200
GREP_LINE_CHARS = 300
GLOB_PATHS = 100
OUTLINE_ROWS = 120
SYMBOL_ROWS = 150
REFS_ROWS = 100
READ_LINES = 400
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
    if handler is None:
        return (host or NullHost()).call(name, arguments)
    try:
        return handler(arguments, retriever, workspace, ledger)
    except (CoreError, OSError, ValueError, KeyError) as error:
        return ToolOutcome(text=f"[error: {name} — {error}]", ok=False)


def _search(arguments, retriever, workspace, ledger):  # noqa: ANN001, ARG001
    """Ranked hits. Has no glob and cannot be narrowed — by design."""
    query = str(arguments.get("query", "")).strip()
    if not query:
        return ToolOutcome(text="[error: search needs a query]", ok=False)

    result = search(retriever, query, limit=SEARCH_HITS)
    if result.is_empty:
        return ToolOutcome(text=f'search "{query}": 0 hits')

    rows = [f'search "{query}": {len(result.hits)} hits, confidence {result.confidence:.2f}']
    rows += [_hit_row(hit) for hit in result.hits]
    if result.notes:
        rows.append(f"[{'; '.join(result.notes)}]")
    return ToolOutcome(text=cap_chars("\n".join(rows), SEARCH_CHARS).text)


def _hit_row(hit) -> str:  # noqa: ANN001
    symbol = f" {hit.symbol}" if hit.symbol else ""
    note = f"  [{hit.note}]" if hit.note else ""
    excerpt = _first_line(hit.chunk.body)
    return f"{hit.location} [{hit.score:.3f}]{symbol} — {excerpt}{note}"


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
    """A span, a symbol, or a whole small file. Path-only reads are allowed
    here and capped, because the discriminated union upstream is a schema
    concern and this layer would rather return 400 lines than an error.
    """
    path = str(arguments.get("path", "")).strip()
    symbol = str(arguments.get("symbol", "")).strip()

    if symbol and not path:
        table = definitions_by_name(retriever.index)
        found = table.get(symbol) or table.get(symbol.rsplit(".", 1)[-1]) or ()
        if not found:
            return ToolOutcome(text=f'read "{symbol}": no such symbol', ok=False)
        path, definition = found[0]
        span = definition.extent
        arguments = {**arguments, "start": span.start, "end": span.end}

    if not path:
        return ToolOutcome(text="[error: read needs a path or a symbol]", ok=False)

    lines = split_lines(workspace.read_bytes(path).decode("utf-8", errors="replace"))
    start = max(1, int(arguments.get("start", 1) or 1))
    end = int(arguments.get("end", 0) or len(lines))
    end = min(end, start + READ_LINES - 1, len(lines))
    if start > len(lines):
        return ToolOutcome(text=f"[error: {path} has {len(lines)} lines]", ok=False)

    body = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))
    more = f"\n… {len(lines) - end} more line(s) (truncated)" if end < len(lines) else ""
    return ToolOutcome(
        text=f"{path}:{start}-{end}\n{body}{more}",
        entry_text=f"{path}:{start}-{end}",
    )


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


def _first_line(body: str) -> str:
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return ""


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

BUILT_IN: frozenset[str] = frozenset(_HANDLERS)
