"""The tool array, frozen at session start."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool's contract, as the model and the gates both see it."""

    name: str
    parameters: tuple[str, ...] = ()
    description: str = ""

    caps: tuple[str, ...] = ()
    """The limits, stated. A cap the model cannot see is a cap it will keep
    walking into — and every one of these ends in a truncation marker, because
    a tool result is a leaf and can never ask whether there was more."""


CATALOGUE: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search",
        parameters=("query",),
        description=(
            "The ranked entry point: hits from the local retriever, reranked when a "
            "reranker is configured. Has no glob and cannot be narrowed — by design."
        ),
        caps=("10 hits", "4,000 chars"),
    ),
    ToolSpec(
        name="grep",
        parameters=("regex", "glob?", "mode?"),
        description=(
            "Literal substring search over indexed source. The regex parameter is a historical name: "
            "regular expressions and | alternation are NOT supported. Search one exact phrase per call. "
            "Optional glob filters paths; mode='files' returns matching paths, otherwise matching lines."
        ),
        caps=("200 rows", "300 chars/line", "8,000 chars"),
    ),
    ToolSpec(
        name="glob",
        parameters=("pattern",),
        description="Paths by pattern. Order is mtime, not relevance.",
        caps=("100 paths", "8,000 chars"),
    ),
    ToolSpec(
        name="outline",
        parameters=("path",),
        description="Definitions and signatures, no bodies. PREFER over read.",
        caps=("120 rows", "8,000 chars"),
    ),
    ToolSpec(
        name="symbols",
        parameters=("path",),
        description="defs: and refs: lists for one file.",
        caps=("150 each", "8,000 chars"),
    ),
    ToolSpec(
        name="refs",
        parameters=("symbol",),
        description=(
            "Reference sites. Falls back to the tag index, appending ' in Container.path' — "
            "the cheapest useful fact about a reference site."
        ),
        caps=("100 rows", "8,000 chars"),
    ),
    ToolSpec(
        name="read",
        parameters=("path", "symbol?", "start?", "end?", "offset?"),
        description=(
            "Lines from one file. A symbol selects its definition's extent; path narrows symbol lookup. "
            "An ambiguous symbol returns candidates instead of choosing a file. "
            "Use grep or outline to locate the relevant span first. "
            "Path-only reads return the first 80 lines. For a long single line, "
            "use start/end for that line and offset for its character position."
        ),
        caps=("80 lines by default", "600 lines maximum", "16,000 chars"),
    ),
    ToolSpec(
        name="explore",
        parameters=("query?", "path?", "symbol?"),
        description="A bounded batch. Returns orientation, not full bodies.",
        caps=("6,500 chars",),
    ),
    ToolSpec(
        name="definition",
        parameters=("symbol", "path?"),
        description="Use it before read when a symbol may be shadowed, imported, or aliased.",
    ),
    ToolSpec(
        name="ast_search",
        parameters=("query", "language?", "glob?"),
        description="Tree-sitter S-expression with a @match capture. It is not regex.",
        caps=("4,000-char query", "24 results", "120 files", "8,000 chars"),
    ),
    ToolSpec(
        name="recall",
        parameters=("id",),
        description=(
            "Dereference a ledger id. Re-reads and re-hashes the refs, and announces "
            "staleness rather than serving the old span."
        ),
    ),
    ToolSpec(
        name="trace_discover",
        parameters=("path", "line", "symbol?"),
        description="The transitive call graph. Three lines to the model; the graph to the UI.",
    ),
    ToolSpec(
        name="read_doctype_schema",
        parameters=("doctype",),
        description=(
            "A Frappe doctype's fields, types, options and links, from its JSON. "
            "PREFER over reading the .json by hand — the file is mostly layout metadata "
            "and the fields are what a question about a doctype is actually asking."
        ),
        caps=("200 fields", "8,000 chars"),
    ),
)

TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in CATALOGUE)
