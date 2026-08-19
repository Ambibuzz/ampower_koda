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
            "The ranked entry point. Runs a literal grep and the semantic retriever in "
            "parallel and merges them into exact: / related:. Has no glob and cannot be "
            "narrowed — by design."
        ),
        caps=("10 hits", "4,000 chars"),
    ),
    ToolSpec(
        name="grep",
        parameters=("regex", "glob?", "mode?"),
        description="Literal regex over the tree.",
        caps=("200 rows", "300 chars/line"),
    ),
    ToolSpec(
        name="glob",
        parameters=("pattern",),
        description="Paths by pattern. Order is mtime, not relevance.",
        caps=("100 paths",),
    ),
    ToolSpec(
        name="outline",
        parameters=("path",),
        description="Definitions and signatures, no bodies. PREFER over read.",
        caps=("120 rows",),
    ),
    ToolSpec(
        name="symbols",
        parameters=("path",),
        description="defs: and refs: lists for one file.",
        caps=("150 each",),
    ),
    ToolSpec(
        name="refs",
        parameters=("symbol",),
        description=(
            "Reference sites. Falls back to the tag index, appending ' in Container.path' — "
            "the cheapest useful fact about a reference site."
        ),
        caps=("100 rows",),
    ),
    ToolSpec(
        name="read",
        parameters=("path", "symbol?", "start?", "end?"),
        description=(
            "Lines from one file. Give a path, or a symbol to resolve to its own extent. "
            "Fires onRead, which is what gates edit."
        ),
        caps=("400 lines",),
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
        caps=("4,000-char query", "24 results", "120 files"),
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
        caps=("200 fields",),
    ),
)

TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in CATALOGUE)


def by_name(name: str) -> ToolSpec | None:
    """One tool's contract, or ``None`` when nothing is spelled that way."""
    return next((spec for spec in CATALOGUE if spec.name == name), None)
