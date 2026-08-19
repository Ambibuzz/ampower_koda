"""Rendering the ranked repository into a fixed token budget."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..constants import MAP_MAX_TOKENS
from ..contracts.analysis import FileAnalysis
from ..contracts.repo_map import FileRanks, RepoMap
from ..contracts.repository import RepositoryIndex
from ..contracts.symbols import Definition
from ..tokens import estimate_tokens, truncate_to_tokens

MAPPED_ROLES: frozenset[str] = frozenset({"function", "method", "class", "enum", "type"})

_SIGNATURE_MAX_LINES = 6

_DEGRADED_HEADER = (
    "(unmapped: no parseable definitions in this repository — "
    "use grep and glob to locate code)"
)


@dataclass(frozen=True, slots=True)
class _Entry:
    """One file's contribution to the map, pre-rendered so it can be measured."""

    path: str
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def render_repo_map(
    index: RepositoryIndex,
    ranks: FileRanks,
    *,
    max_tokens: int = MAP_MAX_TOKENS,
) -> RepoMap:
    """Render the highest-ranked files that fit into ``max_tokens``."""
    entries = [
        entry
        for path in ranks.ordered()
        if path in index.files
        for entry in (_render_file(path, index.files[path]),)
        if entry is not None
    ]

    if not entries:
        return _degraded(index, max_tokens)

    shown = _largest_prefix_that_fits(entries, max_tokens)
    text = "\n\n".join(entry.text for entry in entries[:shown])

    return RepoMap(
        text=text,
        files_shown=shown,
        files_total=len(entries),
        tokens=estimate_tokens(text),
    )


def _render_file(path: str, analysis: FileAnalysis) -> _Entry | None:
    """Render one file's definitions, or ``None`` if it has none worth showing."""
    definitions = [
        definition for definition in analysis.definitions if definition.role in MAPPED_ROLES
    ]
    if not definitions:
        return None

    signatures = _signatures(analysis)
    lines = [f"{path}:"]
    for definition in sorted(definitions, key=lambda item: (item.extent.start, item.name)):
        indent = "  " * (definition.depth + 1)
        lines.append(f"{indent}{signatures.get(definition.qualified_name, definition.name)}")

    return _Entry(path=path, lines=tuple(lines))


def _signatures(analysis: FileAnalysis) -> Mapping[str, str]:
    """Qualified name → the source line the definition's identifier sits on."""
    lines = _line_table(analysis)
    return {
        definition.qualified_name: signature
        for definition in analysis.definitions
        if (signature := _signature_at(lines, definition.name_line))
    }


def _line_table(analysis: FileAnalysis) -> Mapping[int, str]:
    """Line number → source text, reassembled from this file's chunks."""
    table: dict[int, str] = {}
    for chunk in analysis.chunks:
        body = chunk.body.split("\n")
        if len(body) != chunk.span.line_count:
            continue
        for offset, text in enumerate(body):
            table.setdefault(chunk.span.start + offset, text)
    return table


def _signature_at(lines: Mapping[int, str], start: int) -> str:
    """The declaration beginning at ``start``, continued until its brackets close."""
    first = lines.get(start, "").strip()
    if not first:
        return ""

    parts = [first]
    depth = _bracket_depth(first)
    cursor = start + 1
    while depth > 0 and cursor < start + _SIGNATURE_MAX_LINES:
        following = lines.get(cursor, "").strip()
        if not following:
            break
        parts.append(following)
        depth += _bracket_depth(following)
        cursor += 1

    return " ".join(parts).rstrip("{:").rstrip()


def _bracket_depth(line: str) -> int:
    """Net opening brackets on a line, ignoring anything inside a string."""
    depth = 0
    quote = ""
    for character in line:
        if quote:
            if character == quote:
                quote = ""
        elif character in "\"'":
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
    return depth


def _largest_prefix_that_fits(entries: Sequence[_Entry], max_tokens: int) -> int:
    """How many of the top-ranked files fit, by binary search."""
    if max_tokens <= 0 or not entries:
        return 0

    low, high = 0, len(entries)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens("\n\n".join(entry.text for entry in entries[:middle])) <= max_tokens:
            low = middle
        else:
            high = middle - 1

    return max(low, 1)


def _degraded(index: RepositoryIndex, max_tokens: int) -> RepoMap:
    """A directory tree, for a repository with no definitions to map."""
    directories: dict[str, list[str]] = {}
    for path in index.paths:
        directory, _, name = path.rpartition("/")
        directories.setdefault(directory or ".", []).append(name)

    lines = [_DEGRADED_HEADER, ""]
    for directory in sorted(directories):
        lines.append(f"{directory}/")
        lines.extend(f"  {name}" for name in sorted(directories[directory]))

    whole = "\n".join(lines)
    text = truncate_to_tokens(whole, max_tokens)

    return RepoMap(
        text=text,
        files_shown=text.count("\n  ") if text != whole else len(index),
        files_total=len(index),
        tokens=estimate_tokens(text),
        degraded=True,
    )


def definitions_in_map(analysis: FileAnalysis) -> tuple[Definition, ...]:
    """The definitions this file would contribute. Exposed for tests and tooling."""
    return tuple(
        definition for definition in analysis.definitions if definition.role in MAPPED_ROLES
    )
