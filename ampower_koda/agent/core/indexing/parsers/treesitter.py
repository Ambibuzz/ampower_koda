"""One parser, driven by a grammar and a query file."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ...contracts.source import Span, split_lines
from ...contracts.symbols import (
    REFERENCE_KINDS,
    SYMBOL_ROLES,
    DefinitionSite,
    ParseResult,
    Reference,
)
from ...errors import CoreError, ParseError
from .grammars import Grammar, LanguageSpec, parser_identity, read_query

_DEFINITION = "definition."
_REFERENCE = "reference."
_INJECTION = "injection."
_NAME = "name"


@dataclass(frozen=True, slots=True)
class TreeSitterParser:
    """A :class:`~.protocol.LanguageParser` backed by one grammar and one query."""

    language: str
    extensions: tuple[str, ...]
    identity: str

    _grammar: Grammar
    _query: object
    """A compiled ``tree_sitter.Query``. Compiled once at build time: query
    compilation is the expensive part, and doing it per file would make cold
    start pay for it once per source file instead of once per session."""

    _injections: Mapping[str, TreeSitterParser] = field(default_factory=dict)

    def parse(self, path: str, text: str) -> ParseResult:
        definitions, references, recovered = self._extract(path, text, line_offset=0)
        return ParseResult(
            language=self.language,
            definitions=tuple(definitions),
            references=tuple(references),
            recovered=recovered,
        )

    def _extract(
        self,
        path: str,
        text: str,
        *,
        line_offset: int,
    ) -> tuple[list[DefinitionSite], list[Reference], bool]:
        """Run the query over ``text`` and shift every result by ``line_offset``."""
        from tree_sitter import Parser, QueryCursor

        try:
            tree = Parser(self._grammar.language).parse(text.encode("utf-8"))
            matches = QueryCursor(self._query).matches(tree.root_node)
        except Exception as exc:  # noqa: BLE001
            raise ParseError(path, f"{self.language} parse failed: {exc}") from exc

        total_lines = len(split_lines(text))
        best: dict[tuple[int, int], tuple[int, DefinitionSite]] = {}
        references: list[Reference] = []
        nested: list[tuple[list[DefinitionSite], list[Reference], bool]] = []
        recovered = tree.root_node.has_error

        for pattern, captures in matches:
            name_node = _first(captures, _NAME)
            if name_node is not None and not _symbol_text(name_node):
                continue

            for capture, nodes in captures.items():
                if capture.startswith(_INJECTION):
                    nested.append(self._inject(path, capture, nodes, line_offset))
                elif name_node is None:
                    continue
                elif capture.startswith(_DEFINITION):
                    site = self._definition(
                        capture,
                        _first(captures, capture),
                        name_node,
                        line_offset,
                        total_lines,
                    )
                    key = (name_node.start_byte, name_node.end_byte)
                    if key not in best or pattern < best[key][0]:
                        best[key] = (pattern, site)
                elif capture.startswith(_REFERENCE):
                    references.append(self._reference(capture, name_node, line_offset))

        definitions = [site for _, site in best.values()]
        for child_definitions, child_references, child_recovered in nested:
            definitions.extend(child_definitions)
            references.extend(child_references)
            recovered = recovered or child_recovered

        return definitions, _dedupe(references), recovered

    def _inject(
        self,
        path: str,
        capture: str,
        nodes: Sequence[object],
        line_offset: int,
    ) -> tuple[list[DefinitionSite], list[Reference], bool]:
        """Re-parse an embedded language, shifting its lines onto the host file."""
        target = self._injections.get(capture[len(_INJECTION) :])
        if target is None:
            return [], [], False

        definitions: list[DefinitionSite] = []
        references: list[Reference] = []
        recovered = False
        for node in nodes:
            found, used, hit = target._extract(  # noqa: SLF001
                path,
                node.text.decode("utf-8", errors="replace"),
                line_offset=line_offset + node.start_point[0],
            )
            definitions.extend(found)
            references.extend(used)
            recovered = recovered or hit
        return definitions, references, recovered

    def _definition(
        self,
        capture: str,
        extent_node: object,
        name_node: object,
        line_offset: int,
        total_lines: int,
    ) -> DefinitionSite:
        role = capture[len(_DEFINITION) :]
        if role not in SYMBOL_ROLES:
            raise CoreError(f"{self.language} query captures unknown role {role!r}")

        extent = _span(extent_node, line_offset, total_lines)
        name_line = name_node.start_point[0] + 1 + line_offset
        return DefinitionSite(
            name=_symbol_text(name_node),
            role=role,  # type: ignore[arg-type]
            extent=extent,
            name_line=min(max(name_line, extent.start), extent.end),
        )

    def _reference(
        self,
        capture: str,
        name_node: object,
        line_offset: int,
    ) -> Reference:
        kind = capture[len(_REFERENCE) :]
        if kind not in REFERENCE_KINDS:
            raise CoreError(f"{self.language} query captures unknown reference {kind!r}")
        return Reference(
            name=_symbol_text(name_node),
            kind=kind,  # type: ignore[arg-type]
            line=name_node.start_point[0] + 1 + line_offset,
        )


def build_parser(
    spec: LanguageSpec,
    grammar: Grammar,
    *,
    built: Mapping[str, TreeSitterParser] | None = None,
) -> TreeSitterParser:
    """Compile one language's query and bind its injections."""
    from tree_sitter import Query

    query_source = read_query(spec)
    available = built or {}

    return TreeSitterParser(
        language=spec.name,
        extensions=spec.extensions,
        identity=parser_identity(spec, grammar, query_source),
        _grammar=grammar,
        _query=Query(grammar.language, query_source),
        _injections={
            capture: available[language]
            for capture, language in spec.injections
            if language in available
        },
    )


def _span(node: object, line_offset: int, total_lines: int) -> Span:
    """Convert a node's point range into an inclusive 1-based line span."""
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    if node.end_point[1] == 0 and end > start:
        end -= 1

    start += line_offset
    end += line_offset
    end = min(max(end, start), max(total_lines + line_offset, start))
    return Span(start, end)


def _text(node: object) -> str:
    return node.text.decode("utf-8", errors="replace")


def _symbol_text(node: object) -> str:
    """A node's text as a *name*, with syntactic quoting removed."""
    text = _text(node)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _first(captures: Mapping[str, Sequence[object]], key: str) -> object | None:
    """The first node bound to ``key``, or ``None``."""
    nodes = captures.get(key)
    return nodes[0] if nodes else None


def _dedupe(references: list[Reference]) -> list[Reference]:
    """Collapse identical ``(name, kind, line)`` triples, preserving order."""
    seen: set[tuple[str, str, int]] = set()
    unique: list[Reference] = []
    for reference in references:
        key = (reference.name, reference.kind, reference.line)
        if key not in seen:
            seen.add(key)
            unique.append(reference)
    return unique
