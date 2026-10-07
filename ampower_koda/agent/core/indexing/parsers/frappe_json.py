"""JSON, read as Frappe means it."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from ...contracts.source import Span, split_lines
from ...contracts.symbols import DefinitionSite, ParseResult, Reference
from ...errors import ParseError
from .grammars import Grammar

EXTRACTION_VERSION = "1"

_LINKING_FIELDTYPES = frozenset(
    {"Link", "Table", "Table MultiSelect", "Tree Select", "Dynamic Link"}
)

_LINK_KEYS = ("link_doctype", "link_to", "parent_doctype", "ref_doctype", "role")


@dataclass(frozen=True, slots=True)
class FrappeJsonParser:
    """A :class:`~.protocol.LanguageParser` for Frappe's JSON artefacts."""

    _grammar: Grammar
    language: str = "json"
    extensions: tuple[str, ...] = (".json",)

    @property
    def identity(self) -> str:
        return f"frappe-json/{self._grammar.version}/{EXTRACTION_VERSION}"

    def parse(self, path: str, text: str) -> ParseResult:
        from tree_sitter import Parser

        try:
            tree = Parser(self._grammar.language).parse(text.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ParseError(path, f"json parse failed: {exc}") from exc

        recovered = tree.root_node.has_error

        root = _first_object(tree.root_node)
        if root is None:
            return ParseResult(language=self.language, recovered=recovered)

        top = dict(_pairs(root))
        if "doctype" not in top:
            return ParseResult(language=self.language, recovered=recovered)

        total_lines = len(split_lines(text))
        definitions: list[DefinitionSite] = []
        references: list[Reference] = []

        record = _record_definition(top, total_lines)
        if record is not None:
            definitions.append(record)

        for node in _objects_from(root):
            fields = dict(_pairs(node))
            name = _string(fields.get("fieldname"))
            if name:
                definitions.append(
                    DefinitionSite(
                        name=name,
                        role="field",
                        extent=_span(node, total_lines),
                        name_line=_line(fields["fieldname"]),
                    )
                )
            references.extend(_references(fields))

        return ParseResult(
            language=self.language,
            definitions=tuple(definitions),
            references=tuple(_dedupe(references)),
            recovered=recovered,
        )


def _record_definition(top: dict[str, object], total_lines: int) -> DefinitionSite | None:
    """The record itself: ``Agent Request`` as a definition spanning the file."""
    name = _string(top.get("name"))
    if not name:
        return None
    return DefinitionSite(
        name=name,
        role="class",
        extent=Span(1, max(total_lines, 1)),
        name_line=_line(top["name"]),
    )


def _references(fields: dict[str, object]) -> Iterator[Reference]:
    """Every name in this object that points at another record."""
    fieldtype = _string(fields.get("fieldtype"))
    if fieldtype in _LINKING_FIELDTYPES:
        target = _string(fields.get("options"))
        if target:
            yield Reference(name=target, kind="class", line=_line(fields["options"]))

    for key in _LINK_KEYS:
        target = _string(fields.get(key))
        if target:
            yield Reference(name=target, kind="class", line=_line(fields[key]))


def _first_object(node: object) -> object | None:
    """The document's root object, or ``None`` for an array or a scalar."""
    for child in node.named_children:
        if child.type == "object":
            return child
    return None


def _pairs(node: object) -> Iterator[tuple[str, object]]:
    """Yield ``(key, value_node)`` for one object's direct pairs."""
    for child in node.named_children:
        if child.type != "pair":
            continue
        key = child.child_by_field_name("key")
        value = child.child_by_field_name("value")
        if key is not None and value is not None:
            text = _string(key)
            if text is not None:
                yield text, value


def _objects_from(node: object) -> Iterator[object]:
    """Yield ``node`` and every object beneath it, in document order."""
    if node.type == "object":
        yield node
    for child in node.named_children:
        yield from _objects_from(child)


def _string(node: object | None) -> str | None:
    """The text of a JSON string node, unescaped enough to be a symbol."""
    if node is None or getattr(node, "type", None) != "string":
        return None
    raw = node.text.decode("utf-8", errors="replace")
    return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\") if len(raw) >= 2 else None


def _line(node: object) -> int:
    return node.start_point[0] + 1


def _span(node: object, total_lines: int) -> Span:
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    if node.end_point[1] == 0 and end > start:
        end -= 1
    return Span(min(start, total_lines or 1), min(max(end, start), max(total_lines, 1)))


def _dedupe(references: list[Reference]) -> list[Reference]:
    seen: set[tuple[str, str, int]] = set()
    unique: list[Reference] = []
    for reference in references:
        key = (reference.name, reference.kind, reference.line)
        if key not in seen:
            seen.add(key)
            unique.append(reference)
    return unique
