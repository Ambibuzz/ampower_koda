"""The reference graph: which file leans on which, and how hard."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from math import log
from types import MappingProxyType

from ..constants import EDGE_WEIGHTS
from ..contracts.repository import RepositoryIndex
from ..contracts.source import Span

EdgeKind = str

_KIND_TO_EDGE: Mapping[str, EdgeKind] = MappingProxyType(
    {"call": "calls", "class": "instantiates", "type": "references"}
)


@dataclass(frozen=True, slots=True)
class Edge:
    """One weighted dependency from one file to another."""

    source: str
    target: str
    symbol: str
    kind: EdgeKind
    weight: float

    def __str__(self) -> str:
        return f"{self.source} -{self.kind}:{self.symbol}-> {self.target} ({self.weight:.3f})"


@dataclass(frozen=True, slots=True)
class CodeGraph:
    """A weighted, directed graph over files."""

    edges: tuple[Edge, ...] = ()
    outgoing: Mapping[str, tuple[Edge, ...]] = field(default_factory=dict)
    incoming: Mapping[str, tuple[Edge, ...]] = field(default_factory=dict)

    definition_sites: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """Symbol → the files that define it, in path order. Kept on the graph
    because both the ambiguity discount and the structural leg need it, and
    recomputing it per query over a four-thousand-file index is the kind of
    waste that only shows up under load."""

    def __post_init__(self) -> None:
        for name in ("outgoing", "incoming", "definition_sites"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every file that has an edge, in codepoint order."""
        endpoints = {edge.source for edge in self.edges} | {edge.target for edge in self.edges}
        return tuple(sorted(endpoints))

    def out_edges(self, path: str) -> tuple[Edge, ...]:
        return self.outgoing.get(path, ())

    def in_edges(self, path: str) -> tuple[Edge, ...]:
        return self.incoming.get(path, ())

    def defines(self, symbol: str) -> tuple[str, ...]:
        return self.definition_sites.get(symbol, ())

    def ambiguity_discount(self, symbol: str) -> float:
        """How much to believe an edge along ``symbol``."""
        return ambiguity_discount(len(self.definition_sites.get(symbol, ())))


def ambiguity_discount(definition_count: int) -> float:
    """``1 / (1 + ln(defs))``, and ``1.0`` for zero or one definition."""
    return 1.0 if definition_count <= 1 else 1.0 / (1.0 + log(definition_count))


def build_graph(index: RepositoryIndex) -> CodeGraph:
    """Build the reference graph from an analysed index."""
    definitions = _definition_sites(index)
    edges: list[Edge] = []

    for path in index.paths:
        seen: set[tuple[str, str, str]] = set()
        for reference in index.files[path].references:
            targets = definitions.get(reference.name)
            if not targets:
                continue

            kind = _KIND_TO_EDGE.get(reference.kind, "references")
            discount = ambiguity_discount(len(targets))
            weight = EDGE_WEIGHTS.get(kind, EDGE_WEIGHTS["references"]) * discount

            for target in targets:
                if target == path:
                    continue
                key = (target, reference.name, kind)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    Edge(
                        source=path,
                        target=target,
                        symbol=reference.name,
                        kind=kind,
                        weight=weight,
                    )
                )

    edges.extend(_containment_edges(index, definitions))
    return CodeGraph(
        edges=tuple(edges),
        outgoing=_group(edges, key=lambda edge: edge.source),
        incoming=_group(edges, key=lambda edge: edge.target),
        definition_sites=definitions,
    )


def _containment_edges(
    index: RepositoryIndex,
    definitions: Mapping[str, tuple[str, ...]],
) -> Iterator[Edge]:
    """Edges from a file to the files defining the base classes it extends."""
    for path in index.paths:
        emitted: set[tuple[str, str]] = set()
        for definition in index.files[path].definitions:
            if definition.role not in ("class", "enum"):
                continue
            for reference in index.files[path].references:
                if reference.kind != "class":
                    continue
                if not definition.extent.contains(Span(reference.line, reference.line)):
                    continue
                for target in definitions.get(reference.name, ()):
                    if target == path or (target, reference.name) in emitted:
                        continue
                    emitted.add((target, reference.name))
                    yield Edge(
                        source=path,
                        target=target,
                        symbol=reference.name,
                        kind="contains",
                        weight=EDGE_WEIGHTS["contains"]
                        * ambiguity_discount(len(definitions.get(reference.name, ()))),
                    )


def _definition_sites(index: RepositoryIndex) -> Mapping[str, tuple[str, ...]]:
    """Symbol → defining files, indexed under both the bare and qualified name."""
    table: dict[str, list[str]] = {}
    for path in index.paths:
        for definition in index.files[path].definitions:
            for name in {definition.name, definition.qualified_name}:
                bucket = table.setdefault(name, [])
                if path not in bucket:
                    bucket.append(path)
    return MappingProxyType({name: tuple(sorted(paths)) for name, paths in table.items()})


def _group(edges: list[Edge], *, key) -> dict[str, tuple[Edge, ...]]:
    grouped: dict[str, list[Edge]] = {}
    for edge in edges:
        grouped.setdefault(key(edge), []).append(edge)
    return {node: tuple(items) for node, items in grouped.items()}
