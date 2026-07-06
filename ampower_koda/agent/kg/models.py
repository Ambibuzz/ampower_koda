# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Core data model for the knowledge graph: nodes, edges, and the graph container.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import orjson


@dataclass
class Node:
    """A single entity in the knowledge graph — a file, function, class,
    DocType, or similar. id is unique within a Graph and formatted as
    "{app}:{relative_path}:{qualname}".
    """
    id: str            # "{app}:{relative_path}:{qualname}"
    type: str           # File|Module|Class|Function|Method|DocType|Hook|WhitelistedAPI|Import
    name: str
    file_path: str
    line_start: int = 0
    line_end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Returns a plain-dict representation, suitable for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        """Reconstructs a Node from its to_dict() representation."""
        return cls(
            id=data["id"],
            type=data["type"],
            name=data["name"],
            file_path=data["file_path"],
            line_start=data.get("line_start", 0),
            line_end=data.get("line_end", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Edge:
    """A directed relationship between two Node ids in the knowledge graph."""
    source_id: str
    target_id: str
    type: str           # defines|calls|imports|extends|uses_doctype|registered_in

    def to_dict(self) -> dict:
        """Returns a plain-dict representation, suitable for JSON serialization."""
        return {"source_id": self.source_id, "target_id": self.target_id, "type": self.type}

    @classmethod
    def from_dict(cls, data: dict) -> "Edge":
        """Reconstructs an Edge from its to_dict() representation."""
        return cls(source_id=data["source_id"], target_id=data["target_id"], type=data["type"])


@dataclass
class Graph:
    """The full knowledge graph for one app at one commit: a dict of Node
    keyed by id, plus a list of Edge between those ids."""
    app: str
    commit_sha: str
    built_at: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def to_dict(self) -> dict:
        return {
            "app": self.app,
            "commit_sha": self.commit_sha,
            "built_at": self.built_at,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self) -> bytes:
        return orjson.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        return cls(
            app=data["app"],
            commit_sha=data["commit_sha"],
            built_at=data["built_at"],
            nodes={k: Node.from_dict(v) for k, v in data.get("nodes", {}).items()},
            edges=[Edge.from_dict(e) for e in data.get("edges", [])],
        )

    @classmethod
    def from_json(cls, data: bytes | str) -> "Graph":
        """Deserializes a Graph from JSON bytes or a JSON string, as produced by to_json()."""
        parsed = orjson.loads(data)
        return cls.from_dict(parsed)
