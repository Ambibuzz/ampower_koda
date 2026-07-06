# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Tree-sitter based symbol extraction, driven by .scm query files.

from __future__ import annotations

import os

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
from tree_sitter import Language, Node as TSNode, Parser, Query, QueryCursor

from ampower_koda.agent.kg.models import Node

QUERIES_DIR = os.path.join(os.path.dirname(__file__), "queries")

_LANGUAGES = {
    "python": Language(tspython.language()),
    "javascript": Language(tsjavascript.language()),
}

_EXT_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
}


def language_for_file(file_path: str) -> str | None:
    _, ext = os.path.splitext(file_path)
    return _EXT_TO_LANGUAGE.get(ext)


class TreeSitterParser:
    """Parses a single file with tree-sitter and extracts Node objects using .scm queries."""

    def __init__(self, language: str):
        if language not in _LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        self.language_name = language
        self.language = _LANGUAGES[language]
        self.parser = Parser(self.language)
        self._queries = self._load_queries(language)

    def _load_queries(self, language: str) -> list[Query]:
        prefix = {"python": ["python_symbols", "python_calls"], "javascript": ["javascript_symbols"]}
        queries = []
        for name in prefix.get(language, []):
            path = os.path.join(QUERIES_DIR, f"{name}.scm")
            if os.path.exists(path):
                src = open(path).read()
                queries.append(Query(self.language, src))
        return queries

    def parse_file(self, abs_path: str, rel_path: str, app_name: str) -> tuple[list[Node], list[dict]]:
        """Returns (nodes, raw_captures). raw_captures is used by the builder for edge resolution
        (e.g. matching a 'call.name' capture to a Node.id in a second pass)."""
        with open(abs_path, "rb") as f:
            source = f.read()

        tree = self.parser.parse(source)
        root = tree.root_node

        nodes: list[Node] = []
        raw_captures: list[dict] = []

        for query in self._queries:
            cursor = QueryCursor(query)
            matches = cursor.matches(root)
            for _pattern_index, captures in matches:
                for capture_name, capture_nodes in captures.items():
                    for ts_node in capture_nodes:
                        text = source[ts_node.start_byte:ts_node.end_byte].decode("utf-8", errors="replace")
                        line_start = ts_node.start_point[0] + 1
                        line_end = ts_node.end_point[0] + 1

                        raw_captures.append({
                            "capture": capture_name,
                            "text": text,
                            "line_start": line_start,
                            "line_end": line_end,
                        })

                        node = self._capture_to_node(
                            capture_name, text, line_start, line_end, rel_path, app_name
                        )
                        if node:
                            nodes.append(node)

        return nodes, raw_captures

    def _capture_to_node(
        self, capture_name: str, text: str, line_start: int, line_end: int,
        rel_path: str, app_name: str,
    ) -> Node | None:
        type_map = {
            "function.name": "Function",
            "class.name": "Class",
            "whitelisted.name": "WhitelistedAPI",
        }
        node_type = type_map.get(capture_name)
        if not node_type:
            return None

        qualname = f"{rel_path}:{text}"
        node_id = f"{app_name}:{qualname}"
        metadata = {}
        if capture_name == "whitelisted.name":
            metadata["decorators"] = ["frappe.whitelist"]

        return Node(
            id=node_id,
            type=node_type,
            name=text,
            file_path=rel_path,
            line_start=line_start,
            line_end=line_end,
            metadata=metadata,
        )
