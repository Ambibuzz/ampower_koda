# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Orchestrates parsing + doctype walking into a single resolved Graph.

from __future__ import annotations

import datetime
import os
import subprocess

from ampower_koda.agent.kg.doctype_walker import parse_all_doctypes
from ampower_koda.agent.kg.models import Edge, Graph, Node
from ampower_koda.agent.kg.parser import TreeSitterParser, language_for_file

IGNORED_DIR_NAMES = {".git", "__pycache__", "node_modules", ".ampower_koda"}


class GraphBuilder:
    def __init__(self):
        self._parsers: dict[str, TreeSitterParser] = {}

    def _parser_for(self, language: str) -> TreeSitterParser:
        if language not in self._parsers:
            self._parsers[language] = TreeSitterParser(language)
        return self._parsers[language]

    def _list_source_files(self, repo_root: str, app_name: str) -> list[str]:
        """List tracked .py/.js files under the app's own module directory,
        respecting git tracking (skips .gitignore'd / untracked build artifacts)."""
        app_module_root = os.path.join(repo_root, app_name)
        if not os.path.isdir(app_module_root):
            app_module_root = repo_root

        try:
            result = subprocess.run(
                ["git", "ls-files", "*.py", "*.js"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                tracked = [
                    os.path.join(repo_root, line.strip())
                    for line in result.stdout.splitlines()
                    if line.strip()
                ]
                return [f for f in tracked if os.path.isfile(f)]
        except Exception:
            pass

        files = []
        for dirpath, dirnames, filenames in os.walk(repo_root):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIR_NAMES]
            for fname in filenames:
                if fname.endswith((".py", ".js")):
                    files.append(os.path.join(dirpath, fname))
        return files

    def build(self, repo_root: str, app_name: str, files: list[str] | None = None) -> Graph:
        """Build a full (or partial, if `files` is given) knowledge graph.
        `commit_sha` is set by the caller after building."""
        graph = Graph(
            app=app_name,
            commit_sha="",
            built_at=datetime.datetime.utcnow().isoformat(),
        )

        source_files = files if files is not None else self._list_source_files(repo_root, app_name)

        all_raw_captures: list[tuple[str, dict]] = []

        for abs_path in source_files:
            language = language_for_file(abs_path)
            if not language:
                continue
            rel_path = os.path.relpath(abs_path, repo_root)
            parser = self._parser_for(language)
            try:
                nodes, raw_captures = parser.parse_file(abs_path, rel_path, app_name)
            except Exception:
                continue

            file_node = Node(
                id=f"{app_name}:{rel_path}",
                type="File",
                name=os.path.basename(rel_path),
                file_path=rel_path,
            )
            graph.add_node(file_node)

            seen_whitelisted_at: dict[tuple[str, int], Node] = {}
            for node in nodes:
                if node.type == "WhitelistedAPI":
                    node.id = f"{node.id}#whitelisted"
                    seen_whitelisted_at[(node.name, node.line_start)] = node

                graph.add_node(node)
                graph.add_edge(Edge(source_id=file_node.id, target_id=node.id, type="defines"))

            for n in nodes:
                if n.type == "Function":
                    key = (n.name, n.line_start)
                    if key in seen_whitelisted_at:
                        graph.add_edge(Edge(
                            source_id=seen_whitelisted_at[key].id,
                            target_id=n.id,
                            type="registered_in",
                        ))

            for cap in raw_captures:
                all_raw_captures.append((rel_path, cap))

        doctype_nodes = parse_all_doctypes(repo_root, app_name)
        for dt_node in doctype_nodes:
            graph.add_node(dt_node)

        self._resolve_call_edges(graph, all_raw_captures)
        self._resolve_doctype_usage_edges(graph, all_raw_captures)

        return graph

    def _resolve_call_edges(self, graph: Graph, all_raw_captures: list[tuple[str, dict]]) -> None:
        name_to_ids: dict[str, list[str]] = {}
        for node_id, node in graph.nodes.items():
            if node.type in ("Function", "Method"):
                name_to_ids.setdefault(node.name, []).append(node_id)

        for rel_path, cap in all_raw_captures:
            if cap["capture"] != "call.name":
                continue
            called_name = cap["text"]
            targets = name_to_ids.get(called_name)
            if not targets:
                continue
            file_id = None
            for nid, n in graph.nodes.items():
                if n.type == "File" and n.file_path == rel_path:
                    file_id = nid
                    break
            if not file_id:
                continue
            for target_id in targets:
                graph.add_edge(Edge(source_id=file_id, target_id=target_id, type="calls"))

    def _resolve_doctype_usage_edges(self, graph: Graph, all_raw_captures: list[tuple[str, dict]]) -> None:
        name_to_id = {n.name: nid for nid, n in graph.nodes.items() if n.type == "DocType"}

        for rel_path, cap in all_raw_captures:
            if cap["capture"] != "call.doctype_arg":
                continue
            doctype_name = cap["text"]
            target_id = name_to_id.get(doctype_name)
            if not target_id:
                continue
            file_id = None
            for nid, n in graph.nodes.items():
                if n.type == "File" and n.file_path == rel_path:
                    file_id = nid
                    break
            if not file_id:
                continue
            graph.add_edge(Edge(source_id=file_id, target_id=target_id, type="uses_doctype"))
