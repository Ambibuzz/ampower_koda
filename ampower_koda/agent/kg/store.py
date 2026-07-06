# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Persistent knowledge graph storage, backed by the Koda Knowledge Graph
# DocType and looked up by (app_name, commit_sha).

from __future__ import annotations

from ampower_koda.agent.kg.models import Graph


class KGGraphStore:
    """Persistent replacement for RedisGraphStore, backed by the
    'Koda Knowledge Graph' DocType instead of a TTL cache. Same save/load
    shape (load() returns None on miss) so callers barely change — the only
    difference is save()/load() now take (app_name, commit_sha) instead of
    a single opaque cache_key string, since the DocType is looked up by
    that pair rather than by a Redis key."""

    def save(self, app_name: str, commit_sha: str, graph: Graph, status: str = "Ready") -> str:
        """Persists the graph for this app+commit and returns the resulting
        Koda Knowledge Graph document name."""
        from ampower_koda.ampower_koda.doctype.koda_knowledge_graph.koda_knowledge_graph import save_graph
        return save_graph(app_name, commit_sha, graph, status=status)

    def load(self, app_name: str, commit_sha: str) -> Graph | None:
        """Returns the persisted Graph for this app+commit, or None if no
        Ready graph exists for that key."""
        from ampower_koda.ampower_koda.doctype.koda_knowledge_graph.koda_knowledge_graph import load_graph
        return load_graph(app_name, commit_sha)

