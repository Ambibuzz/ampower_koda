# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Summarizes a Graph object for prompt injection. The existence check itself
# (does a cached graph exist for the current commit?) lives directly in
# executor.py — no doctype lookup needed, since the cache key is deterministic.

from __future__ import annotations

from ampower_koda.agent.kg.models import Graph


def summarize_graph(graph: Graph, max_doctypes: int = 15, max_apis: int = 15) -> str:
    """Produce a short, human-readable summary of the graph for prompt injection.
    Not the full graph — just enough to orient the LLM before it starts exploring,
    so it can target reads instead of scanning blind."""
    by_type: dict[str, list] = {}
    for node in graph.nodes.values():
        by_type.setdefault(node.type, []).append(node)

    lines = [
        f"This app already has a pre-built knowledge graph "
        f"({len(graph.nodes)} nodes, {len(graph.edges)} edges, "
        f"built from commit {graph.commit_sha[:8] if graph.commit_sha else 'unknown'}).",
        "Use this to target your exploration instead of scanning files blindly.",
        "",
    ]

    doctypes = by_type.get("DocType", [])
    if doctypes:
        lines.append(f"DocTypes ({len(doctypes)}):")
        for n in doctypes[:max_doctypes]:
            fields_count = len(n.metadata.get("fields", []))
            lines.append(f"  - {n.name} ({fields_count} fields) — {n.file_path}")
        if len(doctypes) > max_doctypes:
            lines.append(f"  ... and {len(doctypes) - max_doctypes} more")
        lines.append("")

    apis = by_type.get("WhitelistedAPI", [])
    if apis:
        lines.append(f"Whitelisted API endpoints ({len(apis)}):")
        for n in apis[:max_apis]:
            lines.append(f"  - {n.name}() — {n.file_path}:{n.line_start}")
        if len(apis) > max_apis:
            lines.append(f"  ... and {len(apis) - max_apis} more")
        lines.append("")

    classes = by_type.get("Class", [])
    if classes:
        lines.append(f"Classes ({len(classes)}):")
        for n in classes[:max_doctypes]:
            lines.append(f"  - {n.name} — {n.file_path}:{n.line_start}")
        if len(classes) > max_doctypes:
            lines.append(f"  ... and {len(classes) - max_doctypes} more")

    return "\n".join(lines)
