# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Persistent Knowledge Graph storage — replaces the Redis TTL cache.
# graph_json (Code field) is used for payloads under MAX_INLINE_GRAPH_BYTES;
# larger payloads are gzip-compressed and stored as a File attachment instead.

from __future__ import annotations

import gzip

import frappe
from frappe.model.document import Document

from ampower_koda.agent.kg.models import Graph

MAX_INLINE_GRAPH_BYTES = 2 * 1024 * 1024  # 2MB — above this, use graph_file instead of graph_json


class KodaKnowledgeGraph(Document):
    """Controller for a single persisted knowledge graph snapshot, keyed by
    app_name + commit_sha. Enforces that exactly one of graph_json or
    graph_file is populated for a Ready document, and keeps graph_key in
    sync as the uniqueness constraint for that key pair.
    """
    
    def autoname(self):
        """Names the document KG-{app_name}-{first 8 chars of commit_sha}."""
        commit_prefix = (self.commit_sha or "")[:8]
        self.name = f"KG-{self.app_name}-{commit_prefix}"

    def before_insert(self):
        """Sets graph_key, the unique constraint field, before the first insert."""
        self.graph_key = f"{self.app_name}::{self.commit_sha}"

    def validate(self):
        """Keeps graph_key in sync with app_name/commit_sha, and enforces that
        a Ready document has exactly one of graph_json or graph_file set.
        """
        self.graph_key = f"{self.app_name}::{self.commit_sha}"

        if self.graph_json and self.graph_file:
            frappe.throw("Only one of Graph JSON or Graph File should be set, not both.")

        if self.status == "Ready" and not self.graph_json and not self.graph_file:
            frappe.throw("A Ready knowledge graph must have graph data (Graph JSON or Graph File).")


# ---------------------------------------------------------------------------
# Module-level helpers.
#
# These are the shared entry points used both by this controller's own logic
# and by ampower_koda.agent.kg.store.KGGraphStore (the executor-facing store
# class that replaces RedisGraphStore). Keeping the save/load logic here
# rather than duplicated in store.py means there's a single place that knows
# about the inline-vs-attachment storage decision.
# ---------------------------------------------------------------------------

def get_graph_doc(app_name: str, commit_sha: str):
    """Return the KodaKnowledgeGraph doc for this app+commit, or None if none exists."""
    graph_key = f"{app_name}::{commit_sha}"
    name = frappe.db.get_value("Koda Knowledge Graph", {"graph_key": graph_key}, "name")
    if not name:
        return None
    return frappe.get_doc("Koda Knowledge Graph", name)


def save_graph(app_name: str, commit_sha: str, graph: Graph, status: str = "Ready") -> str:
    """Create or update the KodaKnowledgeGraph doc for this app+commit with the
    given Graph object. Returns the doc name.

    Routes to a gzip-compressed file attachment instead of the inline Code
    field when the serialized payload exceeds MAX_INLINE_GRAPH_BYTES. Large
    payloads require the document to exist (and have a name) before a file
    can be attached to it, so a new large-graph doc is first inserted with
    status="Building" before the attachment step.
    """
    payload = graph.to_json()  # bytes, via orjson (Graph.to_json in models.py)
    large = len(payload) > MAX_INLINE_GRAPH_BYTES

    doc = get_graph_doc(app_name, commit_sha) or frappe.new_doc("Koda Knowledge Graph")
    doc.app_name = app_name
    doc.commit_sha = commit_sha
    doc.built_at = graph.built_at
    doc.node_count = len(graph.nodes)
    doc.edge_count = len(graph.edges)
    doc.schema_version = 1
    doc.error_message = ""

    if not large:
        doc.graph_json = payload.decode("utf-8")
        doc.graph_file = None
        doc.status = status
        _insert_or_save(doc)
    else:
        if doc.is_new():
            # Need a name before we can attach a file to this doc.
            doc.status = "Building"
            doc.graph_json = None
            doc.graph_file = None
            doc.insert(ignore_permissions=True)
        else:
            _delete_attached_file(doc)

        compressed = gzip.compress(payload)
        file_doc = frappe.get_doc({
            "doctype": "File",
            "file_name": f"{doc.name}.json.gz",
            "attached_to_doctype": "Koda Knowledge Graph",
            "attached_to_name": doc.name,
            "content": compressed,
            "is_private": 1,
        })
        file_doc.insert(ignore_permissions=True)

        doc.graph_json = None
        doc.graph_file = file_doc.file_url
        doc.status = status
        doc.save(ignore_permissions=True)

    frappe.db.commit()
    return doc.name


def load_graph(app_name: str, commit_sha: str) -> Graph | None:
    """Load and deserialize the Graph for this app+commit. Returns None if no
    Ready graph exists for that key — mirrors RedisGraphStore.load()'s
    None-on-miss contract so callers don't need to change.
    """
    doc = get_graph_doc(app_name, commit_sha)
    if not doc or doc.status != "Ready":
        return None

    try:
        if doc.graph_file:
            raw = gzip.decompress(_read_attached_file(doc.graph_file))
        else:
            raw = (doc.graph_json or "").encode("utf-8")
        if not raw:
            return None
        return Graph.from_json(raw)
    except Exception:
        frappe.log_error(frappe.get_traceback(), "Koda Knowledge Graph Load Error")
        return None


@frappe.whitelist()
def get_graph_payload(name: str):
    """Whitelisted endpoint used by koda_knowledge_graph.js. Returns the graph
    as a plain dict regardless of whether it's stored inline or as a
    compressed attachment, so the client never needs to know which.
    """
    doc = frappe.get_doc("Koda Knowledge Graph", name)
    graph = load_graph(doc.app_name, doc.commit_sha)
    if not graph:
        return None
    return graph.to_dict()


def _insert_or_save(doc):
    """Inserts a new document or saves an existing one, whichever applies."""
    if doc.is_new():
        doc.insert(ignore_permissions=True)
    else:
        doc.save(ignore_permissions=True)


def _read_attached_file(file_url: str) -> bytes:
    """Returns the raw byte content of a File document given its file_url."""

    file_doc = frappe.get_doc("File", {"file_url": file_url})
    return file_doc.get_content()


def _delete_attached_file(doc):
    """Deletes the File attachment referenced by doc.graph_file, if any.
    Best-effort: a failed deletion is logged implicitly via the exception
    being swallowed rather than raised, since a dangling file is a minor
    leak and not worth failing the caller's save over.
    """
    if not doc.graph_file:
        return
    try:
        file_doc = frappe.get_doc("File", {"file_url": doc.graph_file})
        file_doc.delete(ignore_permissions=True)
    except Exception:
        # Best-effort cleanup; a dangling file is a minor leak, not worth
        # failing the save over.
        pass
