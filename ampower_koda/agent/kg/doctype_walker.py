# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Parses Frappe DocType .json definitions into DocType nodes (no tree-sitter — plain JSON).

from __future__ import annotations

import json
import os

from ampower_koda.agent.kg.models import Node


def find_doctype_files(app_root: str) -> list[str]:
    """Walk the app directory and return absolute paths to every DocType definition JSON.
    A DocType folder always contains a JSON file named after the folder itself,
    e.g. .../doctype/ai_agent_request/ai_agent_request.json"""
    matches = []
    for dirpath, dirnames, filenames in os.walk(app_root):
        if os.path.basename(dirpath) == "__pycache__" or "node_modules" in dirpath:
            continue
        if os.sep + "doctype" + os.sep in dirpath + os.sep:
            folder_name = os.path.basename(dirpath)
            expected = f"{folder_name}.json"
            if expected in filenames:
                matches.append(os.path.join(dirpath, expected))
    return matches


def parse_doctype_file(abs_path: str, rel_path: str, app_name: str) -> Node | None:
    """Parse a single DocType JSON into a Node. Returns None if the file isn't
    a valid DocType definition (defensive — skips malformed or unrelated JSON)."""
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if data.get("doctype") != "DocType" or not data.get("name"):
        return None

    doctype_name = data["name"]
    fields = data.get("fields", [])
    field_summary = [
        {"fieldname": f.get("fieldname"), "fieldtype": f.get("fieldtype"), "options": f.get("options")}
        for f in fields
        if f.get("fieldname")
    ]
    links = [link.get("link_doctype") for link in data.get("links", []) if link.get("link_doctype")]

    node_id = f"{app_name}:{rel_path}:{doctype_name}"

    return Node(
        id=node_id,
        type="DocType",
        name=doctype_name,
        file_path=rel_path,
        line_start=0,
        line_end=0,
        metadata={
            "module": data.get("module"),
            "is_single": bool(data.get("issingle")),
            "is_child_table": bool(data.get("istable")),
            "fields": field_summary,
            "linked_doctypes": links,
        },
    )


def parse_all_doctypes(app_root: str, app_name: str) -> list[Node]:
    """Find and parse every DocType definition in the app."""
    nodes = []
    for abs_path in find_doctype_files(app_root):
        rel_path = os.path.relpath(abs_path, app_root)
        node = parse_doctype_file(abs_path, rel_path, app_name)
        if node:
            nodes.append(node)
    return nodes
