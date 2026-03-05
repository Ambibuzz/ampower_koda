# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Agent tools for reading/writing and searching the target app codebase

import os
import re

import frappe


def _app_root() -> str:
    """Return the root path of the target app configured in AI Agents Settings."""
    settings = frappe.get_single("AI Agents Settings")
    app_name = (settings.target_app_name or "").strip()
    if not app_name:
        frappe.throw("Target App Name not set in AI Agents Settings")
    return frappe.get_app_path(app_name)


def _resolve_path(relative_path: str) -> str:
    """Resolve path relative to app root. Prevent directory traversal."""
    root = _app_root()
    path = os.path.normpath(os.path.join(root, relative_path.lstrip("/")))
    if not path.startswith(root):
        raise ValueError(f"Path outside app: {relative_path}")
    return path


def list_directory(path: str) -> str:
    """List files and directories at the given path (relative to app root).
    Example: list_directory('task_management/doctype')"""
    try:
        full = _resolve_path(path)
        if not os.path.isdir(full):
            return f"Not a directory: {path}"
        entries = sorted(os.listdir(full))
        lines = []
        for e in entries:
            p = os.path.join(full, e)
            prefix = "[DIR] " if os.path.isdir(p) else ""
            lines.append(prefix + e)
        return "\n".join(lines) if lines else "(empty)"
    except Exception as ex:
        return f"Error: {ex}"


def read_file(path: str) -> str:
    """Read the contents of a file. Path is relative to app root.
    Example: read_file('task_management/doctype/tm_task/tm_task.py')"""
    try:
        full = _resolve_path(path)
        if not os.path.isfile(full):
            return f"Not a file: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as ex:
        return f"Error: {ex}"


def search_code(pattern: str, path: str = "") -> str:
    """Search for a regex pattern in files under the given path (relative to app root).
    path can be a directory or empty to search the whole app.
    Example: search_code('def validate', 'task_management/doctype/tm_task')"""
    try:
        root = _resolve_path(path) if path else _app_root()
        if path and not os.path.isdir(root):
            return f"Not a directory: {path}"
        regex = re.compile(pattern, re.MULTILINE)
        app_root = _app_root()
        results = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.endswith((".py", ".js", ".json", ".html", ".md", ".txt", ".css")):
                    full = os.path.join(dirpath, name)
                    try:
                        with open(full, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except Exception:
                        continue
                    rel = os.path.relpath(full, app_root)
                    for m in regex.finditer(content):
                        start = max(0, m.start() - 60)
                        end = min(len(content), m.end() + 80)
                        snippet = content[start:end].replace("\n", " ")
                        results.append(f"{rel}: {snippet}")
        return "\n".join(results[:50]) if results else f"No matches for: {pattern}"
    except Exception as ex:
        return f"Error: {ex}"


def write_file(path: str, content: str) -> str:
    """Write or overwrite a file. Path is relative to app root.
    Use for new files or full file replacement."""
    try:
        full = _resolve_path(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"WRITE_OK: Wrote {len(content)} chars to {path}"
    except Exception as ex:
        return f"WRITE_FAILED: Error: {ex}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in the file (first occurrence).
    Path is relative to app root. Preserves formatting.
    IMPORTANT: old_string must match exactly (same whitespace, newlines, indentation).
    If it fails, the actual file content is returned so you can see what's there."""
    try:
        full = _resolve_path(path)
        if not os.path.isfile(full):
            return f"EDIT_FAILED: Not a file: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if old_string not in content:
            preview = content[:3000]
            if len(content) > 3000:
                preview += f"\n... ({len(content)} chars total, showing first 3000)"
            return (
                f"EDIT_FAILED: old_string not found in {path}. "
                f"The old_string you provided does not match any text in the file. "
                f"Here is the actual file content — use read_file or copy an exact "
                f"substring from below:\n\n{preview}"
            )
        content = content.replace(old_string, new_string, 1)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"EDIT_OK: Updated {path} successfully."
    except Exception as ex:
        return f"EDIT_FAILED: Error: {ex}"


def read_doctype_schema(doctype_name: str) -> str:
    """Read the DocType JSON schema for a given DocType.
    Searches all module directories in the target app for the DocType folder."""
    try:
        app_root = _app_root()
        name_lower = doctype_name.replace(" ", "_").lower()
        target_file = f"{name_lower}.json"
        for dirpath, _dirnames, filenames in os.walk(app_root):
            if os.path.basename(dirpath) == name_lower and target_file in filenames:
                full = os.path.join(dirpath, target_file)
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
        return f"DocType schema not found: {doctype_name}"
    except Exception as ex:
        return f"Error: {ex}"


TOOL_DEFINITIONS = [
    ("list_directory", list_directory, "List files and dirs at path (relative to app root). Input: path string."),
    ("read_file", read_file, "Read file content. Input: path relative to app root."),
    ("search_code", search_code, "Search for regex pattern in codebase. Input: pattern, optional path."),
    ("write_file", write_file, "Write or overwrite file. Input: path, content."),
    ("edit_file", edit_file, "Replace old_string with new_string in file (first occurrence). Input: path, old_string, new_string."),
    ("read_doctype_schema", read_doctype_schema, "Read DocType JSON schema. Input: doctype name e.g. TM Task."),
]
