# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Agent tools for reading/writing and searching the target app codebase

import os
import re

import frappe


def _app_root(app_name: str) -> str:
    """Return the root path of the given Frappe app."""
    if not app_name:
        frappe.throw("Target App Name is required")
    return frappe.get_app_path(app_name)


def _resolve_path(app_name: str, relative_path: str) -> str:
    """Resolve path relative to app root. Prevent directory traversal."""
    root = _app_root(app_name)
    path = os.path.normpath(os.path.join(root, relative_path.lstrip("/")))
    if not path.startswith(root):
        raise ValueError(f"Path outside app: {relative_path}")
    return path


def list_directory(app_name: str, path: str) -> str:
    """List files and directories at the given path (relative to app root)."""
    try:
        full = _resolve_path(app_name, path)
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


IGNORE_DIRS = {
    "__pycache__", "node_modules", ".git", ".github", ".vscode",
    ".eggs", "*.egg-info", "dist", "build",
}


def find_files(app_name: str, pattern: str = "", max_depth: int = 6) -> str:
    """Recursively list the entire directory tree of the app (or a subtree).
    Returns an indented tree view. Use pattern to filter by filename glob.
    This gives you a bird's-eye view of the whole codebase structure in one call."""
    try:
        import fnmatch
        root = _app_root(app_name)
        lines = []
        count = 0
        max_entries = 1500

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in sorted(dirnames)
                if d not in IGNORE_DIRS and not d.endswith(".egg-info")
            ]

            rel_dir = os.path.relpath(dirpath, root)
            depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
            if depth > max_depth:
                dirnames.clear()
                continue

            indent = "  " * depth
            dir_name = os.path.basename(dirpath) if rel_dir != "." else "."
            lines.append(f"{indent}{dir_name}/")
            count += 1

            for fname in sorted(filenames):
                if pattern and not fnmatch.fnmatch(fname, pattern):
                    continue
                lines.append(f"{indent}  {fname}")
                count += 1
                if count >= max_entries:
                    lines.append(f"\n... (truncated at {max_entries} entries)")
                    return "\n".join(lines)

        return "\n".join(lines) if lines else "(empty)"
    except Exception as ex:
        return f"Error: {ex}"


def read_file(app_name: str, path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a file with line numbers. Path is relative to app root.
    If start_line and end_line are given, reads only that range (1-indexed, inclusive).
    Otherwise reads the full file. Line numbers help you reference exact locations for edits."""
    try:
        full = _resolve_path(app_name, path)
        if not os.path.isfile(full):
            return f"Not a file: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total = len(all_lines)
        if start_line > 0 and end_line > 0:
            s = max(0, start_line - 1)
            e = min(total, end_line)
            lines = all_lines[s:e]
            numbered = [f"{s + i + 1:5d} | {line.rstrip()}" for i, line in enumerate(lines)]
            header = f"[{path}] lines {s+1}-{e} of {total}"
            return header + "\n" + "\n".join(numbered)

        if total > 500:
            numbered = [f"{i+1:5d} | {line.rstrip()}" for i, line in enumerate(all_lines)]
            return f"[{path}] {total} lines total\n" + "\n".join(numbered)

        numbered = [f"{i+1:5d} | {line.rstrip()}" for i, line in enumerate(all_lines)]
        return f"[{path}] {total} lines\n" + "\n".join(numbered)
    except Exception as ex:
        return f"Error: {ex}"


def search_code(app_name: str, pattern: str, path: str = "") -> str:
    """Search for a regex pattern in files under the given path (relative to app root).
    Returns matches with file path, line number, and surrounding context lines."""
    try:
        root = _resolve_path(app_name, path) if path else _app_root(app_name)
        if path and not os.path.isdir(root):
            return f"Not a directory: {path}"
        regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
        app_root = _app_root(app_name)
        results = []
        CONTEXT_LINES = 3
        MAX_RESULTS = 80

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {
                "__pycache__", "node_modules", ".git", ".eggs"
            }]
            for name in filenames:
                if name.endswith((".py", ".js", ".json", ".html", ".md", ".txt", ".css")):
                    full = os.path.join(dirpath, name)
                    try:
                        with open(full, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                    except Exception:
                        continue

                    rel = os.path.relpath(full, app_root)
                    for i, line in enumerate(lines):
                        if regex.search(line):
                            start = max(0, i - CONTEXT_LINES)
                            end = min(len(lines), i + CONTEXT_LINES + 1)
                            context = []
                            for j in range(start, end):
                                marker = ">>>" if j == i else "   "
                                context.append(f"  {marker} {j + 1:4d} | {lines[j].rstrip()}")
                            results.append(f"{rel}:{i + 1}\n" + "\n".join(context))

                            if len(results) >= MAX_RESULTS:
                                results.append(f"\n... (stopped at {MAX_RESULTS} matches)")
                                return "\n\n".join(results)

        return "\n\n".join(results) if results else f"No matches for: {pattern}"
    except Exception as ex:
        return f"Error: {ex}"


def write_file(app_name: str, path: str, content: str) -> str:
    """Write or overwrite a file. Path is relative to app root."""
    try:
        full = _resolve_path(app_name, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"WRITE_OK: Wrote {len(content)} chars to {path}"
    except Exception as ex:
        return f"WRITE_FAILED: Error: {ex}"


def edit_file(app_name: str, path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in the file (first occurrence).
    IMPORTANT: old_string must match exactly (same whitespace, newlines, indentation)."""
    try:
        full = _resolve_path(app_name, path)
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


def replace_lines(app_name: str, path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace lines start_line through end_line (1-indexed, inclusive) with new_content.
    This is more reliable than edit_file for large files.
    Use read_file first to see the exact line numbers, then replace the target range."""
    try:
        full = _resolve_path(app_name, path)
        if not os.path.isfile(full):
            return f"EDIT_FAILED: Not a file: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total = len(lines)
        if start_line < 1 or end_line < start_line or start_line > total:
            context_start = max(0, min(start_line, total) - 5)
            context_end = min(total, max(start_line, end_line) + 5)
            context_preview = "".join(
                f"  {j+1:5d} | {lines[j].rstrip()}\n"
                for j in range(context_start, context_end) if j < total
            )
            return (
                f"EDIT_FAILED: Invalid line range {start_line}-{end_line}. "
                f"File has {total} lines. Nearby content:\n{context_preview}"
                f"Re-read the file with read_file to get correct line numbers."
            )
        end_line = min(end_line, total)

        before = lines[:start_line - 1]
        after = lines[end_line:]
        if not new_content.endswith("\n"):
            new_content += "\n"
        new_lines = new_content.splitlines(True)

        result = before + new_lines + after
        with open(full, "w", encoding="utf-8") as f:
            f.writelines(result)

        return (
            f"EDIT_OK: Replaced lines {start_line}-{end_line} in {path} "
            f"({end_line - start_line + 1} old lines → {len(new_lines)} new lines). "
            f"File now has {len(result)} lines."
        )
    except Exception as ex:
        return f"EDIT_FAILED: Error: {ex}"


def insert_lines(app_name: str, path: str, after_line: int, new_content: str) -> str:
    """Insert new_content AFTER the specified line number (1-indexed).
    Use after_line=0 to insert at the very beginning of the file."""
    try:
        full = _resolve_path(app_name, path)
        if not os.path.isfile(full):
            return f"EDIT_FAILED: Not a file: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total = len(lines)
        if after_line < 0 or after_line > total:
            return f"EDIT_FAILED: Invalid line {after_line}. File has {total} lines."

        if not new_content.endswith("\n"):
            new_content += "\n"
        new_lines = new_content.splitlines(True)

        result = lines[:after_line] + new_lines + lines[after_line:]
        with open(full, "w", encoding="utf-8") as f:
            f.writelines(result)

        return (
            f"EDIT_OK: Inserted {len(new_lines)} lines after line {after_line} in {path}. "
            f"File now has {len(result)} lines."
        )
    except Exception as ex:
        return f"EDIT_FAILED: Error: {ex}"


def get_file_outline(app_name: str, path: str) -> str:
    """Return a lightweight outline of a Python/JS file — class and function signatures with line numbers.
    Much cheaper than reading the whole file. Use this to understand a file's structure before reading specific sections."""
    try:
        full = _resolve_path(app_name, path)
        if not os.path.isfile(full):
            return f"Not a file: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        outline = []
        total = len(lines)
        is_py = path.endswith(".py")
        is_js = path.endswith(".js") or path.endswith(".ts")

        if is_py:
            for i, line in enumerate(lines):
                stripped = line.rstrip()
                lstrip = line.lstrip()
                if (lstrip.startswith("class ") or lstrip.startswith("def ")
                        or lstrip.startswith("async def ")
                        or lstrip.startswith("@frappe.whitelist")
                        or lstrip.startswith("@property")
                        or (lstrip.startswith("import ") and i < 30)
                        or (lstrip.startswith("from ") and i < 30)):
                    indent = len(line) - len(lstrip)
                    outline.append(f"{i+1:5d} | {'  ' * (indent // 4)}{stripped.strip()}")
        elif is_js:
            in_class = False
            for i, line in enumerate(lines):
                stripped = line.rstrip()
                lstrip = line.lstrip()
                indent = len(line) - len(lstrip)

                if re.match(r'^(export\s+)?class\s', lstrip):
                    in_class = True
                    outline.append(f"{i+1:5d} | {stripped.strip()}")
                elif re.match(r'^(export\s+)?(function|const|let|var)\s', lstrip):
                    outline.append(f"{i+1:5d} | {stripped.strip()}")
                elif re.match(r'^frappe\.(ui\.form\.on|listview_settings|call|pages)', lstrip):
                    outline.append(f"{i+1:5d} | {stripped.strip()}")
                elif re.match(r'^[a-zA-Z_$]+\s*[:=]\s*function', lstrip):
                    outline.append(f"{i+1:5d} | {stripped.strip()}")
                elif re.match(r'^[a-zA-Z_$]+\s*\(', lstrip) and indent == 0:
                    outline.append(f"{i+1:5d} | {stripped.strip()}")
                elif in_class and indent <= 4 and re.match(r'^(async\s+)?[a-zA-Z_$]+\s*\(', lstrip):
                    outline.append(f"{i+1:5d} |   {stripped.strip()}")
                elif re.match(r'^\$\(|^jQuery\(', lstrip) and '.on(' in lstrip:
                    outline.append(f"{i+1:5d} | {stripped.strip()[:100]}")
        else:
            return f"Outline not supported for this file type. Use read_file instead. ({total} lines)"

        if not outline:
            return f"[{path}] {total} lines — no class/function signatures found. Use read_file to inspect."

        return f"[{path}] {total} lines — outline:\n" + "\n".join(outline)
    except Exception as ex:
        return f"Error: {ex}"


def read_doctype_schema(app_name: str, doctype_name: str) -> str:
    """Read the DocType JSON schema for a given DocType."""
    try:
        app_root = _app_root(app_name)
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
