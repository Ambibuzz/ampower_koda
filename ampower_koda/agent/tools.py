# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Agent tools for reading/writing and searching the target app codebase

import os
import re
import site
import sys

import frappe

from ampower_koda.agent.errors import log_agent_error
from ampower_koda.agent.atomic import atomic_write, read_bytes
from ampower_koda.agent import checkpoint
from ampower_koda.agent.run_control import check_active
from ampower_koda.agent.core.constants import CACHE_DIRECTORY
from ampower_koda.agent.core.globs import compile_globs
from ampower_koda.agent.javascript_validation import globals_configs, validate_javascript_names



def command_environment() -> dict:
    """Resolve bench/Python and nvm Node in workers without a login-shell PATH."""
    env = os.environ.copy()
    # RQ started by a service does not inherit ~/.local/bin. Bench commonly
    # lives there, while Python entry points live beside the worker interpreter.
    python_bin = os.path.dirname(sys.executable)
    user_bin = os.path.join(site.getuserbase(), 'Scripts' if os.name == 'nt' else 'bin')
    additions = [path for path in (python_bin, user_bin) if os.path.isdir(path)]
    env['PATH'] = os.pathsep.join([*additions, env.get('PATH', '')])
    versions = os.path.join(env.get("NVM_DIR", os.path.expanduser("~/.nvm")), "versions", "node")
    if os.path.isdir(versions):
        candidates = sorted((name for name in os.listdir(versions) if re.fullmatch(r"v\d+(?:\.\d+)*", name)),
                            key=lambda name: tuple(map(int, name[1:].split("."))), reverse=True)
        for name in candidates:
            bin_dir = os.path.join(versions, name, "bin")
            if os.path.isfile(os.path.join(bin_dir, "node")):
                env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
                break
    return env


def _write_source(app_name, full, content, original, *, exclusive=False):
    path = os.path.relpath(full, os.path.realpath(_app_root(app_name))).replace("\\", "/")
    def prepare():
        check_active(reserve=5)
        checkpoint.write_intent(
            {path: original.decode("utf-8", errors="surrogateescape") if original is not None else None},
            {path: content.decode("utf-8", errors="surrogateescape")})
    atomic_write(full, content, expected=original, before_replace=prepare, exclusive=exclusive,
                 on_temporary=lambda temporary: checkpoint.temporary_file(temporary, path))


def _tool_error(tool: str, exc: Exception, message: str) -> str:
    log_agent_error(f"Agent Tool: {tool}", f"{exc}\n{frappe.get_traceback()}")
    return message


def _app_root(app_name: str) -> str:
    """Return the root path of the given Frappe app."""
    if not app_name:
        frappe.throw("Target App Name is required")
    return frappe.get_app_path(app_name)


def _resolve_path(app_name: str, relative_path: str) -> str:
    """Resolve path relative to app root. Prevent directory traversal."""
    root = os.path.realpath(_app_root(app_name))
    if os.path.isabs(relative_path) or os.path.splitdrive(relative_path)[0]:
        raise ValueError(f"Path must be relative to app: {relative_path}")
    path = os.path.realpath(os.path.join(root, relative_path))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(f"Path outside app: {relative_path}")
    if _archived(os.path.relpath(path, root)):
        raise ValueError(f"{relative_path} holds tests archived from earlier runs; they are not part of this request.")
    return path


#: Tests left by earlier runs are archived here; they are not part of the current request.
ARCHIVE_DIRECTORY = ".koda/archive"
ARCHIVE_PARTS = tuple(ARCHIVE_DIRECTORY.split("/"))
#: The directory of Koda's own index caches: machine output that only ever matched as noise.
CACHE_PARTS = tuple(CACHE_DIRECTORY.split("/")[:2])


def _under(relative: str, parts: tuple[str, ...]) -> bool:
    return tuple(relative.replace("\\", "/").split("/")[:len(parts)]) == parts


def _archived(relative: str) -> bool:
    """Archived tests of earlier runs: no tool lists, searches or opens them."""
    return _under(relative, ARCHIVE_PARTS)


def _unlisted(relative: str) -> bool:
    """Paths a listing or search skips: archived tests and Koda's caches."""
    return _archived(relative) or _under(relative, CACHE_PARTS)


def _glob_matcher(glob: str):
    """Unanchored matcher for a glob relative to the searched directory; None for no glob.

    "wt/*.py" also matches "page/wt/a.py"; a glob without "/" matches the file name.
    """
    return compile_globs([glob.replace("\\", "/")], anchored=False) if glob and glob.strip() else None


def _glob_missed(glob: str, files: str) -> str:
    """The result when files existed but the glob excluded every one."""
    return (f"Glob {glob!r} matched no files; it excluded all {files}. A glob matches paths relative to the "
            "searched directory, at any depth: '*.py', 'doctype/**/*.json', '*.{py,js}'.")


def list_directory(app_name: str, path: str) -> str:
    """List files and directories at path (relative to app root).

    - Directories are prefixed with [DIR] in the output.
    - Files are listed with their names.
    """
    try:
        full = _resolve_path(app_name, path)
        if not os.path.isdir(full):
            return f"Not a directory: {path}"
        root = os.path.realpath(_app_root(app_name))
        entries = sorted(e for e in os.listdir(full) if not _unlisted(os.path.relpath(os.path.join(full, e), root)))
        lines = []
        for e in entries:
            p = os.path.join(full, e)
            prefix = "[DIR] " if os.path.isdir(p) else ""
            lines.append(prefix + e)
        return "\n".join(lines) if lines else "(empty)"
    except Exception as ex:
        return _tool_error("list_directory", ex, f"Error: {ex}")


IGNORE_DIRS = {
    "__pycache__", "node_modules", ".git", ".github", ".vscode",
    ".eggs", "dist", "build",
}


def _walked_dirs(dirpath: str, dirnames: list[str], app_root: str) -> list[str]:
    """The subdirectories of ``dirpath`` a listing or search descends into, sorted."""
    return sorted(d for d in dirnames if d not in IGNORE_DIRS and not d.endswith(".egg-info")
                  and not _unlisted(os.path.relpath(os.path.join(dirpath, d), app_root)))


def find_files(app_name: str, pattern: str = "", max_depth: int = 6) -> str:
    """List the app's files: an indented tree, or with ``pattern`` the matching paths, one per line.

    - pattern: optional glob on file paths relative to the app, matched at any
      depth (e.g. '*.py', 'doctype/**/*.json', '*.{py,js}').
    - max_depth: depth of recursion (defaults to 6).

    Call this FIRST to map the codebase structure before reading individual files.
    """
    try:
        root = _app_root(app_name)
        within = _glob_matcher(pattern)
        lines = []
        count = listed = excluded = 0
        max_entries = 1500

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = _walked_dirs(dirpath, dirnames, root)

            rel_dir = os.path.relpath(dirpath, root)
            depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
            if depth > max_depth:
                dirnames.clear()
                continue

            indent = "  " * depth
            if within is None:
                dir_name = os.path.basename(dirpath) if rel_dir != "." else "."
                lines.append(f"{indent}{dir_name}/")
                count += 1

            for fname in sorted(filenames):
                relative_file = os.path.relpath(os.path.join(dirpath, fname), root).replace(os.sep, '/')
                if within is not None and within(relative_file) is None:
                    excluded += 1
                    continue
                listed += 1
                # A filtered listing is flat: full paths, no directories without a match.
                lines.append(relative_file if within is not None else f"{indent}  {fname}")
                count += 1
                if count >= max_entries:
                    lines.append(f"\n... (truncated at {max_entries} entries)")
                    return "\n".join(lines)

        if excluded and not listed:
            return _glob_missed(pattern, f"{excluded} files in the app")
        return "\n".join(lines) if lines else "(empty)"
    except Exception as ex:
        return _tool_error("find_files", ex, f"Error: {ex}")


def read_file(app_name: str, path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a file (path relative to app root) with line numbers.

    - If start_line and end_line are both > 0, reads only that range (1-indexed, inclusive).
    - Otherwise reads the full file.

    Returns numbered lines (format: '    1 | content') so that line numbers can be used
    directly with replace_lines / insert_lines.
    """
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
        return _tool_error("read_file", ex, f"Error: {ex}")


def search_code(app_name: str, pattern: str, path: str = "") -> str:
    """Search for a regex pattern in the codebase.

    - pattern: A standard regex pattern to search for.
    - path: Optional directory filter (relative to app root).

    Returns matches with 3 lines of surrounding context and line numbers.
    """
    try:
        root = _resolve_path(app_name, path) if path else _app_root(app_name)
        if path and not os.path.isdir(root):
            return f"Not a directory: {path}"
        regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
        app_root = _app_root(app_name)
        results = []
        context_lines = 3
        max_results = 80

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
                    except Exception as e:
                        log_agent_error(
                            "Agent Tool: search_code read",
                            f"path={full}\n{e}\n{frappe.get_traceback()}",
                        )
                        continue

                    rel = os.path.relpath(full, app_root)
                    for i, line in enumerate(lines):
                        if regex.search(line):
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            context = []
                            for j in range(start, end):
                                marker = ">>>" if j == i else "   "
                                context.append(f"  {marker} {j + 1:4d} | {lines[j].rstrip()}")
                            results.append(f"{rel}:{i + 1}\n" + "\n".join(context))

                            if len(results) >= max_results:
                                results.append(f"\n... (stopped at {max_results} matches)")
                                return "\n\n".join(results)

        return "\n\n".join(results) if results else f"No matches for: {pattern}"
    except Exception as ex:
        return _tool_error("search_code", ex, f"Error: {ex}")


def write_file(app_name: str, path: str, content: str) -> str:
    """Write or overwrite a file at the given relative path.

    - Automatically creates parent directories if they don't exist.
    - Only use this for creating NEW files or completely replacing content.
    """
    try:
        full = _resolve_path(app_name, path)
        original = read_bytes(full)
        _write_source(app_name, full, content.encode("utf-8", errors="surrogateescape"), original)
        return f"WRITE_OK: Wrote {len(content)} chars to {path}"
    except Exception as ex:
        return _tool_error("write_file", ex, f"WRITE_FAILED: Error: {ex}")


def edit_file(app_name: str, path: str, old_string: str, new_string: str) -> str:
    """Replace the FIRST occurrence of old_string with new_string in a file.

    - path: relative to app root.
    - old_string: Must match EXACTLY, including whitespace and indentation.

    For multi-line edits or replacing specific ranges, use replace_lines instead.
    """
    try:
        full = _resolve_path(app_name, path)
        if not os.path.isfile(full):
            return f"EDIT_FAILED: Not a file: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if not old_string or content.count(old_string) != 1:
            preview = content[:3000]
            if len(content) > 3000:
                preview += f"\n... ({len(content)} chars total, showing first 3000)"
            return (
                f"EDIT_FAILED: old_string must match exactly once in {path} ({len(content)} chars, {content.count(chr(10))+1} lines). "
                f"Use read_file to inspect the file and copy the exact string to match, "
                f"including whitespace and indentation."
            )
        content = content.replace(old_string, new_string, 1)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"EDIT_OK: Updated {path} successfully."
    except Exception as ex:
        return _tool_error("edit_file", ex, f"EDIT_FAILED: Error: {ex}")


def replace_lines(app_name: str, path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace lines start_line through end_line (1-indexed, inclusive) with new_content.

    - Path: relative to app root.
    - Use read_file first to identify the exact line range.

    This is the PREFERRED tool for multi-line modifications.
    """
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
        if end_line > total:
            return (
                f"EDIT_FAILED: end_line {end_line} exceeds file length {total}. "
                f"Re-read the file with read_file to get correct line numbers."
            )

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
        return _tool_error("replace_lines", ex, f"EDIT_FAILED: Error: {ex}")


def insert_lines(app_name: str, path: str, after_line: int, new_content: str) -> str:
    """Insert new_content AFTER the specified 1-indexed line number.

    - path: relative to app root.
    - after_line: 1-indexed line number.
    - Use after_line=0 to insert at the very beginning of the file.
    """
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
        return _tool_error("insert_lines", ex, f"EDIT_FAILED: Error: {ex}")


def get_file_outline(app_name: str, path: str) -> str:
    """Extract class and function definitions from a file (.py, .js, .ts)."""
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
        return _tool_error("get_file_outline", ex, f"Error: {ex}")


def read_doctype_schema(app_name: str, doctype_name: str) -> str:
    """Read the JSON schema file for a Frappe DocType."""
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
        return _tool_error("read_doctype_schema", ex, f"Error: {ex}")


def _syntax_excerpt(line: str, column: int) -> str:
    """Keep the failing anchor even on a generated line thousands of characters long."""
    column = min(len(line), max(0, column))
    start = max(0, column - 140)
    end = min(len(line), column + 180)
    return (('…' if start else '') + line[start:end] + ('…' if end < len(line) else '')
            + '\n' + ' ' * (column - start + int(bool(start))) + '^')


def _javascript_syntax_error(path: str, output: str) -> str:
    lines = output.splitlines()
    reason = next((line.strip() for line in lines if re.match(r'^\w*Error:', line.strip())), 'JavaScript syntax check failed')
    for index, line in enumerate(lines[:-2]):
        location = re.search(r':(\d+)\s*$', line)
        if location:
            if '^' in lines[index + 2]:
                column = lines[index + 2].index('^')
                return (f'SYNTAX_ERROR in {path} at line {location.group(1)}, col {column + 1}: {reason}\n'
                        + _syntax_excerpt(lines[index + 1], column))
            # V8 truncates the caret line on very long source lines. Preserve
            # its actual line number without inventing an unreported column.
            source = lines[index + 1]
            excerpt = source if len(source) <= 400 else source[:180] + ' … ' + source[-220:]
            return f'SYNTAX_ERROR in {path} at line {location.group(1)}: {reason}\n{excerpt}'
    return f'SYNTAX_ERROR in {path}: {reason}\n{output[-800:]}'


def validate_code(app_name: str, path: str) -> str:
    """Check Python syntax or JavaScript syntax/undefined names (app-relative path)."""
    try:
        full = _resolve_path(app_name, path)
        if not os.path.isfile(full):
            return f"VALIDATION_FAILED: Not a file: {path}"

        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if path.endswith(".py"):
            try:
                import ast
                ast.parse(content)
                return f"VALID: {path} has no syntax errors."
            except SyntaxError as e:
                return (
                    f"SYNTAX_ERROR in {path} at line {e.lineno}, col {e.offset}: "
                    f"{e.msg}\n" + _syntax_excerpt((e.text or '').rstrip(), (e.offset or 1) - 1)
                )
            except Exception as e:
                log_agent_error(
                    "Agent Tool: validate_code python",
                    f"path={path}\n{e}\n{frappe.get_traceback()}",
                )
                return f"VALIDATION_ERROR: {e}"

        elif path.endswith(".js"):
            import subprocess
            env = command_environment()
            try:
                result = subprocess.run(
                    ["node", "--check", full],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env=env,
                )
                if result.returncode == 0:
                    return validate_javascript_names(path, content, configs=globals_configs(
                        full, _app_root(app_name), frappe.get_app_path("frappe")), env=env)
                return _javascript_syntax_error(path, result.stderr.strip() or result.stdout.strip())
            except FileNotFoundError:
                return "VALIDATION_UNAVAILABLE: Node.js is missing; JavaScript syntax was not checked."
            except subprocess.TimeoutExpired:
                return "VALIDATION_UNAVAILABLE: node --check timed out; JavaScript syntax was not checked."
            except Exception as e:
                log_agent_error(
                    "Agent Tool: validate_code javascript",
                    f"path={path}\n{e}\n{frappe.get_traceback()}",
                )
                return f"VALIDATION_ERROR: {e}"

        return f"SKIP: Validation not supported for this file type: {path}"
    except Exception as ex:
        return _tool_error("validate_code", ex, f"VALIDATION_FAILED: Error: {ex}")
