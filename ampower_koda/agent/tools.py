# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Agent tools for reading/writing and searching the target app codebase

import difflib
import hashlib
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

READ_WHOLE_FILE_LINES = 2000
# Longer code files are first shown as a summary: every read rides along in later requests.
SUMMARY_MIN_LINES = 200
# Runs between signatures up to this long are shown verbatim.
SUMMARY_KEEP_RUN = 6
MAX_READ_RANGES = 20


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


def _parse_ranges(spec: str, total: int) -> list[tuple[int, int]]:
    """``"40-80,120"`` -> merged, clamped 1-indexed inclusive spans."""
    spans = []
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        if not first.isdigit() or (last and not last.isdigit()):
            raise ValueError(f"Invalid range {part!r}: use start-end, e.g. 40-80,120-160")
        a, b = int(first), int(last or first)
        if a < 1 or b < a:
            raise ValueError(f"Invalid range {part!r}: lines are 1-indexed and start <= end")
        if a <= total:
            spans.append((a, min(b, total)))
    if len(spans) > MAX_READ_RANGES:
        raise ValueError(f"At most {MAX_READ_RANGES} ranges per read")
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _summary(path: str, all_lines: list[str]) -> str:
    """Signatures and short runs verbatim, long bodies elided; "" when that saves too little."""
    declared = {index for index, _ in _declarations(path, all_lines)}
    if not declared:
        return ""
    total, rows, shown, index = len(all_lines), [], 0, 0
    while index < total:
        if index in declared:
            rows.append(f"{index + 1:5d} | {all_lines[index].rstrip()}")
            shown, index = shown + 1, index + 1
            continue
        run_end = index
        while run_end < total and run_end not in declared:
            run_end += 1
        if run_end - index <= SUMMARY_KEEP_RUN:
            rows.extend(f"{i + 1:5d} | {all_lines[i].rstrip()}" for i in range(index, run_end))
            shown += run_end - index
        else:
            rows.append(f"      … lines {index + 1}-{run_end} elided")
        index = run_end
    if shown > total * 0.6:
        return ""
    return (f"[{path}] summary of {total} lines: signatures kept, long bodies elided\n" + "\n".join(rows)
            + f"\n[Read what you need in one call, e.g. ranges=\"40-80,120-160\"; ranges=\"1-{total}\" "
            "reads the whole file. Edit only lines you have read.]")


def _redaction_matcher(app_name: str):
    """(real app root, matcher) for the redaction the core index applies: its defaults plus the
    app's .koda/config.toml. A .env the index skips must not come back whole through a read."""
    from pathlib import Path

    from ampower_koda.agent.core.workspace.local import LocalWorkspace
    from ampower_koda.agent.core.workspace.redaction import redaction_matcher

    root = os.path.realpath(_app_root(app_name))
    return root, redaction_matcher(LocalWorkspace(root_path=Path(root)))


def redaction_pattern(app_name: str, path: str, full: str, matcher=None) -> str | None:
    """The redaction pattern the requested or resolved path matches, or None.

    ``matcher`` is ``_redaction_matcher``'s pair, passed by callers that check many files.
    """
    root, match = matcher or _redaction_matcher(app_name)
    requested = os.path.normpath(path).replace(os.sep, "/")
    return match(requested) or match(os.path.relpath(full, root).replace(os.sep, "/"))


def read_file(app_name: str, path: str, start_line: int = 0, end_line: int = 0, ranges: str = "") -> str:
    """Read a file (path relative to app root) with line numbers.

    - ``ranges`` ("40-80,120-160") reads several spans in one call.
    - If start_line and end_line are both > 0, reads only that range (1-indexed, inclusive).
    - Otherwise reads the full file, except that a long code file is summarized
      (signatures kept, long bodies elided) so the model reads only the ranges it needs.

    Returns numbered lines (format: '    1 | content') so that line numbers can be used
    when citing source.
    """
    try:
        full = _resolve_path(app_name, path)
        if not os.path.isfile(full):
            return f"Not a file: {path}"
        pattern = redaction_pattern(app_name, path, full)
        if pattern:
            return f"READ_FAILED: {path} matches the redaction pattern {pattern} (secrets are never read)"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total = len(all_lines)
        if not ranges and start_line > 0 and end_line > 0:
            ranges = f"{start_line}-{end_line}"
        if ranges:
            spans = _parse_ranges(ranges, total)
            if not spans:
                return f"[{path}] has {total} lines; no requested line exists."
            blocks = ["\n".join(f"{i + 1:5d} | {all_lines[i].rstrip()}" for i in range(a - 1, b))
                      for a, b in spans]
            label = ",".join(f"{a}-{b}" for a, b in spans)
            return f"[{path}] lines {label} of {total}\n" + "\n      …\n".join(blocks)

        if total >= SUMMARY_MIN_LINES:
            summary = _summary(path, all_lines)
            if summary:
                return summary

        # Read whole, not in slices: each slice resends the conversation, and
        # edits made from partial views miss their own dependencies.
        if total > READ_WHOLE_FILE_LINES:
            preview = all_lines[:READ_WHOLE_FILE_LINES]
            numbered = [f"{i+1:5d} | {line.rstrip()}" for i, line in enumerate(preview)]
            return (
                f"[{path}] lines 1-{len(preview)} of {total}\n"
                + "\n".join(numbered)
                + f"\n[Read start_line={len(preview) + 1} end_line={total} for the rest.]"
            )

        numbered = [f"{i+1:5d} | {line.rstrip()}" for i, line in enumerate(all_lines)]
        return f"[{path}] {total} lines\n" + "\n".join(numbered)
    except Exception as ex:
        return _tool_error("read_file", ex, f"Error: {ex}")


SEARCH_SUFFIXES = (".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".json", ".html", ".jinja",
                   ".j2", ".md", ".txt", ".css", ".scss", ".less", ".sql", ".toml", ".cfg", ".ini", ".yml",
                   ".yaml", ".xml")
SEARCH_SKIP_SUFFIXES = (".min.js", ".min.css", ".bundle.js")
SEARCH_MAX_FILE_BYTES = 2_000_000
SEARCH_LINE_CHARS = 240
SEARCH_MAX_CHARS = 20_000
SEARCH_DEFAULT_LIMIT = {"files": 50, "content": 40}
SEARCH_MAX_LIMIT = 200
SEARCH_MAX_CONTEXT = 8


def _search_files(app_name: str, root: str):
    """(full, app-relative) searchable files under ``root``, a file or directory, in path order."""
    # Built once per search: every walked file is checked against it.
    redaction = _redaction_matcher(app_name)
    app_root = redaction[0]
    if os.path.isfile(root):
        relative = os.path.relpath(root, app_root).replace(os.sep, "/")
        if not redaction_pattern(app_name, relative, root, redaction):
            yield root, relative
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = _walked_dirs(dirpath, dirnames, app_root)
        for name in sorted(filenames):
            if not name.endswith(SEARCH_SUFFIXES) or name.endswith(SEARCH_SKIP_SUFFIXES):
                continue
            full = os.path.join(dirpath, name)
            # A secrets.yml would otherwise print its lines as matches.
            if redaction_pattern(app_name, os.path.relpath(full, app_root), full, redaction):
                continue
            try:
                if os.path.getsize(full) > SEARCH_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield full, os.path.relpath(full, app_root).replace(os.sep, "/")


def _read_lines(full: str) -> list[str] | None:
    """A searched file's lines, or None, logged, when it cannot be read."""
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except Exception as e:
        log_agent_error("Agent Tool: search_code read", f"path={full}\n{e}\n{frappe.get_traceback()}")
        return None


def search_code(app_name: str, pattern: str, path: str = "", *, glob: str = "", output_mode: str = "",
                context: int = 2, head_limit: int = 0) -> str:
    """Search for a regular expression, case-insensitively, one line at a time.

    ``output_mode`` defaults to "content" for one file and "files" (match counts) otherwise.
    """
    try:
        root = _resolve_path(app_name, path) if path else os.path.realpath(_app_root(app_name))
        if path and not os.path.exists(root):
            return f"Not a file or directory: {path}"
        note = ""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
            note = f"[Not a valid regular expression ({error}); searched it as literal text.]\n"
        mode = (output_mode or ("content" if os.path.isfile(root) else "files")).strip().lower()
        if mode not in SEARCH_DEFAULT_LIMIT:
            return "SEARCH_FAILED: output_mode is 'files' or 'content'."
        limit = max(1, min(int(head_limit or 0) or SEARCH_DEFAULT_LIMIT[mode], SEARCH_MAX_LIMIT))
        context = max(0, min(int(context or 0), SEARCH_MAX_CONTEXT))
        within = _glob_matcher(glob) if os.path.isdir(root) else None

        # found holds (full, relative, matching line indexes) only; rendered files are read again.
        found, searched, excluded = [], 0, 0
        for full, relative in _search_files(app_name, root):
            if within is not None and within(os.path.relpath(full, root).replace(os.sep, "/")) is None:
                excluded += 1
                continue
            searched += 1
            hits = [i for i, line in enumerate(_read_lines(full) or ()) if regex.search(line)]
            if hits:
                found.append((full, relative, hits))
        if not found:
            if excluded and not searched:
                return note + _glob_missed(glob, f"{excluded} searchable files under {path or 'the app'}")
            return note + f"No matches for: {pattern}"
        total = sum(len(hits) for _, _, hits in found)

        if mode == "files":
            ranked = sorted(found, key=lambda item: (-len(item[2]), item[1]))
            shown = ranked[:limit]
            body = "\n".join(f"{relative} ({len(hits)})" for _, relative, hits in shown)
            more = (f"\n… {len(ranked) - len(shown)} more files; narrow path or glob, or raise head_limit "
                    f"(max {SEARCH_MAX_LIMIT})." if len(ranked) > len(shown) else "")
            return (note + f"{len(found)} files, {total} matching lines. output_mode='content' shows the lines.\n"
                    + body + more)

        blocks, shown, chars, remaining = [], 0, 0, limit
        for full, relative, hits in found:
            if remaining <= 0:
                break
            lines = _read_lines(full) or []
            hits = [index for index in hits[:remaining] if index < len(lines)]
            remaining -= len(hits)
            windows = []  # [start, end, matched indexes], overlapping windows merged
            for index in hits:
                start, end = max(0, index - context), min(len(lines), index + context + 1)
                if windows and start <= windows[-1][1]:
                    windows[-1][1] = max(windows[-1][1], end)
                    windows[-1][2].add(index)
                else:
                    windows.append([start, end, {index}])
            for start, end, matched in windows:
                rows = [f"{relative}:{min(matched) + 1}"]
                for j in range(start, end):
                    text = lines[j].rstrip()
                    if len(text) > SEARCH_LINE_CHARS:
                        text = text[:SEARCH_LINE_CHARS] + "…"
                    rows.append(f"{'>' if j in matched else ' '}{j + 1:5d} | {text}")
                block = "\n".join(rows)
                if chars + len(block) > SEARCH_MAX_CHARS:
                    if not blocks:  # the first window alone is over the cap: show the rows that fit
                        fit, size = 1, len(rows[0])
                        while fit < len(rows) and size + 1 + len(rows[fit]) <= SEARCH_MAX_CHARS:
                            size, fit = size + 1 + len(rows[fit]), fit + 1
                        blocks.append("\n".join(rows[:fit]))
                        shown += len(matched.intersection(range(start, start + fit - 1)))
                    remaining = 0
                    break
                blocks.append(block)
                chars += len(block) + 2
                shown += len(matched)
        more = (f"\n\n… showing {shown} of {total} matching lines in {len(found)} files; narrow path or glob, "
                "or use output_mode='files' to see where the rest are." if shown < total else "")
        return note + "\n\n".join(blocks) + more
    except Exception as ex:
        return _tool_error("search_code", ex, f"Error: {ex}")


def write_file(app_name: str, path: str, content: str) -> str:
    """Write or overwrite a file at the given relative path.

    - Automatically creates parent directories if they don't exist.
    - Only use this for creating NEW files or completely replacing content.
    """
    try:
        full = _resolve_path(app_name, path)
        pattern = redaction_pattern(app_name, path, full)
        if pattern:
            return f"WRITE_FAILED: {path} matches the redaction pattern {pattern} (secrets are never written)"
        original = read_bytes(full)
        _write_source(app_name, full, content.encode("utf-8", errors="surrogateescape"), original)
        return f"WRITE_OK: Wrote {len(content)} chars to {path}"
    except Exception as ex:
        return _tool_error("write_file", ex, f"WRITE_FAILED: Error: {ex}")


def copy_file(app_name: str, source_path: str, destination_path: str, *,
              replacements: dict[str, str] | None = None, expected_sha256: str = "") -> str:
    """Copy a reference without regenerating it or overwriting an unrelated file."""
    try:
        check_active(reserve=5)
        source = _resolve_path(app_name, source_path)
        destination = _resolve_path(app_name, destination_path)
        if source == destination:
            return "COPY_FAILED: Source and destination are the same file."
        # A copy is a read: a redacted source would come back through the destination.
        pattern = redaction_pattern(app_name, source_path, source)
        if pattern:
            return f"COPY_FAILED: {source_path} matches the redaction pattern {pattern} (secrets are never read)"
        content = read_bytes(source)
        if content is None:
            return f"COPY_FAILED: Source does not exist: {source_path}"
        if expected_sha256 and hashlib.sha256(content).hexdigest() != expected_sha256:
            return f"COPY_FAILED: Source changed since it was read: {source_path}. Read it again."
        substitutions = replacements or {}
        if not isinstance(substitutions, dict) or len(substitutions) > 12:
            return "COPY_FAILED: Supply at most twelve literal identity replacements."
        # One identity map is usually sent for a page's .json, .py and .js; a
        # name one of them lacks is skipped and reported, not a failed copy.
        absent = []
        if substitutions:
            text = content.decode('utf-8')
            for old, new in substitutions.items():
                if not isinstance(old, str) or not old or not isinstance(new, str):
                    return "COPY_FAILED: Replacement keys must be nonempty strings and values must be strings."
                if old not in text:
                    absent.append(old)
                    continue
                text = text.replace(old, new)
            content = text.encode('utf-8')
        existing = read_bytes(destination)
        if existing is not None:
            if existing == content:
                return f"COPY_OK: {destination_path} already has the requested copied content."
            return f"COPY_FAILED: {destination_path} already exists with different content; edit that file instead."
        _write_source(app_name, destination, content, None, exclusive=True)
        skipped = f" Not in the source, so not applied: {', '.join(map(repr, absent))}." if absent else ""
        return (f"COPY_OK: Copied {source_path} to {destination_path}; applied "
                f"{len(substitutions) - len(absent)} literal identity replacement(s).{skipped}")
    except Exception as ex:
        return _tool_error("copy_file", ex, f"COPY_FAILED: {ex}")


def delete_file(app_name: str, path: str, *, expected_sha256: str) -> str:
    """Delete one explicitly approved file; reconcile retries and lost acknowledgements."""
    try:
        full = _resolve_path(app_name, path)
        pattern = redaction_pattern(app_name, path, full)
        if pattern:
            return f"DELETE_FAILED: {path} matches the redaction pattern {pattern} (secrets are never deleted)"
        current = read_bytes(full)
        if current is None:
            return f"DELETE_OK: {path} is already absent."
        if not expected_sha256 or hashlib.sha256(current).hexdigest() != expected_sha256:
            return f"DELETE_FAILED: {path} changed since it was read; read it again."
        check_active(reserve=5)
        canonical = os.path.relpath(full, os.path.realpath(_app_root(app_name))).replace("\\", "/")
        checkpoint.write_intent({canonical: current.decode("utf-8", errors="surrogateescape")}, {canonical: None})
        try:
            os.unlink(full)
        except OSError:
            if os.path.lexists(full):
                raise
        return f"DELETE_OK: Removed {path}."
    except Exception as ex:
        return _tool_error("delete_file", ex, f"DELETE_FAILED: {ex}")


def rename_file(app_name: str, source_path: str, destination_path: str, *, expected_sha256: str = "") -> str:
    """Move one file without overwriting another; interrupted moves can resume.

    Linking the destination is exclusive on both POSIX and Windows. If removing
    the source fails, both names point to the same file and the next call can
    finish the move. A successful retry with an absent source requires a known
    content digest, rather than assuming any existing destination is ours.
    """
    try:
        source = _resolve_path(app_name, source_path)
        destination = _resolve_path(app_name, destination_path)
        if os.path.normcase(source) == os.path.normcase(destination):
            return "RENAME_FAILED: Source and destination must be different file paths."
        # Moving a redacted file to an unredacted name would make it readable.
        pattern = redaction_pattern(app_name, source_path, source)
        if pattern:
            return f"RENAME_FAILED: {source_path} matches the redaction pattern {pattern} (secrets are never moved)"

        def digest(path):
            with open(path, "rb") as stream:
                return hashlib.sha256(stream.read()).hexdigest()

        if not os.path.isfile(source):
            if (not os.path.lexists(source) and expected_sha256 and os.path.isfile(destination)
                    and digest(destination) == expected_sha256):
                return f"RENAME_OK: Already moved {source_path} to {destination_path}; content verified."
            return f"RENAME_FAILED: Source is missing or not a file: {source_path}. Inspect the current paths."
        expected = expected_sha256 or digest(source)
        if digest(source) != expected:
            return f"RENAME_FAILED: Source changed since it was read: {source_path}. Read it again."
        if os.path.lexists(destination):
            if not os.path.samefile(source, destination) or os.stat(source).st_nlink < 2:
                return f"RENAME_FAILED: Destination already exists: {destination_path}. No file overwritten."
        else:
            check_active(reserve=5)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            os.link(source, destination)  # Fails if another writer creates the destination first.
        if not os.path.samefile(source, destination) or digest(destination) != expected:
            return "RENAME_FAILED: Source changed during the move. Both paths were preserved; inspect them."
        try:
            check_active(reserve=5)
            os.unlink(source)
        except OSError:
            # An acknowledgement can fail after the filesystem change, or a
            # concurrent retry can remove the same source first. Verify reality.
            if os.path.lexists(source) or not os.path.isfile(destination) or digest(destination) != expected:
                raise
        return f"RENAME_OK: Renamed {source_path} to {destination_path}; content preserved."
    except Exception as ex:
        return _tool_error("rename_file", ex, f"RENAME_FAILED: {ex}. Inspect both paths before retrying.")


def edit_file(app_name: str, path: str, old_string: str, new_string: str, expected_occurrences: int = 1) -> str:
    """Replace exact text, requiring an explicit count for a deliberate repeated rename.

    - path: relative to app root.
    - old_string: the current text; a unique match that differs only in whitespace is also accepted.

    CRLF files keep CRLF. The receipt shows the edited region with current line numbers.
    """
    try:
        full = _resolve_path(app_name, path)
        if not os.path.isfile(full):
            return f"EDIT_FAILED: Not a file: {path}"
        # Checked before the read: a missed anchor or the receipt would quote the file back.
        pattern = redaction_pattern(app_name, path, full)
        if pattern:
            return f"EDIT_FAILED: {path} matches the redaction pattern {pattern} (secrets are never edited)"
        original = read_bytes(full)
        content = original.decode("utf-8", errors="surrogateescape")
        if content.count("\r\n") * 2 > content.count("\n"):  # mostly CRLF: edits keep those line endings
            old_string, new_string = _crlf(old_string), _crlf(new_string)
        matches = content.count(old_string) if old_string else 0
        if type(expected_occurrences) is not int or not 1 <= expected_occurrences <= 1000:
            return 'EDIT_FAILED: expected_occurrences must be an integer from 1 to 1000.'
        loose = (_loose_edit(content, old_string, new_string)
                 if old_string and matches == 0 and expected_occurrences == 1 else None)
        if loose is not None:
            first, end, new_string, how = loose
            content = content[:first] + new_string + content[end:]
            _write_source(app_name, full, content.encode("utf-8", errors="surrogateescape"), original)
            return f"EDIT_OK: Updated {path}; old_string matched once {how}.\n" + _edit_excerpt(
                content, first, new_string, 1)
        if not old_string or matches != expected_occurrences:
            return (
                f"EDIT_FAILED: old_string matched {matches} times in {path}; expected {expected_occurrences}. "
                + _anchor_help(content, old_string, matches)
            )
        first = content.find(old_string)
        content = content.replace(old_string, new_string)
        _write_source(app_name, full, content.encode("utf-8", errors="surrogateescape"), original)
    except Exception as ex:
        return _tool_error("edit_file", ex, f"EDIT_FAILED: Error: {ex}")
    return f"EDIT_OK: Updated {path} successfully.\n" + _edit_excerpt(
        content, first, new_string, expected_occurrences)


def _crlf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def _indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip(" \t"))]


def _loose_edit(content: str, old_string: str, new_string: str):
    """``(start, end, new_string, how)`` where only whitespace kept ``old_string`` from matching, or None.

    Needs one unique run of stripped lines off by a consistent indent, which ``new_string`` gets too.
    """
    wanted = old_string.split("\n")
    trailing = wanted[-1] == ""
    if trailing:
        wanted = wanted[:-1]
    if not wanted or not any(len(line.strip()) >= 4 for line in wanted):
        return None
    lines = content.split("\n")
    keys = [line.strip() for line in wanted]
    places = [i for i in range(len(lines) - len(wanted) + 1)
              if lines[i].strip() == keys[0] and all(lines[i + k].strip() == keys[k] for k in range(len(keys)))]
    if len(places) != 1:
        return None
    first = places[0]
    actual = lines[first:first + len(wanted)]
    change = None  # (added prefix, removed prefix), the same for every non-blank line
    for given, found in zip(wanted, actual):
        if not given.strip():
            continue
        g, f = _indent(given), _indent(found)
        this = ("", "") if g == f else (f[:len(f) - len(g)], "") if f.endswith(g) else \
            ("", g[:len(g) - len(f)]) if g.endswith(f) else None
        if this is None or (change is not None and this != change):
            return None
        change = this
    added, removed = change or ("", "")

    def shift(line: str) -> str:
        if not line.strip():
            return line
        if removed and line.startswith(removed):
            return line[len(removed):]
        return added + line

    start = sum(len(line) + 1 for line in lines[:first])
    end = start + sum(len(line) + 1 for line in actual) - 1
    if trailing and end < len(content):
        end += 1
    elif actual[-1].endswith("\r") and not wanted[-1].endswith("\r"):
        end -= 1  # old_string stops before the line break, so the line keeps its CR
    how = ("with its indentation corrected; new_string was shifted the same way" if added or removed
           else "when trailing whitespace is ignored")
    return start, end, "\n".join(shift(line) for line in new_string.split("\n")), how


EDIT_EXCERPT_CONTEXT = 3
EDIT_EXCERPT_MAX_LINES = 40
ANCHOR_MATCH_LINES = 10


def _anchor_help(content: str, old_string: str, matches: int) -> str:
    """What the file holds where a failed anchor was aimed, so the retry needs no read.

    A miss gets the closest numbered window; a wrong count gets each copy's line.
    """
    lines = [line.rstrip("\r") for line in content.split("\n")]
    if matches:
        starts, offset = [], content.find(old_string)
        while offset != -1 and len(starts) < ANCHOR_MATCH_LINES:
            starts.append(content.count("\n", 0, offset) + 1)
            offset = content.find(old_string, offset + len(old_string))
        if matches == 1:
            return f"It occurs once, at line {starts[0]}: pass expected_occurrences=1 to replace that copy."
        where = ", ".join(map(str, starts)) + (", …" if matches > len(starts) else "")
        return (f"It occurs at lines {where}. Widen old_string with neighboring text until it is unique, "
                f"or pass expected_occurrences={matches} to replace every copy.")
    wanted = [line.strip() for line in (old_string or "").split("\n")]
    probe = next((line for line in wanted if len(line) >= 4), "")
    if not probe or not lines:
        return "Copy the exact current text, including whitespace and indentation."
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(probe)
    best, score = 0, 0.0
    for index, line in enumerate(lines):
        matcher.set_seq1(line.strip())
        if matcher.real_quick_ratio() > score and matcher.quick_ratio() > score:
            ratio = matcher.ratio()
            if ratio > score:
                best, score = index, ratio
    if score < 0.5:
        return ("No similar text was found; the code may have moved or been removed. Search for it "
                "rather than guessing the anchor again.")
    offset = wanted.index(probe)
    low = max(0, best - offset - EDIT_EXCERPT_CONTEXT)
    high = min(len(lines), low + len(wanted) + 2 * EDIT_EXCERPT_CONTEXT)
    window = "\n".join(f"{number:5d} | {lines[number - 1]}" for number in range(low + 1, high + 1))
    return ("The closest current text is below; copy old_string from it exactly, including whitespace "
            f"and indentation:\n{window}")


def _edit_excerpt(content: str, offset: int, new_string: str, occurrences: int) -> str:
    """The first replaced region as it now reads, numbered, with a little context."""
    lines = [line.rstrip("\r") for line in content.split("\n")]
    start = content.count("\n", 0, offset)
    end = start + new_string.count("\n")
    low, high = max(0, start - EDIT_EXCERPT_CONTEXT), min(len(lines), end + EDIT_EXCERPT_CONTEXT + 1)
    numbered = [f"{number:5d} | {lines[number - 1]}" for number in range(low + 1, high + 1)]
    if len(numbered) > EDIT_EXCERPT_MAX_LINES:
        half = EDIT_EXCERPT_MAX_LINES // 2
        numbered = numbered[:half] + ["  ... |"] + numbered[-half:]
    note = f"\n[first of {occurrences} replacements shown]" if occurrences > 1 else ""
    return "\n".join(numbered) + note


def _declarations(path: str, lines: list[str]) -> list[tuple[int, str]]:
    """(0-based line index, rendered text) of each class/function-level declaration."""
    outline = []
    if path.endswith(".py"):
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
                outline.append((i, f"{'  ' * (indent // 4)}{stripped.strip()}"))
    elif path.endswith(".js") or path.endswith(".ts"):
        in_class = False
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            lstrip = line.lstrip()
            indent = len(line) - len(lstrip)

            if re.match(r'^(export\s+)?class\s', lstrip):
                in_class = True
                outline.append((i, stripped.strip()))
            elif re.match(r'^(export\s+)?(function|const|let|var)\s', lstrip):
                outline.append((i, stripped.strip()))
            elif re.match(r'^frappe\.(ui\.form\.on|listview_settings|call|pages)', lstrip):
                outline.append((i, stripped.strip()))
            elif re.match(r'^[a-zA-Z_$]+\s*[:=]\s*function', lstrip):
                outline.append((i, stripped.strip()))
            elif re.match(r'^[a-zA-Z_$]+\s*\(', lstrip) and indent == 0:
                outline.append((i, stripped.strip()))
            elif in_class and indent <= 4 and re.match(r'^(async\s+)?[a-zA-Z_$]+\s*\(', lstrip):
                outline.append((i, f"  {stripped.strip()}"))
            elif re.match(r'^\$\(|^jQuery\(', lstrip) and '.on(' in lstrip:
                outline.append((i, stripped.strip()[:100]))
    return outline


#: A declaration row longer than this is cut: a whole minified statement is not a signature.
OUTLINE_ROW_CHARS = 160


def get_file_outline(app_name: str, path: str) -> str:
    """Extract class and function definitions from a file (.py, .js, .ts)."""
    try:
        full = _resolve_path(app_name, path)
        if not os.path.isfile(full):
            return f"Not a file: {path}"
        pattern = redaction_pattern(app_name, path, full)
        if pattern:
            return f"READ_FAILED: {path} matches the redaction pattern {pattern} (secrets are never read)"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total = len(lines)
        if not (path.endswith(".py") or path.endswith(".js") or path.endswith(".ts")):
            return f"Outline not supported for this file type. Use read_file instead. ({total} lines)"
        outline = [f"{i+1:5d} | {text if len(text) <= OUTLINE_ROW_CHARS else text[:OUTLINE_ROW_CHARS] + '…'}"
                   for i, text in _declarations(path, lines)]

        if not outline:
            return f"[{path}] {total} lines — no class/function signatures found. Use read_file to inspect."

        return f"[{path}] {total} lines — outline:\n" + "\n".join(outline)
    except Exception as ex:
        return _tool_error("get_file_outline", ex, f"Error: {ex}")


def read_doctype_schema(app_name: str, doctype_name: str) -> str:
    """Read app source, or the installed schema of a dependency DocType."""
    try:
        app_root = _app_root(app_name)
        name_lower = doctype_name.replace(" ", "_").lower()
        target_file = f"{name_lower}.json"
        for dirpath, _dirnames, filenames in os.walk(app_root):
            if os.path.basename(dirpath) == name_lower and target_file in filenames:
                full = os.path.join(dirpath, target_file)
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
        # A DocType of another installed app (ERPNext, Frappe) is read from the
        # site's installed metadata, the schema that actually applies.
        if callable(getattr(frappe, "get_meta", None)):
            import json
            meta = frappe.get_meta(doctype_name)
            try:
                columns = frappe.db.get_table_columns(doctype_name) if hasattr(frappe.db, "get_table_columns") else []
            except Exception:
                columns = []  # Single and virtual DocTypes have no table; their fields still apply
            return json.dumps({"source": "installed site metadata", "name": meta.name,
                "module": meta.module, "database_columns": columns,
                "fields": [{key: field.get(key) for key in ("fieldname", "fieldtype", "options", "reqd")}
                           for field in meta.fields if field.get("fieldname")]}, ensure_ascii=True)
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
        # A syntax error echoes the offending line.
        pattern = redaction_pattern(app_name, path, full)
        if pattern:
            return f"VALIDATION_FAILED: {path} matches the redaction pattern {pattern} (secrets are never read)"

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
