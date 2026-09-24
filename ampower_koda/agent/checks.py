"""Real, mechanical Frappe health checks — run before any LLM review.

Task checks cover syntax and JSON. Final integration also checks imports and
wiring, once all dependent tasks have been implemented. Mechanical checks do
not replace review against the approved acceptance criteria.

Each checker runs per file in isolation, so a bad file or a crashing checker
becomes a failure line instead of aborting the whole check pass.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys

import frappe

from ampower_koda.agent.tools import _resolve_path, validate_code
from ampower_koda.agent.run_control import check_active

REQUIRED_DOCTYPE_KEYS = ("doctype", "name", "module")
REQUIRED_REPORT_KEYS = ("doctype", "report_name", "ref_doctype")
REQUIRED_PAGE_KEYS = ("doctype", "page_name")

# Which required-key set applies, keyed by the JSON's own "doctype" value —
# this is the same field Frappe itself uses to know what kind of record it is.
_REQUIRED_KEYS_BY_DOCTYPE = {
    "DocType": REQUIRED_DOCTYPE_KEYS,
    "Report": REQUIRED_REPORT_KEYS,
    "Page": REQUIRED_PAGE_KEYS,
}


class CheckResult:
    """One check's verdict: did it pass, and what should a human/LLM read."""

    __slots__ = ("name", "passed", "detail", "verified", "owner")

    def __init__(self, name: str, passed: bool, detail: str = "", *, verified: bool = True,
                 owner: str = "implementation"):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.verified = verified
        self.owner = owner

    def line(self) -> str:
        status = ("OK" if self.verified else "UNVERIFIED") if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}" if self.detail else f"[{status}] {self.name}"


class HealthReport:
    """The combined result of every check run for one review pass."""

    def __init__(self, results: list[CheckResult]):
        self.results = results

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def environment_failures(self) -> list[CheckResult]:
        return [r for r in self.failures if r.owner == "environment"]

    def summary(self, limit: int = 12) -> str:
        """Readable report — failures first, so a human/LLM sees them without scrolling."""
        ordered = self.failures + [r for r in self.results if r.passed]
        lines = [r.line() for r in ordered[:limit]]
        if len(ordered) > limit:
            lines.append(f"... and {len(ordered) - limit} more check(s).")
        return "\n".join(lines)


def run_health_checks(app_name: str, edits: list[dict]) -> HealthReport:
    """Run every mechanical check against this run's edited files."""
    paths = [e.get("path", "") for e in (edits or []) if e.get("path")]
    results: list[CheckResult] = []
    for checker in (_syntax_checks, _json_checks, _import_checks, _wiring_checks):
        results.extend(_isolated_checks(checker, app_name, paths))
    return HealthReport(results)


def run_task_checks(app_name: str, paths: list[str]) -> HealthReport:
    """Check an intermediate task without requiring later tasks' wiring."""
    return HealthReport(_syntax_checks(app_name, paths) + _json_checks(app_name, paths))


def _isolated_checks(checker, app_name: str, paths: list[str]) -> list[CheckResult]:
    """A malformed file must not abort checks or discard the node's repair path."""
    results = []
    for path in dict.fromkeys(paths):
        check_active(reserve=20, max_age=2)
        try:
            results.extend(checker(app_name, [path]))
        except Exception as exc:
            results.append(CheckResult(f"{checker.__name__}:{path}", False,
                                       f"Could not check this file: {type(exc).__name__}: {exc}",
                                       owner="environment"))
    return results


# ---------------------------------------------------------------------------
# Syntax — thin wrapper over the existing validate_code tool
# ---------------------------------------------------------------------------

def _syntax_checks(app_name: str, paths: list[str]) -> list[CheckResult]:
    results = []
    for path in paths:
        if not path.endswith((".py", ".js")):
            continue
        outcome = validate_code(app_name, path)
        if "Not a file" in outcome:
            # A phantom path parsed from the model's own summary text, not a
            # real edit — nothing to check.
            continue
        ok = outcome.startswith("VALID:")
        unavailable = outcome.startswith("VALIDATION_UNAVAILABLE:")
        results.append(CheckResult(f"syntax:{path}", ok, outcome[:2000],
                                   verified=not unavailable, owner="environment" if unavailable else "implementation"))
    return results


# ---------------------------------------------------------------------------
# DocType / Report / Page JSON
# ---------------------------------------------------------------------------

def _json_checks(app_name: str, paths: list[str]) -> list[CheckResult]:
    results = []
    for path in paths:
        if not path.endswith(".json"):
            continue
        try:
            full = _resolve_path(app_name, path)
        except ValueError as e:
            results.append(CheckResult(f"json:{path}", False, str(e)))
            continue
        if not os.path.isfile(full):
            continue

        try:
            with open(full, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, UnicodeError, json.JSONDecodeError) as e:
            results.append(CheckResult(f"json:{path}", False, f"invalid JSON: {e}"))
            continue

        if not isinstance(data, dict) or "doctype" not in data:
            # Not a Frappe metadata file (e.g. a plain config/data JSON) —
            # valid JSON is all that's expected of it.
            results.append(CheckResult(f"json:{path}", True))
            continue

        if not isinstance(data["doctype"], str) or not data["doctype"].strip():
            results.append(CheckResult(f"json:{path}", False, "doctype must be a non-empty string"))
            continue
        required = _REQUIRED_KEYS_BY_DOCTYPE.get(data["doctype"])
        if required is None:
            # A DocType/Report/Page JSON of a kind we don't have a specific
            # rule for yet — valid JSON with a doctype key is as far as we check.
            results.append(CheckResult(f"json:{path}", True))
            continue

        missing = [key for key in required if not data.get(key)]
        if missing:
            results.append(CheckResult(f"json:{path}", False, f"missing required key(s): {', '.join(missing)}"))
        else:
            results.append(CheckResult(f"json:{path}", True))
    return results


# ---------------------------------------------------------------------------
# Import smoke test — run in a subprocess so a broken import can't take
# down the checker process itself, and so partially-applied module state
# from one bad import never leaks into the next check.
# ---------------------------------------------------------------------------

def _import_checks(app_name: str, paths: list[str]) -> list[CheckResult]:
    results = []
    for path in paths:
        if not path.endswith(".py") or path.endswith("__init__.py"):
            continue
        try:
            full = _resolve_path(app_name, path)
        except ValueError as e:
            results.append(CheckResult(f"import:{path}", False, str(e)))
            continue
        if not os.path.isfile(full):
            continue

        module_name = _module_name_for(app_name, path)
        if module_name is None:
            continue

        try:
            proc = subprocess.run(
                [sys.executable, "-c", f"import {module_name}"],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=frappe.get_bench_path() if hasattr(frappe, "get_bench_path") else None,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            # The interpreter could not start or the bench was too slow: not a code defect.
            results.append(CheckResult(f"import:{path}", False, str(exc), owner="environment"))
            continue
        if proc.returncode == 0:
            results.append(CheckResult(f"import:{path}", True))
        else:
            failure = (proc.stderr or proc.stdout).strip().splitlines()
            last_line = failure[-1] if failure else "import failed"
            hint = ""
            missing = re.search(r"No module named ['\"]([^'\"]+)['\"]", last_line)
            if missing:
                missing_module = missing.group(1)
                matches, references = _module_recovery(app_name, missing_module)
                if len(matches) == 1:
                    relative, canonical = matches[0]
                    hint = f"; matching module exists at {relative}; import it as '{canonical}'"
                    if references:
                        hint += (f"; replace prefix '{missing_module}' with '{canonical}' at "
                                 + ", ".join(references[:6]))
                elif matches:
                    hint = "; possible module locations: " + ", ".join(
                        f"{relative} ({canonical})" for relative, canonical in matches[:4]
                    )
            results.append(CheckResult(f"import:{path}", False, (last_line + hint)[:1200]))
    return results


def _module_name_for(app_name: str, relative_path: str) -> str | None:
    """Return the canonical import represented by a Python path under the app."""
    normalized = relative_path.replace("\\", "/")
    if normalized.endswith("/__init__.py"):
        normalized = normalized[:-len("/__init__.py")]
    elif normalized.endswith(".py"):
        normalized = normalized[:-3]
    else:
        return None
    parts = [app_name, *[part for part in normalized.split("/") if part]]
    return ".".join(parts) if all(part.isidentifier() for part in parts) else None


def _module_recovery(app_name: str, dotted_module: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Find matching modules and broken import sites in one source-tree walk.

    Frappe apps can place Python packages directly below ``get_app_path`` or
    below another package directory with the app's name.  A checker must not
    guess which layout is in use.  It may, however, report a unique source
    match and its exact import name so an integration repair is deterministic.
    """
    parts = dotted_module.split(".")
    if not parts or parts[0] != app_name or not all(part.isidentifier() for part in parts):
        return [], []
    suffix = "/".join(parts[1:])
    wanted = {suffix + ".py", suffix + "/__init__.py"} if suffix else {"__init__.py"}
    # Frappe controllers conventionally repeat the DocType/Page directory name
    # as the source filename (``doctype/x/x.py``).  A malformed call can stop
    # one segment early and treat ``x`` as the function.  Include that physical
    # shape in discovery; the caller still requires an exact whitelisted
    # function match before offering it as a repair.
    if suffix and len(parts) > 1:
        wanted.add(f"{suffix}/{parts[-1]}.py")
    root = frappe.get_app_path(app_name)
    matches: list[tuple[str, str]] = []
    references: list[str] = []
    ignored = {"__pycache__", "node_modules", ".git", ".eggs", "dist", "build"}
    for directory, dirnames, filenames in os.walk(root):
        check_active(reserve=10, max_age=2)
        dirnames[:] = [name for name in dirnames if name not in ignored and not name.endswith(".egg-info")]
        for filename in filenames:
            if filename != "__init__.py" and not filename.endswith(".py"):
                continue
            full = os.path.join(directory, filename)
            relative = os.path.relpath(full, root).replace("\\", "/")
            if any(relative == item or relative.endswith("/" + item) for item in wanted):
                canonical = _module_name_for(app_name, relative)
                if canonical and canonical != dotted_module and (relative, canonical) not in matches:
                    matches.append((relative, canonical))
            try:
                with open(full, encoding="utf-8", errors="replace") as source:
                    text = source.read()
                # Only a file that names the module can import it; skip parsing the rest.
                if dotted_module not in text:
                    continue
                tree = ast.parse(text)
            except (OSError, SyntaxError, UnicodeError, ValueError):
                continue
            for node in ast.walk(tree):
                imported = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = [node.module]
                for name in imported:
                    if name == dotted_module or name.startswith(dotted_module + "."):
                        entry = f"{relative}:{node.lineno} (imports '{name}')"
                        if entry not in references:
                            references.append(entry)
    return sorted(matches), sorted(references)


# ---------------------------------------------------------------------------
# Client <-> server wiring — every frappe.call({method: "..."}) in edited JS
# must resolve to a real, @frappe.whitelist()-decorated Python function.
# ---------------------------------------------------------------------------

_CALL_METHOD_RE = None  # compiled lazily to keep the import list minimal


def _wiring_checks(app_name: str, paths: list[str]) -> list[CheckResult]:
    global _CALL_METHOD_RE
    if _CALL_METHOD_RE is None:
        _CALL_METHOD_RE = re.compile(r"""frappe\.call\(\s*\{[^}]*?method\s*:\s*["']([\w.]+)["']""")

    results = []
    for path in paths:
        if not path.endswith(".js"):
            continue
        try:
            full = _resolve_path(app_name, path)
        except ValueError as e:
            results.append(CheckResult(f"wiring:{path}", False, str(e)))
            continue
        if not os.path.isfile(full):
            continue

        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        methods = _CALL_METHOD_RE.findall(content)
        if not methods:
            continue

        for method in methods:
            ok, reason = _whitelisted(method)
            results.append(CheckResult(f"wiring:{path} -> {method}", ok, reason))
    return results


def _whitelisted(dotted_method: str) -> tuple[bool, str]:
    """Inspect current source, without importing cached or site-dependent code."""
    try:
        module_name, _, func_name = dotted_method.rpartition(".")
        if not module_name:
            return False, "not a fully-qualified method path"
        parts = module_name.split(".")
        if not all(p.isidentifier() for p in [*parts, func_name]):
            return False, "invalid Python method path"
        relative = "/".join(parts[1:])
        full = _resolve_path(parts[0], relative + ".py") if relative else ""
        if not full or not os.path.isfile(full):
            full = _resolve_path(parts[0], (relative + "/" if relative else "") + "__init__.py")
        with open(full, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        frappe_aliases, whitelist_aliases = set(), set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                frappe_aliases.update(a.asname or a.name for a in node.names if a.name == "frappe")
            elif isinstance(node, ast.ImportFrom) and node.module == "frappe":
                whitelist_aliases.update(a.asname or a.name for a in node.names if a.name == "whitelist")
        functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name]
        if not functions:
            return False, "method not found"
        for decorator in functions[-1].decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Name) and target.id in whitelist_aliases:
                return True, "whitelist decorator found in current source"
            if isinstance(target, ast.Attribute) and target.attr == "whitelist" and isinstance(target.value, ast.Name) and target.value.id in frappe_aliases:
                return True, "whitelist decorator found in current source"
        return False, "could not verify a frappe.whitelist decorator in current source"
    except Exception as e:  # a bad import here is itself a wiring failure, not a crash
        return False, f"could not resolve: {e}"
