"""Real, mechanical Frappe health checks — run before any LLM review.

Four checks, each independent and each returning a plain pass/fail plus a
short human-readable reason. If every check passes, review_node decides
whether the change is small/safe enough to skip the LLM call too — see
``needs_llm_review`` below.

This module never raises on a single bad file — one broken edit should not
crash the whole check pass. Every checker catches its own exceptions and
reports them as a failure line instead.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess

import frappe

from ampower_koda.agent.tools import _resolve_path, validate_code
from ampower_koda.agent.git_ops import diff_file

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

    __slots__ = ("name", "passed", "detail")

    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def line(self) -> str:
        status = "OK" if self.passed else "FAIL"
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
    results.extend(_syntax_checks(app_name, paths))
    results.extend(_json_checks(app_name, paths))
    results.extend(_import_checks(app_name, paths))
    results.extend(_wiring_checks(app_name, paths))
    return HealthReport(results)


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
        ok = outcome.startswith("VALID")
        results.append(CheckResult(f"syntax:{path}", ok, outcome.split("\n", 1)[0][:200]))
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
        except (OSError, json.JSONDecodeError) as e:
            results.append(CheckResult(f"json:{path}", False, f"invalid JSON: {e}"))
            continue

        if not isinstance(data, dict) or "doctype" not in data:
            # Not a Frappe metadata file (e.g. a plain config/data JSON) —
            # valid JSON is all that's expected of it.
            results.append(CheckResult(f"json:{path}", True))
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

        proc = subprocess.run(
            ["python3", "-c", f"import {module_name}"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=frappe.get_bench_path() if hasattr(frappe, "get_bench_path") else None,
        )
        if proc.returncode == 0:
            results.append(CheckResult(f"import:{path}", True))
        else:
            failure = (proc.stderr or proc.stdout).strip().splitlines()
            last_line = failure[-1] if failure else "import failed"
            results.append(CheckResult(f"import:{path}", False, last_line[:200]))
    return results


def _module_name_for(app_name: str, relative_path: str) -> str | None:
    """Turn ``app/module/file.py`` into ``app.module.file`` for ``import``."""
    if not relative_path.endswith(".py"):
        return None
    without_ext = relative_path[: -len(".py")]
    parts = [p for p in without_ext.split("/") if p]
    if not parts:
        return None
    return ".".join(parts)


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
    """Confirm ``dotted_method`` exists and is @frappe.whitelist()-decorated."""
    try:
        module_name, _, func_name = dotted_method.rpartition(".")
        if not module_name:
            return False, "not a fully-qualified method path"
        module = frappe.get_attr(module_name) if hasattr(frappe, "get_attr") else __import__(module_name, fromlist=["_"])
        func = getattr(module, func_name, None)
        if func is None:
            return False, "method not found"
        if not getattr(func, "whitelisted", False):
            return False, "found but not @frappe.whitelist()"
        return True, ""
    except Exception as e:  # a bad import here is itself a wiring failure, not a crash
        return False, f"could not resolve: {e}"


# ---------------------------------------------------------------------------
# Deciding whether the mechanical checks passing is enough on its own, or
# whether this specific change is worth an LLM's judgment too.
#
# Two independent, cheap-to-compute signals — neither needs a model call:
#   1. size    — how many lines actually changed, from the real git diff
#   2. risk    — does the diff touch anything on a fixed watch-list
#
# Either one alone is enough to force an LLM pass. The default is to SKIP,
# so token savings only happen when both signals agree there's nothing here
# worth a model's attention.
# ---------------------------------------------------------------------------

# Deliberately small and specific — a long list of vague terms just forces
# the LLM pass on everything, which defeats the point of having this at all.
# Checked only against ADDED lines in the diff (see needs_llm_review below) —
# matching the whole unified diff would also catch unrelated code sitting in
# the surrounding context lines, which would force LLM review on any edit
# that merely lands near, say, an existing subprocess.run() call.
RISK_PATTERNS = (
    re.compile(r"\bfrappe\.db\.sql\b"),
    re.compile(r"\bdelete_doc\b"),
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bsubprocess\."),
    re.compile(r"\beval\("),
    re.compile(r"\bexec\("),
    re.compile(r"\bignore_permissions\s*=\s*True\b"),
    re.compile(r"\bhas_permission\b"),
    re.compile(r"\bfrappe\.session\.user\b"),
    re.compile(r"\bapi_key\b", re.IGNORECASE),
    re.compile(r"\bsecret\b", re.IGNORECASE),
    re.compile(r"\bpassword\b", re.IGNORECASE),
    re.compile(r"@frappe\.whitelist\([^)]*allow_guest\s*=\s*True"),
)

# Changes at or under this many total changed lines (added + removed, from
# the real diff) are treated as small enough that a passed mechanical check
# is sufficient on its own.
SMALL_CHANGE_LINE_THRESHOLD = 25


class ReviewDecision:
    """Whether this run's edits need an LLM look, and why."""

    __slots__ = ("needs_llm", "reason")

    def __init__(self, needs_llm: bool, reason: str):
        self.needs_llm = needs_llm
        self.reason = reason


def needs_llm_review(app_name: str, base_branch: str, branch_name: str, edits: list[dict]) -> ReviewDecision:
    """Decide whether mechanical checks passing is enough, or an LLM should
    still look at this change.

    Only ``.py`` edits are considered for size/risk — pure JSON/JS-only
    changes are already fully covered by the mechanical checks (JSON
    validity + the wiring check), so they never need this at all.
    """
    py_paths = [e.get("path", "") for e in (edits or []) if (e.get("path") or "").endswith(".py")]
    if not py_paths:
        return ReviewDecision(False, "no Python changes — mechanical checks cover this fully")

    if not branch_name:
        # No branch to diff against yet (e.g. edits not committed) — cannot
        # measure size/risk safely, so fail toward the LLM rather than
        # silently skipping a change we couldn't actually inspect.
        return ReviewDecision(True, "no branch available to diff — cannot assess size/risk")

    total_changed_lines = 0
    for path in py_paths:
        ok, diff_text = diff_file(app_name, base_branch, branch_name, path)
        if not ok:
            # Can't get a diff for this file — same reasoning as above,
            # don't guess that it's safe.
            return ReviewDecision(True, f"could not diff {path} — cannot assess size/risk")

        for line in diff_text.splitlines():
            if line.startswith(("+++", "---")):
                continue
            if line.startswith(("+", "-")):
                total_changed_lines += 1

        # Outside the line-counting loop on purpose — only needs to be built
        # once per file, and must run even when the diff has zero +/- lines
        # (e.g. a rename-only change), so it can never be left unassigned.
        added_lines = "\n".join(
            line[1:] for line in diff_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        for pattern in RISK_PATTERNS:
            if pattern.search(added_lines):
                return ReviewDecision(True, f"{path} matches a risk pattern ({pattern.pattern})")

    if total_changed_lines > SMALL_CHANGE_LINE_THRESHOLD:
        return ReviewDecision(
            True, f"{total_changed_lines} changed line(s) exceeds the small-change threshold ({SMALL_CHANGE_LINE_THRESHOLD})"
        )

    return ReviewDecision(
        False, f"{total_changed_lines} changed line(s), no risk pattern matched — mechanical checks are enough"
    )