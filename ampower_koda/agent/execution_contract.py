"""Framework-independent contracts and change evidence for execution."""

import difflib
import hashlib
import json
from pathlib import Path

from ampower_koda.agent.plan_contract import PlanValidationError, validate_plan


def load_plan(value) -> dict:
    if not value:
        raise PlanValidationError([
            "This request has no structured plan. Generate a new plan before execution."
        ])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise PlanValidationError(["Structured plan must contain valid JSON."]) from exc
    return validate_plan(value)


def read_snapshot(path: str) -> str | None:
    """None distinguishes a missing file from an existing empty file."""
    file = Path(path)
    if not file.exists():
        return None
    return file.read_bytes().decode("utf-8", errors="surrogateescape")


def revision(content: str | None) -> str:
    return "missing" if content is None else hashlib.sha256(
        content.encode("utf-8", errors="surrogateescape")
    ).hexdigest()


def change_evidence(before: dict, read_current, *, limit: int = 20000) -> tuple[list, str]:
    """Compare real file content, including new/empty files and reverted edits."""
    changes, blocks = [], []
    remaining = limit
    for path, original in before.items():
        current = read_current(path)
        if original == current:
            continue
        change = {
            "path": path, "before": revision(original), "after": revision(current),
            "summary": "Deleted" if current is None else ("Created" if original is None else "Modified"),
        }
        changes.append(change)
        header = f"\n### {path} ({change['summary']})\nSHA256 {change['before']} -> {change['after']}\n"
        diff = "".join(difflib.unified_diff(
            (original or "").splitlines(keepends=True),
            (current or "").splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}",
        ))
        # Every file keeps its manifest entry even when the diff budget runs out.
        portion = diff[:max(remaining, 0)]
        remaining -= len(portion)
        blocks.append(header + portion + (
            "\n[Diff truncated; read the current file with tools.]\n" if len(portion) < len(diff) else ""
        ))
    return changes, "".join(blocks) or "No net file changes."


def renamed_sources(file_moves: list[dict], read_current) -> set[str]:
    """Missing sources are intentional only when a recorded move still has a destination.

    Follow chains because a file may be renamed more than once before final
    integration. Cycles with no surviving file never excuse missing paths.
    """
    destinations = {move["source"]: move["destination"] for move in file_moves
                    if move.get("source") and move.get("destination")}
    allowed = set()
    for source in destinations:
        if read_current(source) is not None:
            continue
        path, visited = source, {source}
        while path in destinations:
            path = destinations[path]
            if read_current(path) is not None:
                allowed.add(source)
                break
            if path in visited:
                break
            visited.add(path)
    return allowed


def _parse_completion_report(text: str) -> tuple[dict, str]:
    """Return the canonical report and why parsing failed, if it did."""
    try:
        payload = json.loads(text or "")
    except (ValueError, TypeError):
        return {}, ("The reply is not a JSON object. Reply with only the JSON report: "
                    '{"status": "complete"|"blocked", "summary": "...", "behavior": [...], '
                    '"verification": [...], "unverified": [...]}.')
    if not isinstance(payload, dict):
        return {}, "The reply must be a single JSON object, not a list or scalar."
    if payload.get("status") not in ("complete", "blocked"):
        return {}, 'The "status" field must be exactly "complete" or "blocked".'
    if not isinstance(payload.get("summary"), str) or not payload["summary"].strip():
        return {}, 'The "summary" field must be a non-empty string.'
    for key in ("behavior", "verification", "unverified"):
        values = payload.get(key)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            return {}, f'The "{key}" field must be a list of non-empty strings (use [] when there are none).'
    if payload["status"] == "complete" and not payload["behavior"]:
        return {}, 'A "complete" report must list at least one entry under "behavior".'
    return {
        key: payload[key]
        for key in ("status", "summary", "behavior", "verification", "unverified")
    }, ""


def completion_report_problem(text: str) -> str:
    """Why ``text`` is not a usable completion report, or "" when it is."""
    return _parse_completion_report(text)[1]


def completion_report(text: str) -> dict:
    """An explicit completion report is a claim, not independent verification."""
    return _parse_completion_report(text)[0]


def _canonical_review_issues(value) -> list[str] | None:
    """Issue objects (``criterion``, ``severity``, ``issue``) or plain strings as text lines.

    None when an entry would have to be invented or dropped.
    """
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return None
    issues = []
    for item in value:
        if isinstance(item, str) and item.strip():
            issues.append(item.strip())
            continue
        if isinstance(item, dict) and isinstance(item.get("issue"), str) and item["issue"].strip():
            criterion = item.get("criterion")
            severity = str(item.get("severity") or "").upper()
            prefix = (f"[{severity}] " if severity in BLOCKING_SEVERITIES + ADVISORY_SEVERITIES else "") + (
                f"Criterion {criterion}: " if type(criterion) is int else "")
            issues.append(prefix + item["issue"].strip())
            continue
        return None
    return issues


BLOCKING_SEVERITIES = ("P0", "P1")
ADVISORY_SEVERITIES = ("P2", "P3")


def _advisory_only(value) -> bool:
    """Whether every finding is explicitly marked P2/P3; one without a severity still blocks."""
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, dict) and str(item.get("severity", "")).upper() in ADVISORY_SEVERITIES
               for item in value)


def review_decision(payload, criteria: list[str]) -> tuple[str, str]:
    """Separate code defects from missing evidence; both remain fail-closed.

    Findings are ranked P0-P3. Only P0/P1 (the request does not work, data is
    wrong, a regression) send the task back; P2/P3 pass with the notes attached.
    """
    invalid = (
        "Invalid review result: return boolean review_passed, issues as an array of "
        '{"criterion", "severity", "issue"} objects, and evidence for every criterion.'
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("review_passed"), bool):
        return "invalid", invalid
    raw_issues = payload.get("issues")
    issues = _canonical_review_issues(raw_issues)
    if issues is None:
        return "invalid", invalid
    payload = {**payload, "issues": issues}
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != len(criteria):
        return "invalid", invalid
    seen, statuses = set(), set()
    for item in evidence:
        if not isinstance(item, dict):
            return "invalid", invalid
        index = item.get("criterion")
        if type(index) is not int or not 1 <= index <= len(criteria) or index in seen:
            return "invalid", invalid
        status = item.get("status")
        if status not in ("satisfied", "unmet", "unverified"):
            return "invalid", invalid
        if not isinstance(item.get("evidence"), str) or not item["evidence"].strip():
            return "invalid", invalid
        seen.add(index)
        statuses.add(status)
    advisory = _advisory_only(raw_issues)
    if payload["review_passed"]:
        if (issues and not advisory) or statuses - {"satisfied"}:
            return "invalid", "Invalid review result: pass contradicts unresolved issues or criteria."
        return "pass", json.dumps(payload, ensure_ascii=True)
    # Resolve unknowns before sending a complete set of concrete defects to repair.
    if "unverified" in statuses:
        return "needs_evidence", json.dumps(payload, ensure_ascii=True)
    if advisory:
        return "pass", json.dumps({**payload, "review_passed": True,
                                   "advisory": "Only P2/P3 findings remain; recorded, not blocking."},
                                  ensure_ascii=True)
    if "unmet" in statuses and issues:
        return "repair", json.dumps(payload, ensure_ascii=True)
    return "invalid", invalid


def review_verdict(payload, criteria: list[str]) -> tuple[bool, str]:
    decision, notes = review_decision(payload, criteria)
    return decision == "pass", notes
