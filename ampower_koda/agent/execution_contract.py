"""Framework-independent contracts and change evidence for execution."""

import difflib
import hashlib
import json
from pathlib import Path

from ampower_koda.agent.plan_contract import PlanValidationError, validate_plan


def load_plan(value, *, path_exists=None) -> dict:
    if not value:
        raise PlanValidationError([
            "This request has no structured plan. Generate a new plan before execution."
        ])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise PlanValidationError(["Structured plan must contain valid JSON."]) from exc
    return validate_plan(value, path_exists=path_exists)


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
            "summary": "Created" if original is None else "Modified",
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


def completion_report(text: str) -> dict:
    """An explicit completion report is a claim, not independent verification."""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return {}
    if not isinstance(payload, dict) or payload.get("status") not in ("complete", "blocked"):
        return {}
    if not isinstance(payload.get("summary"), str) or not payload["summary"].strip():
        return {}
    for key in ("behavior", "verification", "unverified"):
        values = payload.get(key)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            return {}
    if payload["status"] == "complete" and not payload["behavior"]:
        return {}
    return {key: payload[key] for key in ("status", "summary", "behavior", "verification", "unverified")}


def review_decision(payload, criteria: list[str]) -> tuple[str, str]:
    """Separate code defects from missing evidence; both remain fail-closed."""
    invalid = "Invalid review result: return boolean review_passed, string issues, and evidence for every criterion."
    if not isinstance(payload, dict) or not isinstance(payload.get("review_passed"), bool):
        return "invalid", invalid
    issues = payload.get("issues")
    if not isinstance(issues, list) or any(not isinstance(i, str) or not i.strip() for i in issues):
        return "invalid", invalid
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
    if payload["review_passed"]:
        if issues or statuses - {"satisfied"}:
            return "invalid", "Invalid review result: pass contradicts unresolved issues or criteria."
        return "pass", json.dumps(payload, ensure_ascii=True)
    # Resolve unknowns before sending a complete set of concrete defects to repair.
    if "unverified" in statuses:
        return "needs_evidence", json.dumps(payload, ensure_ascii=True)
    if "unmet" in statuses and issues:
        return "repair", json.dumps(payload, ensure_ascii=True)
    return "invalid", invalid


def review_verdict(payload, criteria: list[str]) -> tuple[bool, str]:
    decision, notes = review_decision(payload, criteria)
    return decision == "pass", notes
