"""(Utility file) Validate and render structured implementation plans."""

from __future__ import annotations

import copy
import difflib
import posixpath
import re
from collections.abc import Callable, Iterable
MAX_PLAN_TASKS = 12
MAX_TASK_CONTEXT_REFS = 6
VALID_ACTIONS = {"MODIFY", "CREATE", "DELETE"}
_TASK_ID = re.compile(r"TODO [1-9][0-9]*")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


PLAN_JSON_SCHEMA = {
    "title": "ImplementationPlan",
    "description": "A complete, reviewable implementation plan.",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overview": {"type": "string"},
        "scope": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "in": {"type": "array", "items": {"type": "string"}},
                "out": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["in", "out"],
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "goal": {"type": "string"},
                    "description": {"type": "string"},
                    "action": {"type": "string", "enum": ["MODIFY", "CREATE", "DELETE"]},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "context_refs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "path": {"type": "string"},
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                                "symbol": {"type": "string"},
                                "why": {"type": "string"},
                            },
                            "required": ["path", "start", "end", "symbol", "why"],
                        },
                    },
                    "acceptance_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "id", "title", "goal", "description", "action", "files",
                    "context_refs", "acceptance_criteria", "depends_on",
                ],
            },
        },
    },
    "required": ["overview", "scope", "assumptions", "risks", "tasks"],
}


class PlanValidationError(ValueError):
    """Raised when a plan cannot safely be approved."""

    def __init__(self, issues: list[str]):
        self.issues = [str(issue) for issue in issues if str(issue).strip()]
        super().__init__("; ".join(self.issues) or "Invalid structured plan")


def _check_keys(value: dict, required: set[str], label: str, issues: list[str]) -> None:
    for key in sorted(required - set(value)):
        issues.append(f"{label}: {key} is required")
    for key in sorted(set(value) - required):
        issues.append(f"{label}: unexpected field {key}")


def _text(value, label: str, issues: list[str], *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        issues.append(f"{label} must be a string")
        return ""
    result = value.strip()
    if not result and not allow_empty:
        issues.append(f"{label} must not be empty")
    return result


def _string_list(value, label: str, issues: list[str], *, required: bool = False) -> list[str]:
    if not isinstance(value, list):
        issues.append(f"{label} must be an array of strings")
        return []
    result: list[str] = []
    for index, item in enumerate(value):
        text = _text(item, f"{label}[{index}]", issues)
        if text:
            if text in result:
                issues.append(f"{label}: duplicate value {text}")
            else:
                result.append(text)
    if required and not result:
        issues.append(f"{label} must not be empty")
    return result


def _path(value, label: str, issues: list[str]) -> str:
    path = _text(value, label, issues).replace("\\", "/")
    if not path:
        return ""
    if path.startswith("/") or _WINDOWS_DRIVE.match(path):
        issues.append(f"{label} must be relative to the target app")
        return ""
    if ".." in path.split("/"):
        issues.append(f"{label} cannot contain parent traversal")
        return ""
    normalized = posixpath.normpath(path)
    if normalized in ("", "."):
        issues.append(f"{label} must name a file")
        return ""
    return normalized.removeprefix("./")


def _line_number(value, label: str, issues: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        issues.append(f"{label} must be a non-negative integer")
        return 0
    return value


def _task(raw, index: int, issues: list[str]) -> dict:
    label = f"task {index + 1}"
    task_keys = {
        "id", "title", "goal", "description", "action", "files",
        "context_refs", "acceptance_criteria", "depends_on",
    }
    if not isinstance(raw, dict):
        issues.append(f"{label} must be an object")
        return {}
    _check_keys(raw, task_keys, label, issues)

    task_id = _text(raw.get("id"), f"{label}.id", issues)
    if task_id and not _TASK_ID.fullmatch(task_id):
        issues.append(f"{label}.id must use the format TODO N")
    action = _text(raw.get("action"), f"{label}.action", issues)
    if action and action not in VALID_ACTIONS:
        issues.append(f"{label}.action must be MODIFY, CREATE or DELETE")

    files = _string_list(raw.get("files"), f"{label}.files", issues, required=True)
    files = [_path(path, f"{label}.files[{i}]", issues) for i, path in enumerate(files)]
    files = [path for path in files if path]
    if len(files) != len(set(files)):
        issues.append(f"{label}.files contains duplicate paths")

    raw_refs = raw.get("context_refs")
    refs: list[dict] = []
    if not isinstance(raw_refs, list):
        issues.append(f"{label}.context_refs must be an array")
    else:
        if not raw_refs:
            issues.append(f"{label}.context_refs must not be empty")
        if len(raw_refs) > MAX_TASK_CONTEXT_REFS:
            issues.append(f"{label}.context_refs allows at most {MAX_TASK_CONTEXT_REFS} entries")
        for ref_index, raw_ref in enumerate(raw_refs):
            ref_label = f"{label}.context_refs[{ref_index}]"
            if not isinstance(raw_ref, dict):
                issues.append(f"{ref_label} must be an object")
                continue
            _check_keys(raw_ref, {"path", "start", "end", "symbol", "why"}, ref_label, issues)
            start = _line_number(raw_ref.get("start"), f"{ref_label}.start", issues)
            end = _line_number(raw_ref.get("end"), f"{ref_label}.end", issues)
            if bool(start) != bool(end):
                issues.append(f"{ref_label} must use two positive lines or two zeroes")
            if start and end and end < start:
                issues.append(f"{ref_label} ends before it starts")
            symbol = _text(raw_ref.get("symbol"), f"{ref_label}.symbol", issues, allow_empty=True)
            why = _text(raw_ref.get("why"), f"{ref_label}.why", issues, allow_empty=True)
            if not symbol and not why:
                issues.append(f"{ref_label} requires symbol or why")
            refs.append({
                "path": _path(raw_ref.get("path"), f"{ref_label}.path", issues),
                "start": start,
                "end": end,
                "symbol": symbol,
                "why": why,
            })

    return {
        "id": task_id,
        "title": _text(raw.get("title"), f"{label}.title", issues),
        "goal": _text(raw.get("goal"), f"{label}.goal", issues),
        "description": _text(raw.get("description"), f"{label}.description", issues),
        "action": action,
        "files": files,
        "context_refs": refs,
        "acceptance_criteria": _string_list(
            raw.get("acceptance_criteria"), f"{label}.acceptance_criteria", issues, required=True
        ),
        "depends_on": _string_list(raw.get("depends_on"), f"{label}.depends_on", issues),
    }


def validate_plan(raw) -> dict:
    """Validate and dependency-order a complete plan.

    Structure only: a task's ``files`` are a starting point, not a write
    permission, so paths are never checked against the disk here.
    """
    issues: list[str] = []
    root_keys = {"overview", "scope", "assumptions", "risks", "tasks"}
    if not isinstance(raw, dict):
        raise PlanValidationError(["plan must be an object"])
    _check_keys(raw, root_keys, "plan", issues)

    scope_raw = raw.get("scope")
    if not isinstance(scope_raw, dict):
        issues.append("plan.scope must be an object")
        scope_raw = {}
    _check_keys(scope_raw, {"in", "out"}, "plan.scope", issues)

    raw_tasks = raw.get("tasks")
    tasks: list[dict] = []
    if not isinstance(raw_tasks, list):
        issues.append("plan.tasks must be an array")
    else:
        if not raw_tasks:
            issues.append("plan.tasks must not be empty")
        if len(raw_tasks) > MAX_PLAN_TASKS:
            issues.append(f"at most {MAX_PLAN_TASKS} tasks are allowed")
        for index, raw_task in enumerate(raw_tasks):
            task = _task(raw_task, index, issues)
            if task:
                tasks.append(task)

    plan = {
        "overview": _text(raw.get("overview"), "plan.overview", issues),
        "scope": {
            "in": _string_list(scope_raw.get("in"), "plan.scope.in", issues),
            "out": _string_list(scope_raw.get("out"), "plan.scope.out", issues),
        },
        "assumptions": _string_list(raw.get("assumptions"), "plan.assumptions", issues),
        "risks": _string_list(raw.get("risks"), "plan.risks", issues),
        "tasks": tasks,
    }
    id_to_index: dict[str, int] = {}
    for index, task in enumerate(tasks):
        key = task["id"].casefold()
        if key in id_to_index:
            issues.append(f"{task['id']}: duplicate task id")
        elif key:
            id_to_index[key] = index
    expected_ids = {f"todo {index}" for index in range(1, len(tasks) + 1)}
    if set(id_to_index) != expected_ids:
        issues.append("task ids must be the contiguous sequence TODO 1 through TODO N")

    dependencies: dict[int, set[int]] = {}
    for index, task in enumerate(tasks):
        dependencies[index] = set()
        for dependency in task["depends_on"]:
            dep_index = id_to_index.get(dependency.casefold())
            if dep_index is None:
                issues.append(f"{task['id']}: unknown dependency {dependency}")
            elif dep_index == index:
                issues.append(f"{task['id']}: task cannot depend on itself")
            else:
                dependencies[index].add(dep_index)

    def depends_transitively(task_index: int, target_index: int) -> bool:
        pending_deps = list(dependencies.get(task_index, set()))
        seen: set[int] = set()
        while pending_deps:
            dependency = pending_deps.pop()
            if dependency == target_index:
                return True
            if dependency not in seen:
                seen.add(dependency)
                pending_deps.extend(dependencies.get(dependency, set()))
        return False

    for left in range(len(tasks)):
        for right in range(left + 1, len(tasks)):
            shared = set(tasks[left]["files"]) & set(tasks[right]["files"])
            if shared and not (
                depends_transitively(left, right) or depends_transitively(right, left)
            ):
                issues.append(
                    f"{tasks[left]['id']} and {tasks[right]['id']} both own "
                    f"{sorted(shared)[0]} but have no dependency ordering"
                )

    ordered: list[int] = []
    placed: set[int] = set()
    pending = list(range(len(tasks)))
    while pending:
        ready = [index for index in pending if dependencies[index] <= placed]
        if not ready:
            ids = ", ".join(tasks[index]["id"] or str(index + 1) for index in pending)
            issues.append(f"dependency cycle involving: {ids}")
            break
        for index in ready:
            ordered.append(index)
            placed.add(index)
            pending.remove(index)

    if issues:
        raise PlanValidationError(issues)
    plan["tasks"] = [tasks[index] for index in ordered]
    return plan


def plan_to_markdown(plan: dict) -> str:
    """Create the complete plan consumed by the current whole-plan executor."""
    lines: list[str] = []
    if plan.get("overview"):
        lines.append(f"## Overview\n{plan['overview']}")
    scope = plan.get("scope") or {}
    if scope.get("in") or scope.get("out"):
        block = ["## Scope"]
        if scope.get("in"):
            block.extend(["**In scope:**", *[f"- {item}" for item in scope["in"]]])
        if scope.get("out"):
            block.extend(["", "**Out of scope:**", *[f"- {item}" for item in scope["out"]]])
        lines.append("\n".join(block))
    if plan.get("assumptions"):
        lines.append("## Assumptions\n" + "\n".join(f"- {item}" for item in plan["assumptions"]))
    if plan.get("tasks"):
        block = ["## Tasks"]
        for task in plan["tasks"]:
            files = ", ".join(f"`{path}`" for path in task["files"])
            depends = ", ".join(task["depends_on"])
            entry = [
                f"### {task['id']} — {task['title']} ({task['action']})",
                f"**Goal:** {task['goal']}", task["description"], f"**Files:** {files}",
            ]
            refs = []
            for ref in task["context_refs"]:
                span = (
                    f"lines {ref['start']}-{ref['end']}"
                    if ref["start"] and ref["end"] else "whole file"
                )
                detail = ref["why"] or ref["symbol"]
                refs.append(f"- `{ref['path']}` ({span})" + (f" — {detail}" if detail else ""))
            entry.append("**Relevant context:**\n" + "\n".join(refs))
            entry.append(
                "**Acceptance criteria:**\n"
                + "\n".join(f"- {criterion}" for criterion in task["acceptance_criteria"])
            )
            if depends:
                entry.append(f"**Depends on:** {depends}")
            block.append("\n\n".join(entry))
        lines.append("\n\n".join(block))
    if plan.get("risks"):
        lines.append("## Risks\n" + "\n".join(f"- {item}" for item in plan["risks"]))
    return "\n\n".join(lines).strip()
def _canonicalize_task_paths(tasks: list[dict], path_exists: Callable[[str], bool], *,
                             prefix: str, start: int = 0,
                             task_id: str = "") -> list[tuple[str, dict[str, str]]]:
    """Remove one proven-redundant app prefix from task paths in place."""
    changed = []
    for index, task in enumerate(tasks):
        if index < start or (task_id and task["id"] != task_id):
            continue
        replacements = {}
        for path in [*task["files"], *[ref["path"] for ref in task["context_refs"]]]:
            if path_exists(path):
                continue
            if not path.startswith(prefix):
                continue
            candidate = path[len(prefix):]
            if candidate and path_exists(candidate):
                replacements[path] = candidate
        if not replacements:
            continue
        task["files"] = [replacements.get(path, path) for path in task["files"]]
        for ref in task["context_refs"]:
            ref["path"] = replacements.get(ref["path"], ref["path"])
        changed.append((task["id"], replacements))
    return changed


def ground_plan_references(plan: dict, path_exists: Callable[[str], bool], *, app_name: str) -> dict:
    """Resolve a proven package-root mismatch before the plan is approved.

    New destinations remain advisory. Existing context references must resolve,
    or be supplied by an earlier dependency. A proven root correction on a peer
    also applies to a new sibling under that same root, never to unrelated paths.
    """
    result = copy.deepcopy(plan)
    prefix = app_name.strip('/') + '/' if app_name else ''
    transforms: dict[str, set[str]] = {}
    replacements = {}
    for task in result['tasks']:
        for ref in task['context_refs']:
            source = ref['path']
            if path_exists(source) or not prefix:
                continue
            candidates = {prefix + source}
            if source.startswith(prefix):
                candidates.add(source[len(prefix):])
            found = [p for p in candidates if path_exists(p)]
            if len(found) != 1:
                continue
            target = found[0]
            replacements[source] = target
            if target == prefix + source:
                old_root = source.split('/')[0] + '/'
                transforms.setdefault(old_root, set()).add(prefix + old_root)
            elif source == prefix + target:
                # Keep the next directory in the key so a correction for
                # app/public/... cannot rewrite app/doctype/... as a side effect.
                new_root = target.split('/')[0] + '/'
                transforms.setdefault(prefix + new_root, set()).add(new_root)
    def resolve(path):
        if path_exists(path):
            return path
        if path in replacements:
            return replacements[path]
        matches = {next(iter(targets)) + path[len(root):]
                   for root, targets in transforms.items()
                   if len(targets) == 1 and path.startswith(root)}
        return next(iter(matches)) if len(matches) == 1 else path
    for task in result['tasks']:
        task['files'] = [resolve(p) for p in task['files']]
        for ref in task['context_refs']:
            ref['path'] = resolve(ref['path'])
    issues, earlier = [], {}
    for task in result['tasks']:
        supplied = set().union(*(earlier.get(dep.casefold(), set()) for dep in task['depends_on']))
        for ref in task['context_refs']:
            if not path_exists(ref['path']) and ref['path'] not in supplied:
                issues.append(f"{task['id']}: context reference {ref['path']!r} does not exist. "
                              "Use the exact verified path from the codebase findings.")
        earlier[task['id'].casefold()] = set(task['files']) | supplied
    if issues:
        raise PlanValidationError(issues)
    # Root corrections may reveal duplicate paths/dependency conflicts.
    return validate_plan(result)


def _path_key(path: str) -> str:
    return re.sub(r"[\s_\-]+", "", path.replace("\\", "/").casefold())


def nearest_paths(path: str, candidates: Iterable[str], *, limit: int = 3) -> list[str]:
    """Existing files a planner most likely meant by ``path``.

    Spaces, underscores and hyphens are ignored, so ``sales_invoice.js`` finds
    ``sales _invoice.js``; a bare suffix finds its prefixed form. Ranked, and
    cut off below a similarity a human would still call "the same file".
    """
    wanted = _path_key(path)
    if not wanted:
        return []
    wanted_base = wanted.rsplit("/", 1)[-1]
    scored: list[tuple[float, str]] = []
    for candidate in candidates:
        key = _path_key(candidate)
        if not key:
            continue
        if key == wanted:
            score = 1.0
        elif key.endswith("/" + wanted) or wanted.endswith("/" + key):
            score = 0.95
        else:
            ratio = difflib.SequenceMatcher(None, wanted, key).ratio()
            base = key.rsplit("/", 1)[-1]
            score = 0.85 + ratio * 0.1 if base == wanted_base else ratio
        if score >= 0.75:
            scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return [candidate for _, candidate in scored[:limit]]


