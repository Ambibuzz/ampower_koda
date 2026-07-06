# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Whitelisted API for the AI Agent

import json
import os
import subprocess
from functools import wraps

import frappe
from frappe import _

from ampower_koda.agent.errors import log_agent_error
from ampower_koda.agent.executor import _generate_patch_diff, as_json_list
from ampower_koda.agent.git_ops import (
    branch_exists,
    checkout_base,
    diff_file,
    get_current_branch,
    get_repo_root,
    list_changed_files,
    run_git,
)
from ampower_koda.agent.graph import _get_bench_env
from ampower_koda.agent.prompts import plan_has_open_questions

DOCTYPE_NAME = "Agent Request"

# Statuses from which a new run or follow-up may be started (agent is idle or finished).
RESTARTABLE_STATUSES = (
    "Queued", "Failed", "Cancelled", "Completed",
    "Awaiting Approval", "Awaiting Push Approval",
)

# Statuses where the agent is actively working, so manual actions must wait.
BUSY_STATUSES = (
    "Understanding", "Planning", "Implementing",
    "Reviewing", "Building", "Pushing",
)


def _whitelist_logged(fn):
    """Log unexpected API failures to Frappe Error Log before re-raising."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (frappe.ValidationError, frappe.PermissionError, frappe.DoesNotExistError):
            raise
        except Exception:
            log_agent_error(f"Agent API: {fn.__name__}")
            raise
    return wrapper


def _validate_provider_key(doc):
    """Ensure the AI provider is enabled and its API key is configured in settings."""
    settings = frappe.get_single("Agent Settings")
    if not settings.enable_ai_agent:
        frappe.throw(_("AI Coding Agent is disabled in Agent Settings."))
    provider = (doc.ai_provider or settings.default_ai_provider or "OpenAI").strip()
    key_checks = {
        "OpenAI": ("openai_api_key", "OpenAI API key"),
        "Gemini": ("google_api_key", "Google API key"),
        "Claude": ("anthropic_api_key", "Anthropic API key"),
    }
    field, label = key_checks.get(provider, key_checks["OpenAI"])
    if not getattr(settings, field, None):
        frappe.throw(_("{0} is not set in Agent Settings.").format(label))


@frappe.whitelist()
@_whitelist_logged
def start_agent(request_name: str):
    """Start a full run from scratch: explore the codebase, then draft a plan for approval."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status not in RESTARTABLE_STATUSES:
        frappe.throw(_("Agent is already busy (status: {0}).").format(doc.status))

    _validate_provider_key(doc)

    frappe.db.set_value(DOCTYPE_NAME, request_name, {
        "status": "Queued",
        "error_log": "",
        "stage_log": "",
        "agent_plan": "",
        "bench_log": "",
        "patch_diff": "",
        "conversation_log": "",
        "understanding_snapshot": "",
    })
    frappe.db.commit()

    frappe.enqueue(
        "ampower_koda.agent.executor.run_planning_phase",
        queue="default",
        timeout=1800,
        request_name=request_name,
    )
    return {"status": "ok", "message": _("Planning phase started.")}


@frappe.whitelist()
@_whitelist_logged
def submit_follow_up(request_name: str, follow_up_message: str):
    """
    Append a user follow-up issue to existing context and run a surgical fix
    on the same branch without restarting explore/plan.
    """
    if not request_name:
        frappe.throw(_("Request name is required."))
    if not (follow_up_message or "").strip():
        frappe.throw(_("Follow-up message is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status not in RESTARTABLE_STATUSES:
        frappe.throw(_("Follow-up is allowed only when the agent is idle (status: {0}).").format(doc.status))
    if not (doc.branch_name or "").strip():
        frappe.throw(_("Follow-up fix needs an existing branch on this request."))

    _validate_provider_key(doc)

    base_request = (doc.user_message or "").strip()
    follow_up = (follow_up_message or "").strip()

    context_lines = []
    if doc.status:
        context_lines.append(f"- Previous status: {doc.status}")
    if doc.branch_name:
        context_lines.append(f"- Previous branch: {doc.branch_name}")
    if doc.pr_url:
        context_lines.append(f"- Previous PR: {doc.pr_url}")
    if doc.error_log:
        context_lines.append(f"- Previous error snapshot: {(doc.error_log or '')[:400]}")
    context_text = "\n".join(context_lines) if context_lines else "- No additional run metadata recorded."

    changed_paths = []
    try:
        for row in as_json_list(doc.files_changed):
            p = (row or {}).get("path") if isinstance(row, dict) else None
            if p:
                changed_paths.append(p)
    except Exception:
        log_agent_error(
            "Agent API: submit_follow_up files_changed",
            f"request={request_name}\n{frappe.get_traceback()}",
        )
    changed_hint = "\n".join(f"- {p}" for p in changed_paths[:20]) if changed_paths else "- (not recorded)"

    merged_message = (
        f"{base_request}\n\n"
        "## FOLLOW-UP ISSUE AFTER USER TESTING\n"
        f"{follow_up}\n\n"
        "## CONTEXT FROM PREVIOUS RUN\n"
        f"{context_text}\n\n"
        "### Instructions for follow-up\n"
        "- Fix the follow-up issue precisely.\n"
        "- Keep all previously working functionality intact.\n"
        "- Prefer surgical changes over broad rewrites.\n"
    )
    focused_plan = (
        "Follow-up fix plan (same branch, no re-explore):\n"
        "1) Reproduce the exact follow-up issue from USER REQUEST.\n"
        "2) Inspect only the files directly related to the issue first.\n"
        "3) Apply minimal targeted edits.\n"
        "4) Run validate_code on changed .py/.js and verify no regressions in touched flows.\n"
        "5) Report precise files changed and why.\n\n"
        "Likely affected files from previous run:\n"
        f"{changed_hint}"
    )

    frappe.db.set_value(DOCTYPE_NAME, request_name, {
        "status": "Implementing",
        "user_message": merged_message[:50000],
        "agent_plan": focused_plan[:50000],
        "error_log": "",
        "bench_log": "",
        "patch_diff": "",
    })
    frappe.db.commit()

    frappe.enqueue(
        "ampower_koda.agent.executor.run_execution_phase",
        queue="default",
        timeout=1800,
        request_name=request_name,
        preserve_branch=1,
    )
    return {"status": "ok", "message": _("Follow-up run started on existing branch (explore/plan skipped).")}


@frappe.whitelist()
@_whitelist_logged
def execute_existing_plan(request_name: str):
    """
    Skips the exploration and planning phases to go straight to implementation.
    This is useful if you have a pre-saved plan on the request that you want to execute immediately.
    """
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if not (doc.agent_plan or "").strip():
        frappe.throw(_("No plan found for this request."))

    if plan_has_open_questions(doc.agent_plan or ""):
        frappe.throw(_(
            "This plan has open questions in 'Questions for User'. "
            "Resolve them before executing."
        ))

    # Implementation can only start if we are at the approval stage or have finished a previous run.
    allowed = ("Awaiting Approval", "Failed", "Cancelled", "Completed", "Awaiting Push Approval")
    if doc.status not in allowed:
        frappe.throw(_("Cannot execute plan. Agent status is {0}.").format(doc.status))

    _validate_provider_key(doc)

    frappe.db.set_value(DOCTYPE_NAME, request_name, {
        "status": "Implementing",
        "error_log": "",
        "bench_log": "",
        "patch_diff": "",
    })
    frappe.db.commit()

    frappe.enqueue(
        "ampower_koda.agent.executor.run_execution_phase",
        queue="default",
        timeout=1800,
        request_name=request_name,
    )
    return {"status": "ok", "message": _("Implementation phase started.")}


@frappe.whitelist()
@_whitelist_logged
def approve_plan(request_name: str, edited_plan: str = None):
    """
    Confirms the plan and begins the implementation phase.
    You can optionally provide an edited version of the plan if you made manual adjustments.
    """
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status != "Awaiting Approval":
        frappe.throw(_("Cannot approve plan. Agent status is {0}.").format(doc.status))

    plan_to_run = (edited_plan or doc.agent_plan or "").strip()
    if not plan_to_run:
        frappe.throw(_("No plan found for this request."))

    if plan_has_open_questions(plan_to_run):
        frappe.throw(_(
            "This plan has open questions in 'Questions for User'. "
            "Answer them in the request description, edit the plan to resolve them, "
            "or replace that section with 'None — request is fully clear.' before approving."
        ))

    if edited_plan is not None and edited_plan.strip():
        frappe.db.set_value(DOCTYPE_NAME, request_name, "agent_plan", edited_plan.strip()[:50000])

    frappe.db.set_value(DOCTYPE_NAME, request_name, {
        "status": "Implementing",
        "patch_diff": "",
    })
    frappe.db.commit()

    frappe.enqueue(
        "ampower_koda.agent.executor.run_execution_phase",
        queue="default",
        timeout=1800,
        request_name=request_name,
    )
    return {"status": "ok", "message": _("Plan approved. Starting implementation.")}


@frappe.whitelist()
@_whitelist_logged
def reject_plan(request_name: str):
    """Reject the plan and set status to Cancelled."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status != "Awaiting Approval":
        frappe.throw(_("Cannot reject. Agent status is {0}.").format(doc.status))

    frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Cancelled")
    frappe.db.commit()

    frappe.publish_realtime("agent_progress", {
        "request_name": request_name,
        "status": "Cancelled",
        "message": _("Plan rejected by user"),
    }, user=doc.owner)

    return {"status": "ok", "message": _("Plan rejected.")}


@frappe.whitelist()
@_whitelist_logged
def approve_bench(request_name: str, commands: str = None):
    """Approves and runs the pending bench commands (migrate, build, clear-cache, etc.).
    Optionally pass an edited list of commands as a JSON array to override the defaults."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status != "Awaiting Bench Approval":
        frappe.throw(_("Cannot approve bench. Agent status is {0}.").format(doc.status))

    if commands:
        try:
            cmd_list = json.loads(commands)
        except (ValueError, TypeError):
            frappe.throw(_("Invalid commands format."))
        if isinstance(cmd_list, list) and cmd_list:
            frappe.db.set_value(
                DOCTYPE_NAME, request_name,
                "pending_bench_commands", json.dumps(cmd_list),
            )

    frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Building")
    frappe.db.commit()

    frappe.enqueue(
        "ampower_koda.agent.executor.run_bench_and_commit",
        queue="default",
        timeout=1800,
        request_name=request_name,
    )

    cmds = []
    try:
        cmds = json.loads(
            frappe.db.get_value(DOCTYPE_NAME, request_name, "pending_bench_commands") or "[]"
        )
    except (ValueError, TypeError):
        cmds = []

    return {
        "status": "ok",
        "message": _("Running {0} bench commands...").format(len(cmds)),
    }


@frappe.whitelist()
@_whitelist_logged
def approve_push(request_name: str, push_branch: int = 1, create_pr: int = 1):
    """Approves pushing the feature branch to remote and/or opening a pull request.
    Set push_branch=1 to push, create_pr=1 to create a PR. At least one must be selected."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status != "Awaiting Push Approval":
        frappe.throw(_("Cannot push. Agent status is {0}.").format(doc.status))

    push_branch = int(push_branch or 0)
    create_pr = int(create_pr or 0)
    if not push_branch and not create_pr:
        frappe.throw(_("Select at least one action (push branch or create PR)."))

    frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Pushing")
    frappe.db.commit()

    frappe.enqueue(
        "ampower_koda.agent.executor.run_deploy_phase",
        queue="default",
        timeout=600,
        request_name=request_name,
        do_push=bool(push_branch),
        do_pr=bool(create_pr),
    )

    if push_branch and create_pr:
        msg = _("Pushing branch and creating PR...")
    elif push_branch:
        msg = _("Pushing branch {0}...").format(doc.branch_name)
    else:
        msg = _("Creating PR...")
    return {"status": "ok", "message": msg}


@frappe.whitelist()
@_whitelist_logged
def checkout_base_branch(request_name: str):
    """Manually checkout the base branch for the target app."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    app_name = (doc.target_app_name or "").strip()
    base_branch = (doc.base_branch or "main").strip()

    if not app_name:
        frappe.throw(_("Target App Name is not set."))

    ok, msg = checkout_base(app_name, base_branch)
    if not ok:
        frappe.throw(_("Failed to checkout base branch: {0}").format(msg))

    frappe.publish_realtime("agent_progress", {
        "request_name": request_name,
        "status": doc.status,
        "message": _("Checked out {0}: {1}").format(base_branch, msg),
    }, user=doc.owner)

    return {"status": "ok", "message": msg}


@frappe.whitelist()
@_whitelist_logged
def get_default_bench_commands(request_name: str):
    """Returns the default list of bench commands for the request's target app:
    migrate, build, clear-cache, and supervisorctl restart."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    app_name = (doc.target_app_name or "").strip()
    site_name = frappe.local.site

    cmds = []
    if app_name:
        cmds = [
            f"bench --site {site_name} migrate",
            f"bench build --app {app_name}",
            f"bench --site {site_name} clear-cache",
            "supervisorctl restart all",
        ]
    return {"commands": cmds}


@frappe.whitelist()
@_whitelist_logged
def run_selected_bench_commands(request_name: str, commands: str = None):
    """Run user-selected bench commands. commands is a JSON array of command strings."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status in BUSY_STATUSES:
        frappe.throw(_("Agent is busy (status: {0}).").format(doc.status))

    cmds = []
    if commands:
        try:
            cmds = json.loads(commands)
        except (ValueError, TypeError):
            frappe.throw(_("Invalid commands format."))

    if not cmds or not isinstance(cmds, list):
        frappe.throw(_("No commands provided."))

    bench_root = os.path.join(frappe.get_app_path("frappe"), "..", "..", "..")
    bench_root = os.path.normpath(bench_root)
    bench_env = _get_bench_env()

    output_parts = []
    for cmd in cmds:
        if not isinstance(cmd, str) or not cmd.strip():
            continue
        try:
            result = subprocess.run(
                cmd.strip().split(),
                cwd=bench_root,
                capture_output=True,
                text=True,
                timeout=900,
                env=bench_env,
            )
            out = (result.stdout or "") + (result.stderr or "")
            status_str = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            output_parts.append(f"$ {cmd}\n{status_str}\n{out.strip()}\n")
        except subprocess.TimeoutExpired:
            output_parts.append(f"$ {cmd}\nTIMEOUT after 900s\n")
            log_agent_error(
                "Agent API: run_selected_bench_commands timeout",
                f"request={request_name}\ncmd={cmd}",
            )
        except Exception as e:
            output_parts.append(f"$ {cmd}\nERROR: {e}\n")
            log_agent_error(
                "Agent API: run_selected_bench_commands",
                f"request={request_name}\ncmd={cmd}\n{e}\n{frappe.get_traceback()}",
            )

    bench_log = "\n".join(output_parts)
    frappe.db.set_value(DOCTYPE_NAME, request_name, "bench_log", bench_log[:50000])
    frappe.db.commit()

    return {"status": "ok", "log": bench_log}


@frappe.whitelist()
@_whitelist_logged
def get_agent_status(request_name: str):
    """Return current status and key fields for a request."""
    if not request_name:
        frappe.throw(_("Request name is required."))
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    return {
        "name": doc.name,
        "status": doc.status,
        "branch_name": doc.branch_name,
        "pr_url": doc.pr_url,
        "pr_number": doc.pr_number,
        "agent_plan": doc.agent_plan,
        "error_log": doc.error_log,
        "stage_log": doc.stage_log,
        "bench_log": doc.bench_log,
        "patch_diff": doc.patch_diff,
    }


_SKIP_TREE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "dist", "build", ".mypy_cache"}
_SKIP_TREE_SUFFIXES = (".pyc", ".pyo")


def _parse_patch_diff_index(patch_diff: str) -> dict[str, str]:
    """Build {relative_path: status} from a stored unified diff."""
    changed: dict[str, str] = {}
    if not patch_diff:
        return changed

    lines = patch_diff.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("--- "):
            i += 1
            continue

        old_raw = line[4:].strip()
        new_raw = ""
        if i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            new_raw = lines[i + 1][4:].strip()
            i += 2
        else:
            i += 1
            continue

        if old_raw == "/dev/null" and new_raw.startswith("b/"):
            changed[new_raw[2:]] = "A"
        elif new_raw == "/dev/null" and old_raw.startswith("a/"):
            changed[old_raw[2:]] = "D"
        elif old_raw.startswith("a/") and new_raw.startswith("b/"):
            changed[new_raw[2:]] = "M"
        elif new_raw.startswith("b/"):
            changed[new_raw[2:]] = "M"

    return changed


def _extract_file_diff_from_patch(patch_diff: str, file_path: str) -> str:
    """Extract one file's diff block from combined patch_diff text."""
    if not patch_diff or not file_path:
        return ""

    blocks = patch_diff.split("\n\n")
    needle_a = f"--- a/{file_path}"
    needle_null_old = "--- /dev/null"
    needle_b = f"+++ b/{file_path}"

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if needle_b in block and (needle_a in block or needle_null_old in block):
            return block
    return ""


def _safe_repo_path(repo_root: str, file_path: str) -> str:
    """Resolve file_path inside repo_root or throw on traversal."""
    if not file_path or file_path.startswith("/") or ".." in file_path.split("/"):
        frappe.throw(_("Invalid file path."))

    full = os.path.normpath(os.path.join(repo_root, file_path))
    root_norm = os.path.normpath(repo_root)
    if not full.startswith(root_norm + os.sep) and full != root_norm:
        frappe.throw(_("File path is outside the repository."))
    return full


def _write_repo_file(repo_root: str, file_path: str, content: str) -> str:
    """Write content to a repo-relative path; return absolute path written."""
    full = _safe_repo_path(repo_root, file_path)
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return full


def _should_skip_tree_entry(name: str) -> bool:
    if name in _SKIP_TREE_DIRS:
        return True
    return any(name.endswith(suffix) for suffix in _SKIP_TREE_SUFFIXES)


def _build_directory_tree(repo_root: str, changed: dict[str, str]) -> list[dict]:
    """Walk repo_root and return nested tree nodes with change metadata."""

    def walk(dir_path: str, rel_prefix: str) -> list[dict]:
        nodes: list[dict] = []
        try:
            entries = sorted(os.listdir(dir_path))
        except OSError:
            return nodes

        for name in entries:
            if _should_skip_tree_entry(name):
                continue

            full = os.path.join(dir_path, name)
            rel = f"{rel_prefix}/{name}" if rel_prefix else name

            if os.path.isdir(full):
                children = walk(full, rel)
                changed_count = sum(
                    1 for c in changed if c == rel or c.startswith(rel + "/")
                )
                nodes.append({
                    "name": name,
                    "type": "folder",
                    "path": rel,
                    "children": children,
                    "changed_count": changed_count,
                    "has_changes": changed_count > 0,
                })
            else:
                status = changed.get(rel)
                nodes.append({
                    "name": name,
                    "type": "file",
                    "path": rel,
                    "status": status,
                    "has_changes": bool(status),
                })

        folders = [n for n in nodes if n["type"] == "folder"]
        files = [n for n in nodes if n["type"] == "file"]
        folders.sort(key=lambda n: n["name"].lower())
        files.sort(key=lambda n: n["name"].lower())
        return folders + files

    return walk(repo_root, "")


def _get_changed_files_for_request(doc) -> tuple[dict[str, str], str]:
    """Return changed file map and data source ('git' or 'patch_diff')."""
    app_name = (doc.target_app_name or "").strip()
    base_branch = (doc.base_branch or "main").strip()
    branch_name = (doc.branch_name or "").strip()

    if app_name and branch_name and branch_exists(app_name, branch_name):
        ok, files = list_changed_files(app_name, base_branch, branch_name)
        if ok and files:
            return {f["path"]: f["status"] for f in files}, "git"

    patch_index = _parse_patch_diff_index(doc.patch_diff or "")
    if patch_index:
        return patch_index, "patch_diff"

    if app_name and branch_name and branch_exists(app_name, branch_name):
        ok, files = list_changed_files(app_name, base_branch, branch_name)
        if ok:
            return {f["path"]: f["status"] for f in files}, "git"

    return {}, "none"


@frappe.whitelist()
@_whitelist_logged
def get_change_tree(request_name: str):
    """Return full app directory tree with agent-modified files highlighted."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    app_name = (doc.target_app_name or "").strip()
    if not app_name:
        frappe.throw(_("Target app is not set on this request."))

    repo_root = get_repo_root(app_name)
    changed, source = _get_changed_files_for_request(doc)
    tree = _build_directory_tree(repo_root, changed)

    branch_state = _get_branch_state(doc)

    pending_cmds = []
    try:
        pending_cmds = json.loads(doc.pending_bench_commands or "[]")
    except Exception:
        log_agent_error(
            "Agent API: get_change_tree pending_bench_commands",
            f"request={request_name}\n{frappe.get_traceback()}",
        )

    return {
        "request_name": doc.name,
        "app_name": app_name,
        "base_branch": (doc.base_branch or "main").strip(),
        "branch_name": (doc.branch_name or "").strip(),
        "current_branch": branch_state["current_branch"],
        "branch_matches": branch_state["matches"],
        "source": source,
        "changed": changed,
        "tree": tree,
        "totals": {"files": len(changed)},
        "request": {
            "status": doc.status,
            "pr_url": doc.pr_url,
            "pr_number": doc.pr_number,
            "pending_bench_commands": pending_cmds,
        },
    }


@frappe.whitelist()
@_whitelist_logged
def get_file_diff(request_name: str, file_path: str):
    """Return untruncated unified diff for one file in the agent's changes."""
    if not request_name:
        frappe.throw(_("Request name is required."))
    if not file_path:
        frappe.throw(_("File path is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    app_name = (doc.target_app_name or "").strip()
    if not app_name:
        frappe.throw(_("Target app is not set on this request."))

    repo_root = get_repo_root(app_name)
    _safe_repo_path(repo_root, file_path)

    base_branch = (doc.base_branch or "main").strip()
    branch_name = (doc.branch_name or "").strip()
    changed, source = _get_changed_files_for_request(doc)
    status = changed.get(file_path, "")

    diff_text = ""
    if branch_name and branch_exists(app_name, branch_name):
        ok, diff_text = diff_file(app_name, base_branch, branch_name, file_path)
        if not ok:
            diff_text = ""

    if not diff_text.strip():
        diff_text = _extract_file_diff_from_patch(doc.patch_diff or "", file_path)

    return {
        "path": file_path,
        "status": status,
        "source": source if diff_text else "none",
        "diff": diff_text,
    }


def _detect_editor_language(file_path: str) -> str:
    ext = os.path.splitext(file_path or "")[1].lower()
    return {
        ".py": "Python",
        ".js": "Javascript",
        ".json": "JSON",
        ".html": "HTML",
        ".css": "CSS",
        ".md": "Markdown",
    }.get(ext, "Text")


def _get_branch_state(doc) -> dict:
    """Return the repo's current branch and whether it matches the request branch.

    This never checks out a branch. `matches` is True only when the request has a
    branch and the repo is currently on it.
    """
    app_name = (doc.target_app_name or "").strip()
    branch_name = (doc.branch_name or "").strip()
    current = get_current_branch(app_name) if app_name else ""
    matches = bool(branch_name) and current == branch_name
    return {"current_branch": current, "branch_name": branch_name, "matches": matches}


def _require_request_branch(doc) -> str:
    """Ensure the repo is on the request's branch; throw on mismatch (no checkout).

    Used to gate write/deploy/push actions so the IDE never silently switches the
    branch the user has checked out.
    """
    app_name = (doc.target_app_name or "").strip()
    if not app_name:
        frappe.throw(_("Target app is not set on this request."))
    branch_name = (doc.branch_name or "").strip()
    if not branch_name:
        frappe.throw(_("No agent branch found on this request."))

    state = _get_branch_state(doc)
    if not state["matches"]:
        frappe.throw(
            _("Repository is on branch '{0}', not this request's branch '{1}'. "
              "Checkout '{1}' to edit, deploy, or push.").format(
                  state["current_branch"] or _("(unknown)"), branch_name)
        )
    return branch_name


def _has_pushable_changes(doc) -> bool:
    """True if the repo has uncommitted work or committed changes vs the base branch.

    Uses only local git state (working tree + base..branch); no remote fetch.
    """
    app_name = (doc.target_app_name or "").strip()
    if not app_name:
        return False

    repo_root = get_repo_root(app_name)
    ok, status_out = run_git(["status", "--short"], cwd=repo_root)
    if ok and status_out.strip():
        return True

    changed, _source = _get_changed_files_for_request(doc)
    return bool(changed)


@frappe.whitelist()
@_whitelist_logged
def get_file_content(request_name: str, file_path: str):
    """Return full file content for the IDE editor."""
    if not request_name:
        frappe.throw(_("Request name is required."))
    if not file_path:
        frappe.throw(_("File path is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    _require_request_branch(doc)
    app_name = (doc.target_app_name or "").strip()
    if not app_name:
        frappe.throw(_("Target app is not set on this request."))

    repo_root = get_repo_root(app_name)
    full = _safe_repo_path(repo_root, file_path)
    if not os.path.isfile(full):
        frappe.throw(_("File not found: {0}").format(file_path))

    with open(full, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    return {
        "path": file_path,
        "full_path": full,
        "content": content,
        "language": _detect_editor_language(file_path),
    }


@frappe.whitelist()
@_whitelist_logged
def save_file_content(request_name: str, file_path: str, content: str):
    """Save edited file content from the IDE to the agent branch working tree."""
    if not request_name:
        frappe.throw(_("Request name is required."))
    if not file_path:
        frappe.throw(_("File path is required."))
    if content is None:
        frappe.throw(_("File content is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status in BUSY_STATUSES:
        frappe.throw(_("Agent is busy (status: {0}).").format(doc.status))

    _require_request_branch(doc)
    app_name = (doc.target_app_name or "").strip()

    repo_root = get_repo_root(app_name)
    full_path = _write_repo_file(repo_root, file_path, content)

    patch_diff = _generate_patch_diff(app_name)
    frappe.db.set_value(DOCTYPE_NAME, request_name, "patch_diff", patch_diff[:100000])
    frappe.db.commit()

    return {
        "status": "ok",
        "message": _("Saved {0} bytes to disk.").format(len(content)),
        "path": file_path,
        "full_path": full_path,
    }


@frappe.whitelist()
@_whitelist_logged
def ide_push(request_name: str, push_branch: int = 1, create_pr: int = 1):
    """Commit current changes, push branch, and optionally create PR from the IDE."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if not (doc.branch_name or "").strip():
        frappe.throw(_("No branch on this request."))

    if doc.status in BUSY_STATUSES:
        frappe.throw(_("Agent is busy (status: {0}).").format(doc.status))

    _require_request_branch(doc)

    push_branch = int(push_branch or 0)
    create_pr = int(create_pr or 0)
    if not push_branch and not create_pr:
        frappe.throw(_("Select at least one action (push branch or create PR)."))

    if not _has_pushable_changes(doc):
        return {"status": "noop", "message": _("No changes to push.")}

    frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Pushing")
    frappe.db.commit()

    frappe.enqueue(
        "ampower_koda.agent.executor.run_deploy_phase",
        queue="default",
        timeout=600,
        request_name=request_name,
        do_push=bool(push_branch),
        do_pr=bool(create_pr),
    )

    if push_branch and create_pr:
        msg = _("Committing, pushing, and creating PR...")
    elif push_branch:
        msg = _("Committing and pushing branch {0}...").format(doc.branch_name)
    else:
        msg = _("Committing and creating PR...")
    return {"status": "ok", "message": msg}


@frappe.whitelist()
@_whitelist_logged
def cancel_agent_request(request_name: str):
    """Cancel a queued or running request (best-effort)."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status in ("Completed", "Failed", "Cancelled"):
        return {"status": "noop", "message": _("Request already finished.")}

    frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Cancelled")
    frappe.db.commit()
    return {"status": "ok", "message": _("Request cancelled.")}
