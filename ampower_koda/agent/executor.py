# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Executor phases: planning (understand + plan), execution (implement + review),
# bench + commit, and deploy (push + PR). Enqueued as background jobs from api.py.

import datetime
import hashlib
import json
import os
import re
import subprocess

import frappe
from ampower_koda.agent.errors import log_agent_error
from ampower_koda.agent.graph import (
    _get_bench_env,
    _message_content_to_str,
    build_planning_graph,
    build_execution_graph,
)
from ampower_koda.agent.git_ops import (
    branch_exists,
    generate_branch_name,
    get_repo_root,
    get_current_branch,
    run_git,
    create_branch,
    commit_changes,
    push_branch,
    create_pull_request,
)

DOCTYPE_NAME = "Agent Request"


def _revert_previous_changes(app_name: str, base_branch: str, request_name: str = "",
                              user: str = "", branch_prefix: str = "ai-agent/"):
    """Revert all uncommitted changes and agent-created branches before a fresh run.
    Steps:
      1. Discard all modified/staged files (git reset --hard + git checkout .)
      2. Remove all untracked files (git clean -fd)
      3. If on an agent branch, switch back to base branch and delete the agent branch
    """
    if not app_name:
        return

    repo_root = get_repo_root(app_name)
    reverted_items = []

    current = get_current_branch(app_name)
    base = (base_branch or "main").strip()
    prefix = (branch_prefix or "ai-agent/").strip()
    is_agent_branch = (
        current and current != base
        and (current.startswith(prefix)
             or current.startswith("ai-agent/")
             or current.startswith("ai-agent-"))
    )

    if is_agent_branch:
        run_git(["reset", "--hard", "HEAD"], cwd=repo_root)
        run_git(["clean", "-fd"], cwd=repo_root)
        ok, _ = run_git(["checkout", base], cwd=repo_root)
        if ok:
            run_git(["branch", "-D", current], cwd=repo_root)
            reverted_items.append(f"switched from {current} → {base} and deleted agent branch")
        else:
            reverted_items.append(f"failed to switch from {current} to {base}")
    else:
        ok_diff, diff_out = run_git(["diff", "--stat"], cwd=repo_root)
        ok_staged, staged_out = run_git(["diff", "--cached", "--stat"], cwd=repo_root)
        ok_untracked, untracked_out = run_git(
            ["ls-files", "--others", "--exclude-standard"], cwd=repo_root
        )
        has_modifications = bool((diff_out or "").strip()) or bool((staged_out or "").strip())
        has_untracked = bool((untracked_out or "").strip())

        if has_modifications:
            run_git(["reset", "--hard", "HEAD"], cwd=repo_root)
            run_git(["checkout", "."], cwd=repo_root)
            reverted_items.append("discarded modified/staged files")

        if has_untracked:
            run_git(["clean", "-fd"], cwd=repo_root)
            reverted_items.append("removed untracked files")

    if reverted_items and request_name:
        summary = "Reverted previous changes: " + "; ".join(reverted_items)
        try:
            frappe.publish_realtime("agent_progress", {
                "request_name": request_name,
                "status": "Queued",
                "message": summary,
            }, user=user or "Administrator")
        except Exception:
            log_agent_error(
                "Agent Executor: revert publish",
                f"request={request_name}\n{frappe.get_traceback()}",
            )
        return summary

    return ""


#: The variables LangChain reads to decide whether, and where, to trace.
#: ``LANGCHAIN_*`` rather than the newer ``LANGSMITH_*`` spelling: both are
#: honoured by current releases, and the older pair is the one every installed
#: version understands.
LANGSMITH_ENV = {
    "enabled": "LANGCHAIN_TRACING_V2",
    "key": "LANGCHAIN_API_KEY",
    "project": "LANGCHAIN_PROJECT",
    "endpoint": "LANGCHAIN_ENDPOINT",
}


def _apply_langsmith(settings) -> None:
    """Turn tracing on or off in this worker, from settings.

    Background jobs do not inherit the web process's environment, so tracing has
    to be configured per job rather than once at boot.

    Both directions matter. A worker is long-lived and serves many requests, so
    the variables set for one of them are still set for the next — and tracing
    left on after it was switched off would keep shipping transcripts to an
    external service the site believes it has stopped using. Clearing is the half
    that makes the checkbox mean what it says.

    Never raises: tracing is an observability nicety, and a request that dies
    because its *tracer* could not be configured has failed for the least
    important reason available.
    """
    try:
        if not getattr(settings, "enable_langsmith", 0):
            for variable in LANGSMITH_ENV.values():
                os.environ.pop(variable, None)
            return

        api_key = (settings.get_password("langsmith_api_key", raise_exception=False) or "").strip()
        if not api_key:
            # Validation blocks this combination on save, but a row written
            # before the field existed can still reach here. Tracing without a
            # key silently drops every trace, so say so rather than pretend.
            for variable in LANGSMITH_ENV.values():
                os.environ.pop(variable, None)
            log_agent_error(
                "Agent LangSmith",
                "LangSmith tracing is enabled but no API key is set — tracing stays off.",
            )
            return

        os.environ[LANGSMITH_ENV["enabled"]] = "true"
        os.environ[LANGSMITH_ENV["key"]] = api_key
        os.environ[LANGSMITH_ENV["project"]] = (
            getattr(settings, "langsmith_project", "") or "Koda"
        ).strip()
        os.environ[LANGSMITH_ENV["endpoint"]] = (
            getattr(settings, "langsmith_endpoint", "") or "https://api.smith.langchain.com"
        ).strip().rstrip("/")
    except Exception:
        log_agent_error("Agent LangSmith", frappe.get_traceback())


def validate_target_app(app_name: str) -> str:
    """Fail on an app that is not on this bench, before anything touches git.

    Without this the first thing to notice is ``get_repo_root``, three frames
    inside the revert step, and what surfaces is ``ModuleNotFoundError: No
    module named 'ampower_visualize'`` — which reads like a broken install
    rather than a typo in a form field, and names nothing the user can act on.

    Checked against ``apps.txt`` rather than the site's *installed* list: an app
    can be present on the bench and legitimately not installed on this site, and
    the agent only ever reads and edits files.
    """
    app_name = (app_name or "").strip()
    if not app_name:
        frappe.throw("Target App Name is required.")

    available = sorted(frappe.get_all_apps(with_internal_apps=False))
    if app_name not in available:
        frappe.throw(
            f"App '{app_name}' is not on this bench. "
            f"Available apps: {', '.join(available) or '(none)'}."
        )
    return app_name


def _get_doc_config(request_name: str) -> dict:
    """Load the Agent Request document and return config needed for the graph state."""
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    settings = frappe.get_single("Agent Settings")

    if not settings.enable_ai_agent:
        frappe.throw("AI Agent is disabled in settings")

    provider = (doc.ai_provider or settings.default_ai_provider or "OpenAI").strip()

    provider_config = {
        "OpenAI": ("openai_api_key", "OPENAI_API_KEY", "OpenAI API key"),
        "Gemini": ("google_api_key", "GOOGLE_API_KEY", "Google API key"),
        "Claude": ("anthropic_api_key", "ANTHROPIC_API_KEY", "Anthropic API key"),
        "OpenRouter": ("openrouter_api_key", "OPENROUTER_API_KEY", "OpenRouter API key"),
    }
    cfg = provider_config.get(provider, provider_config["OpenAI"])
    field_name, env_var, label = cfg
    api_key = settings.get_password(field_name) or ""
    if not api_key.strip():
        frappe.throw(f"{label} not set in Agent Settings")
    os.environ[env_var] = api_key.strip()

    _apply_langsmith(settings)

    return {
        "doc": doc,
        "user": doc.owner or frappe.session.user,
        # Validated here as well as at the API, because a queued job can be
        # retried or replayed without passing back through start_agent.
        "target_app_name": validate_target_app(doc.target_app_name),
        "ai_provider": provider,
        "ai_model": (doc.ai_model or settings.default_ai_model or "gpt-4o-mini").strip(),
        "github_repo_url": (doc.github_repo_url or "").strip(),
        "github_token": (doc.get_password("github_token") or "").strip(),
        "base_branch": (doc.base_branch or "main").strip(),
        "branch_prefix": (doc.branch_prefix or "ai-agent/").strip(),
        "git_user_name": (doc.git_user_name or "AI Agent").strip(),
        "git_user_email": (doc.git_user_email or "ai-agent@ampower.com").strip(),
        "api_key": api_key,
    }


def _update_status(request_name: str, user: str, status: str, message: str = "", **kwargs):
    """Persist the status (plus any allowed fields) and broadcast progress via realtime."""
    frappe.db.set_value(DOCTYPE_NAME, request_name, "status", status)
    allowed_fields = [
        "branch_name", "pr_url", "pr_number", "conversation_log",
        "agent_plan", "files_changed", "error_log", "tokens_used",
        "cost_estimate", "stage_log", "bench_log", "patch_diff",
        "pending_bench_commands", "change_summary",
        "implementation_snapshot", "follow_up_message", "follow_up_count",
    ]
    if kwargs:
        for k, v in kwargs.items():
            if k in allowed_fields:
                frappe.db.set_value(DOCTYPE_NAME, request_name, k, v)
    frappe.db.commit()
    payload = {"request_name": request_name, "status": status, "message": message, **kwargs}
    frappe.publish_realtime("agent_progress", payload, user=user)


# ---------------------------------------------------------------------------
# Phase 1: Planning (Understand + Plan)
# ---------------------------------------------------------------------------

def run_planning_phase(request_name: str) -> None:
    """Run the planning phase (understand + plan) and pause for plan approval."""
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        log_agent_error("Agent Planning Config Error", frappe.get_traceback())
        frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Failed")
        frappe.db.set_value(DOCTYPE_NAME, request_name, "error_log", str(e))
        frappe.db.commit()
        return

    user = config["user"]
    doc = config["doc"]

    try:
        revert_msg = _revert_previous_changes(
            config["target_app_name"], config["base_branch"],
            request_name=request_name, user=user,
            branch_prefix=config["branch_prefix"],
        )
        if revert_msg:
            _update_status(request_name, user, "Queued", revert_msg)

        _update_status(request_name, user, "Understanding", "Exploring codebase...")

        graph = build_planning_graph()
        initial = {
            "user_message": doc.user_message or "",
            "request_type": doc.request_type or "Improvement",
            "request_name": request_name,
            "target_app_name": config["target_app_name"],
            "ai_provider": config["ai_provider"],
            "ai_model": config["ai_model"],
            "github_repo_url": config["github_repo_url"],
            "github_token": config["github_token"],
            "base_branch": config["base_branch"],
            "branch_prefix": config["branch_prefix"],
            "git_user_name": config["git_user_name"],
            "git_user_email": config["git_user_email"],
            "intermediate_steps": [],
            "edits_made": [],
            "stage_log": [],
        }

        final_state = graph.invoke(initial)

        if final_state.get("error"):
            _save_logs(request_name, final_state)
            _update_status(request_name, user, "Failed",
                final_state["error"],
                error_log=final_state.get("error_log") or final_state["error"])
            return

        plan = final_state.get("plan", "")
        _save_logs(request_name, final_state)

        understanding = _message_content_to_str(final_state.get("understanding_summary", "")).strip()
        plan_object = final_state.get("plan_object") or {}
        tasks = plan_object.get("tasks") or []
        if not tasks:
            raise ValueError("Planning completed without a structured task list")

        frappe.db.set_value(DOCTYPE_NAME, request_name, {
            "agent_plan": plan[:50000],
            "understanding_snapshot": understanding,
        })
        frappe.db.commit()

        approval_msg = (
            "Plan generated with %d task(s). Review them and approve to start implementation."
            % len(tasks)
        )
        _update_status(request_name, user, "Awaiting Approval", approval_msg,
            tokens_used=int(final_state.get("tokens_used") or 0))

    except Exception as e:
        tb = frappe.get_traceback()
        log_agent_error("Agent Planning Error", tb)
        _update_status(request_name, user, "Failed", str(e), error_log=tb)


# ---------------------------------------------------------------------------
# Phase 2: Execution (Implement + Review)
# ---------------------------------------------------------------------------

def run_execution_phase(request_name: str, preserve_branch: int = 0, is_follow_up: int = 0) -> None:
    """Create/reuse the working branch, run implement + review, then await bench approval.

    When preserve_branch is set, reuse the request's existing branch for a follow-up
    patch instead of creating a fresh one. is_follow_up enables patch-only mode
    without replacing the approved plan.
    """
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        log_agent_error("Agent Execution Config Error", frappe.get_traceback())
        frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Failed")
        frappe.db.set_value(DOCTYPE_NAME, request_name, "error_log", str(e))
        frappe.db.commit()
        return

    user = config["user"]
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)

    try:
        app_name = config["target_app_name"]
        keep_same_branch = bool(int(preserve_branch or 0))
        is_follow_up_mode = bool(int(is_follow_up or 0)) or keep_same_branch

        if keep_same_branch:
            branch_name = (doc.branch_name or "").strip()
            if not branch_name:
                _update_status(request_name, user, "Failed",
                    "Follow-up mode requires an existing branch on this request.",
                    error_log="Missing branch_name for follow-up execution.")
                return
            if not branch_exists(app_name, branch_name):
                _update_status(request_name, user, "Failed",
                    f"Follow-up branch '{branch_name}' was not found in local repo.",
                    error_log=f"Branch not found: {branch_name}")
                return

            repo_root = get_repo_root(app_name)
            current = get_current_branch(app_name)
            if current != branch_name:
                ok, msg = run_git(["checkout", branch_name], cwd=repo_root)
                if not ok:
                    _update_status(request_name, user, "Failed",
                        f"Could not checkout follow-up branch '{branch_name}': {msg}",
                        error_log=f"checkout failed: {msg}")
                    return
            # Preserve existing branch state — do NOT reset/clean or we wipe the
            # implementation the user is asking us to patch.
            _update_status(request_name, user, "Implementing",
                f"Follow-up patch on branch '{branch_name}' (plan preserved).",
                branch_name=branch_name)
        else:
            revert_msg = _revert_previous_changes(
                config["target_app_name"], config["base_branch"],
                request_name=request_name, user=user,
                branch_prefix=config["branch_prefix"],
            )
            if revert_msg:
                _update_status(request_name, user, "Implementing", f"Cleaned up: {revert_msg}")

            branch_name = generate_branch_name(request_name, config["branch_prefix"], app_name)
            ok, msg = create_branch(app_name, branch_name, config["base_branch"])
            if not ok:
                _update_status(request_name, user, "Failed",
                    f"Failed to create working branch: {msg}",
                    error_log=f"create_branch failed: {msg}")
                return
            _update_status(request_name, user, "Implementing",
                f"Created branch '{branch_name}'. Starting implementation...",
                branch_name=branch_name)

        plan = doc.agent_plan or ""
        prior_changed_paths = _prior_changed_paths(doc) if is_follow_up_mode else []
        implementation_memory = (
            (doc.implementation_snapshot or doc.change_summary or "").strip()
            if is_follow_up_mode else ""
        )
        follow_up_message = (doc.follow_up_message or "").strip() if is_follow_up_mode else ""

        # Carry the planning-phase stage log into execution for continuity.
        prev_stage_log = _parse_stage_log(doc.stage_log or "")

        repo_root = get_repo_root(app_name)
        worktree_before = _worktree_signature(repo_root) if is_follow_up_mode else ""

        graph = build_execution_graph()
        initial = {
            "user_message": doc.user_message or "",
            "request_type": doc.request_type or "Improvement",
            "request_name": request_name,
            "plan": plan,
            "understanding_summary": _extract_understanding(doc),
            "target_app_name": config["target_app_name"],
            "ai_provider": config["ai_provider"],
            "ai_model": config["ai_model"],
            "github_repo_url": config["github_repo_url"],
            "github_token": config["github_token"],
            "base_branch": config["base_branch"],
            "branch_prefix": config["branch_prefix"],
            "git_user_name": config["git_user_name"],
            "git_user_email": config["git_user_email"],
            "intermediate_steps": [],
            "edits_made": [],
            "stage_log": prev_stage_log,
            "tokens_used": int(doc.tokens_used or 0),
            "is_follow_up": is_follow_up_mode,
            "follow_up_message": follow_up_message,
            "prior_changed_paths": prior_changed_paths,
            "implementation_memory": implementation_memory,
        }

        final_state = graph.invoke(initial)

        _save_logs(request_name, final_state)

        if final_state.get("error"):
            _update_status(request_name, user, "Failed",
                final_state["error"],
                error_log=final_state.get("error_log") or final_state["error"])
            return

        repo_root = get_repo_root(app_name)

        # Drop throwaway notes/marker files the model may have created so they
        # never reach the diff or PR.
        stripped = _strip_agent_scratch_files(repo_root)
        if stripped:
            _update_status(request_name, user, "Implementing",
                f"Removed {len(stripped)} stray file(s): {', '.join(stripped[:5])}")

        worktree_after = _worktree_signature(repo_root) if is_follow_up_mode else ""
        run_made_changes = (not is_follow_up_mode) or (worktree_before != worktree_after)

        ok_diff, diff_out = run_git(["diff", "--stat"], cwd=repo_root)
        ok_ut, untracked = run_git(["ls-files", "--others", "--exclude-standard"], cwd=repo_root)
        has_changes = bool((diff_out or "").strip()) or bool((untracked or "").strip())

        if is_follow_up_mode and not run_made_changes:
            _update_status(request_name, user, "Failed",
                "Follow-up made no changes. The patch did not modify any files on disk.",
                error_log="Follow-up produced no net file changes.")
            return

        if not has_changes:
            _update_status(request_name, user, "Failed",
                "No code changes were produced. The implement phase did not modify any files on disk.",
                error_log="No file changes detected after implementation.")
            return

        patch_diff = _generate_patch_diff(app_name)

        # Keep the recorded change set in sync with what actually remains on disk.
        stripped_set = set(stripped)
        edits = [
            e for e in (final_state.get("edits_made") or [])
            if e.get("path") not in stripped_set
            and not (e.get("path") and _is_scratch_file(e["path"]))
        ]
        if is_follow_up_mode:
            edits = _merge_file_edits(as_json_list(doc.files_changed), edits)
        bench_cmds = _compute_bench_commands(app_name, edits)

        change_summary = (final_state.get("change_summary") or "").strip()
        if is_follow_up_mode and change_summary:
            prior_summary = (doc.change_summary or "").strip()
            if prior_summary:
                change_summary = f"{prior_summary}\n\n--- Follow-up #{doc.follow_up_count or 1} ---\n{change_summary}"

        _save_implementation_snapshot(request_name, doc, final_state, is_follow_up_mode)

        _update_status(
            request_name, user, "Awaiting Bench Approval",
            f"{'Follow-up patch' if is_follow_up_mode else 'Implementation'} complete. {len(bench_cmds)} bench commands need approval.",
            patch_diff=patch_diff,
            files_changed=dump_json_capped(edits),
            change_summary=change_summary,
            pending_bench_commands=json.dumps(bench_cmds),
            tokens_used=int(final_state.get("tokens_used") or 0),
        )

    except Exception as e:
        tb = frappe.get_traceback()
        log_agent_error("Agent Execution Error", tb)
        _update_status(request_name, user, "Failed", str(e), error_log=tb)


# ---------------------------------------------------------------------------
# Helpers: bench command computation
# ---------------------------------------------------------------------------

def _worktree_signature(repo_root: str) -> str:
    """Hash the full working-tree content vs HEAD to detect net file changes.

    Uses content diffs (not just file names) so a follow-up that patches a file
    already in the changed set is still detected. Includes untracked file
    contents so newly created files count too.
    """
    parts = []
    # Full content diff of tracked files (staged + unstaged) against HEAD.
    ok, out = run_git(["diff", "HEAD"], cwd=repo_root)
    if ok and out:
        parts.append(out)

    # Untracked files: include their contents, not just their names.
    ok, untracked = run_git(["ls-files", "--others", "--exclude-standard"], cwd=repo_root)
    if ok and untracked:
        for rel in sorted(untracked.splitlines()):
            rel = rel.strip()
            if not rel:
                continue
            parts.append(f"\n### UNTRACKED {rel}\n")
            try:
                with open(os.path.join(repo_root, rel), "r", encoding="utf-8", errors="replace") as fh:
                    parts.append(fh.read())
            except OSError:
                parts.append(f"(unreadable: {rel})")

    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _prior_changed_paths(doc) -> list:
    """Canonical paths from the previous run's files_changed list."""
    paths = []
    for row in as_json_list(doc.files_changed):
        if isinstance(row, dict):
            p = (row.get("path") or "").strip()
            if p and p not in paths:
                paths.append(p)
    return paths


def _merge_file_edits(prior: list, new: list) -> list:
    """Merge prior and new edit records, updating summaries for same paths."""
    merged = [dict(e) for e in prior if isinstance(e, dict)]
    index = {e.get("path"): i for i, e in enumerate(merged) if e.get("path")}
    for entry in new:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if path and path in index:
            merged[index[path]]["summary"] = entry.get("summary", merged[index[path]].get("summary"))
        elif path:
            merged.append(dict(entry))
            index[path] = len(merged) - 1
        else:
            merged.append(dict(entry))
    return merged


def _save_implementation_snapshot(request_name: str, doc, final_state: dict, is_follow_up: bool):
    """Remember what this run built so follow-ups can patch instead of re-implement."""
    summary = (final_state.get("change_summary") or "").strip()
    edits = final_state.get("edits_made") or []
    paths = [e.get("path") for e in edits if isinstance(e, dict) and e.get("path")]
    lines = []
    if summary:
        lines.append(summary)
    if paths:
        lines.append("\nFiles touched this run:")
        lines.extend(f"- {p}" for p in paths[:30])
    snapshot = "\n".join(lines).strip()
    if not snapshot:
        return
    prior = (doc.implementation_snapshot or "").strip()
    if is_follow_up and prior:
        snapshot = f"{prior}\n\n--- Follow-up run ---\n{snapshot}"
    frappe.db.set_value(DOCTYPE_NAME, request_name, {
        "implementation_snapshot": snapshot[:50000],
    })


# Names/patterns that indicate throwaway "scratch" files the model sometimes
# creates (progress notes, status markers, metadata dumps). These are never part
# of a real code change and must not land in the PR.
_SCRATCH_NAME_KEYWORDS = (
    "note", "notes", "metadata", "summary", "readme", "changelog",
    "finalize", "implementation_done", "done", "todo", "scratch",
)


def _is_scratch_file(rel_path: str) -> bool:
    """True for agent-created doc/marker files that don't belong in the change set."""
    name = os.path.basename(rel_path).lower()
    _, ext = os.path.splitext(name)
    # Loose .txt files are never a legitimate Frappe code artifact.
    if ext == ".txt":
        return True
    # Markdown only when it looks like an agent note/marker (keep real code .md rare).
    if ext == ".md" and any(k in name for k in _SCRATCH_NAME_KEYWORDS):
        return True
    if re.match(r"(?i)(implementation_done|finalize_|ai_feature|ai_progress)", name):
        return True
    return False


def _strip_agent_scratch_files(repo_root: str) -> list[str]:
    """Delete newly-created scratch/marker files from the working tree.

    Returns the list of removed repo-relative paths. Only untracked files are
    considered, so tracked source code is never touched.
    """
    ok, untracked = run_git(["ls-files", "--others", "--exclude-standard"], cwd=repo_root)
    if not ok or not (untracked or "").strip():
        return []
    removed = []
    for rel in untracked.splitlines():
        rel = rel.strip()
        if not rel or not _is_scratch_file(rel):
            continue
        try:
            os.remove(os.path.join(repo_root, rel))
            removed.append(rel)
        except OSError:
            log_agent_error("Agent Execution: strip scratch file",
                f"could not remove {rel}\n{frappe.get_traceback()}")
    return removed


def _compute_bench_commands(app_name: str, edits: list) -> list[str]:
    """Determine which bench commands are needed based on which file types were edited.
    Always includes clear-cache and supervisorctl restart."""
    edited_paths = [e.get("path", "") for e in edits if e.get("path")]
    site_name = frappe.local.site

    has_doctype_changes = any(
        p.endswith(".json") and "/doctype/" in p for p in edited_paths
    )
    has_report_changes = any(
        p.endswith(".json") and "/report/" in p for p in edited_paths
    )
    has_js_css_changes = any(
        p.endswith((".js", ".css", ".html")) for p in edited_paths
    )

    cmds = []
    if has_doctype_changes or has_report_changes:
        cmds.append(f"bench --site {site_name} migrate")
    if has_js_css_changes:
        cmds.append(f"bench build --app {app_name}")
    cmds.append(f"bench --site {site_name} clear-cache")
    cmds.append("supervisorctl restart all")
    return cmds


# ---------------------------------------------------------------------------
# Phase 2b: Bench + Commit — Run bench commands, then branch+commit
# ---------------------------------------------------------------------------

def _publish_bench_log(user, request_name, cmd, success, output_preview=""):
    """Broadcast a bench command start/result event via Frappe realtime."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frappe.publish_realtime("agent_log", {
        "request_name": request_name,
        "type": "bench_command",
        "command": cmd,
        "timestamp": ts,
    }, user=user)
    frappe.publish_realtime("agent_log", {
        "request_name": request_name,
        "type": "bench_result",
        "success": success,
        "output_preview": (output_preview or "")[:180],
        "timestamp": ts,
    }, user=user)


def run_bench_and_commit(request_name: str) -> None:
    """Run the approved bench commands, then pause for push approval so the user can test."""
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        log_agent_error("Agent Bench Config Error", frappe.get_traceback())
        frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Failed")
        frappe.db.set_value(DOCTYPE_NAME, request_name, "error_log", str(e))
        frappe.db.commit()
        return

    user = config["user"]
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    app_name = config["target_app_name"]
    branch_name = (doc.branch_name or "").strip()

    try:
        _update_status(request_name, user, "Building", "Running bench commands...")

        cmds_json = doc.pending_bench_commands or "[]"
        try:
            cmds = json.loads(cmds_json)
        except (json.JSONDecodeError, TypeError):
            cmds = []

        if not cmds:
            cmds = _compute_bench_commands(app_name, [])


        bench_root = os.path.join(frappe.get_app_path("frappe"), "..", "..", "..")
        bench_root = os.path.normpath(bench_root)
        bench_env = _get_bench_env()

        deferred_cmds = []
        immediate_cmds = []
        for cmd in cmds:
            if "supervisorctl" in cmd.lower():
                deferred_cmds.append(cmd)
            else:
                immediate_cmds.append(cmd)

        bench_output_parts = []
        failed_cmds = []
        for cmd in immediate_cmds:
            _publish_bench_log(user, request_name, cmd, True, "Running...")
            try:
                result = subprocess.run(
                    cmd.split(),
                    cwd=bench_root,
                    capture_output=True,
                    text=True,
                    timeout=900,
                    env=bench_env,
                )
                out = (result.stdout or "") + (result.stderr or "")
                ok = result.returncode == 0
                status_str = "OK" if ok else f"FAILED (exit {result.returncode})"
                bench_output_parts.append(f"$ {cmd}\n{status_str}\n{out.strip()}\n")
                _publish_bench_log(user, request_name, cmd, ok, out[:180])
                if not ok:
                    failed_cmds.append(f"{cmd} (exit {result.returncode})")
            except subprocess.TimeoutExpired:
                bench_output_parts.append(f"$ {cmd}\nTIMEOUT after 900s\n")
                _publish_bench_log(user, request_name, cmd, False, "TIMEOUT after 900s")
                failed_cmds.append(f"{cmd} (timeout)")
                log_agent_error(
                    "Agent Executor: bench command timeout",
                    f"request={request_name}\ncmd={cmd}",
                )
            except Exception as e:
                bench_output_parts.append(f"$ {cmd}\nERROR: {e}\n")
                _publish_bench_log(user, request_name, cmd, False, str(e))
                failed_cmds.append(f"{cmd} ({e})")
                log_agent_error(
                    "Agent Executor: bench command",
                    f"request={request_name}\ncmd={cmd}\n{e}\n{frappe.get_traceback()}",
                )

        if deferred_cmds:
            for cmd in deferred_cmds:
                bench_output_parts.append(f"$ {cmd}\n(deferred — runs after status update)\n")

        bench_log = "\n".join(bench_output_parts)

        # `ok` was computed per command, written into the log text, published to
        # the form — and then dropped. A failing `bench migrate` still arrived
        # here as "Bench commands done", so the status said the build succeeded
        # and only the log said otherwise. The pause for human testing is kept
        # either way: a failed command is exactly when someone should look before
        # pushing, and marking the request Failed would take that choice away.
        extra = {"bench_log": bench_log[:50000]}
        if failed_cmds:
            message = (
                f"{len(failed_cmds)} of {len(immediate_cmds)} bench command(s) FAILED on "
                f"branch '{branch_name}': {', '.join(failed_cmds[:3])}"
                f"{'…' if len(failed_cmds) > 3 else ''}. "
                "Check the bench log before approving push."
            )
            extra["error_log"] = "\n".join(failed_cmds)
        else:
            message = (
                f"Bench commands done on branch '{branch_name}'. "
                "Test the changes, then approve push to commit and push."
            )

        _update_status(
            request_name, user, "Awaiting Push Approval", message, **extra
        )

        frappe.db.commit()

        for cmd in deferred_cmds:
            try:
                subprocess.Popen(
                    cmd.split(),
                    cwd=bench_root,
                    env=bench_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                log_agent_error(
                    "Agent Executor: deferred supervisorctl",
                    f"request={request_name}\ncmd={cmd}\n{e}\n{frappe.get_traceback()}",
                )

    except Exception as e:
        tb = frappe.get_traceback()
        log_agent_error("Agent Bench+Commit Error", tb)
        _update_status(request_name, user, "Failed", str(e), error_log=tb)


# ---------------------------------------------------------------------------
# Phase 3: Deployment (Push + Pull Request)
# ---------------------------------------------------------------------------

def run_deploy_phase(request_name: str, do_push: bool = True, do_pr: bool = True) -> None:
    """Commit the changes, then optionally push the branch and open a pull request."""
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        log_agent_error("Agent Deploy Config Error", frappe.get_traceback())
        frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Failed")
        frappe.db.set_value(DOCTYPE_NAME, request_name, "error_log", str(e))
        frappe.db.commit()
        return

    user = config["user"]
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    branch_name = (doc.branch_name or "").strip()
    app_name = config["target_app_name"]

    if not branch_name:
        branch_name = generate_branch_name(request_name, config["branch_prefix"])

    try:
        current = get_current_branch(app_name)
        if current != branch_name:
            repo_root = get_repo_root(app_name)
            ok, out = run_git(["checkout", branch_name], cwd=repo_root)
            if not ok:
                _update_status(request_name, user, "Failed",
                    f"Could not checkout branch '{branch_name}': {out}",
                    error_log=f"checkout failed: {out}")
                return

        _update_status(request_name, user, "Pushing", "Committing changes...")
        user_msg = (doc.user_message or "")[:200]
        commit_msg = f"[AI Agent] {doc.request_type or 'Improvement'}: {request_name}\n\n{user_msg}"
        ok, msg = commit_changes(
            app_name, commit_msg,
            config["git_user_name"], config["git_user_email"],
        )
        if not ok:
            # Nothing new to commit. Only treat this as a failure if there is also
            # nothing already committed on the branch to push. This is a safety net
            # for races; ide_push guards the no-changes case first.
            if "no changes to commit" in (msg or "").lower():
                if _branch_has_commits_vs_base(app_name, config["base_branch"], branch_name):
                    # There are existing commits worth pushing/PRing; keep going.
                    pass
                else:
                    status = "Completed" if (doc.pr_url or "").strip() else "Awaiting Push Approval"
                    _update_status(request_name, user, status, "No changes to push.")
                    return
            else:
                _update_status(request_name, user, "Failed",
                    f"Failed to commit changes: {msg}",
                    error_log=f"commit_changes failed: {msg}")
                return

        pr_url = None
        pr_number = None

        if do_push:
            _update_status(request_name, user, "Pushing", f"Pushing branch '{branch_name}' to GitHub...")
            ok, msg = push_branch(
                app_name, branch_name,
                config["github_repo_url"], config["github_token"],
            )
            if not ok:
                _update_status(request_name, user, "Failed",
                    f"Push failed: {msg}", error_log=f"push_branch: {msg}")
                return

        if do_pr:
            _update_status(request_name, user, "Pushing", "Creating pull request...")
            user_message = (doc.user_message or "")[:500]
            pr_title = f"[AI Agent] {doc.request_type or 'Improvement'}: {request_name}"
            pr_body = f"## Request\n{user_message}\n\n## Plan\n{doc.agent_plan or ''}"
            ok, msg, pr_url, pr_number = create_pull_request(
                pr_title, pr_body, branch_name,
                config["github_repo_url"], config["github_token"],
                config["base_branch"],
            )
            if not ok:
                _update_status(request_name, user, "Failed",
                    f"PR creation failed: {msg}", error_log=f"create_pull_request: {msg}")
                return

        summary_parts = []
        if do_push:
            summary_parts.append(f"Branch '{branch_name}' pushed")
        if do_pr and pr_number:
            summary_parts.append(f"PR #{pr_number} created")

        extra = {"branch_name": branch_name}
        if pr_url:
            extra["pr_url"] = pr_url
        if pr_number:
            extra["pr_number"] = pr_number

        _update_status(
            request_name, user, "Completed",
            " | ".join(summary_parts) or "Deploy completed",
            **extra,
        )

    except Exception as e:
        tb = frappe.get_traceback()
        log_agent_error("Agent Deploy Error", tb)
        _update_status(request_name, user, "Failed", str(e), error_log=tb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _branch_has_commits_vs_base(app_name: str, base_branch: str, branch_name: str) -> bool:
    """True if `branch_name` has commits that `base_branch` does not (local only)."""
    if not (app_name and base_branch and branch_name):
        return False
    try:
        repo_root = get_repo_root(app_name)
        ok, out = run_git(["rev-list", "--count", f"{base_branch}..{branch_name}"], cwd=repo_root)
        return ok and out.strip().isdigit() and int(out.strip()) > 0
    except Exception:
        log_agent_error("Agent Deploy: rev-list vs base", frappe.get_traceback())
        return False


def _generate_patch_diff(app_name: str) -> str:
    """Generate a unified diff of all uncommitted changes in the target app repo."""
    try:
        repo_root = get_repo_root(app_name)
        ok_staged, staged = run_git(["diff", "--cached"], cwd=repo_root)
        ok_unstaged, unstaged = run_git(["diff"], cwd=repo_root)
        ok_untracked, untracked_files = run_git(
            ["ls-files", "--others", "--exclude-standard"], cwd=repo_root
        )
        parts = []
        if ok_staged and staged.strip():
            parts.append(staged.strip())
        if ok_unstaged and unstaged.strip():
            parts.append(unstaged.strip())
        if ok_untracked and untracked_files.strip():
            for fpath in untracked_files.strip().split("\n"):
                fpath = fpath.strip()
                if not fpath:
                    continue
                full = os.path.join(repo_root, fpath)
                try:
                    with open(full, "r", errors="replace") as f:
                        content = f.read(50000)
                    parts.append(f"--- /dev/null\n+++ b/{fpath}\n" +
                                 "\n".join(f"+{line}" for line in content.split("\n")))
                except Exception as e:
                    log_agent_error(
                        "Agent Executor: patch diff file read",
                        f"app={app_name}\npath={fpath}\n{e}\n{frappe.get_traceback()}",
                    )
                    parts.append(f"--- /dev/null\n+++ b/{fpath}\n+[binary or unreadable]")
        return "\n\n".join(parts)[:100000]
    except Exception as e:
        log_agent_error(
            "Agent Executor: generate patch diff",
            f"app={app_name}\n{e}\n{frappe.get_traceback()}",
        )
        return f"(could not generate diff: {e})"


def _save_logs(request_name: str, final_state: dict):
    """Persist stage_log and append this run's conversation_log to the document."""
    stage_logs = final_state.get("stage_log") or []
    if isinstance(stage_logs, list):
        stage_text = "\n".join(
            f"[{l.get('timestamp', '')}] {l.get('stage', '')} - {l.get('status', '')}: {l.get('summary', '')}"
            for l in stage_logs
        )
    else:
        stage_text = str(stage_logs)

    try:
        new_block = _format_conversation_log(final_state.get("intermediate_steps", []))
    except Exception as e:
        log_agent_error(
            "Agent Executor: serialize conversation log",
            f"request={request_name}\n{e}\n{frappe.get_traceback()}",
        )
        new_block = f"Could not format conversation log: {e}"

    conversation_log = _append_conversation_log(request_name, new_block)

    try:
        frappe.db.set_value(DOCTYPE_NAME, request_name, {
            "stage_log": stage_text[:50000],
            "conversation_log": conversation_log,
        })
        frappe.db.commit()
    except Exception as e:
        log_agent_error(
            "Agent Executor: save logs",
            f"request={request_name}\n{e}\n{frappe.get_traceback()}",
        )
        raise


def _parse_stage_log(stage_log_text: str) -> list[dict]:
    """Parse stored stage log text back into list of dicts for graph state continuity."""
    if not stage_log_text:
        return []
    entries = []
    for line in stage_log_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            ts_end = line.index("]")
            timestamp = line[1:ts_end]
            rest = line[ts_end + 2:]
            parts = rest.split(" - ", 1)
            stage = parts[0].strip()
            status_summary = parts[1] if len(parts) > 1 else ""
            sp = status_summary.split(": ", 1)
            status = sp[0].strip()
            summary = sp[1].strip() if len(sp) > 1 else ""
            entries.append({
                "stage": stage,
                "status": status,
                "summary": summary,
                "timestamp": timestamp,
            })
        except (ValueError, IndexError):
            continue
    return entries


def dump_json_capped(obj, limit: int = 50000) -> str:
    """Serialize obj to valid JSON no longer than `limit` chars.

    JSON DocType columns enforce json_valid(); naive string truncation would
    corrupt the JSON and fail the constraint. If the full dump is too large, long
    string values are shortened so the result stays valid JSON.
    """
    text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text

    def shrink(value, budget):
        if isinstance(value, str) and len(value) > budget:
            return value[:budget] + "…[truncated]"
        if isinstance(value, list):
            return [shrink(v, budget) for v in value]
        if isinstance(value, dict):
            return {k: shrink(v, budget) for k, v in value.items()}
        return value

    budget = 4000
    while budget >= 200:
        candidate = json.dumps(shrink(obj, budget), indent=2, ensure_ascii=False, default=str)
        if len(candidate) <= limit:
            return candidate
        budget //= 2

    return json.dumps(
        {"_truncated": True, "note": "Log too large to store."},
        ensure_ascii=False,
    )


def as_json_list(value) -> list:
    """Coerce a JSON-typed field into a list.

    Tolerates None, empty string, a JSON string, or an already-parsed list/dict —
    JSON DocType fields may surface as either a raw string or a parsed value.
    """
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return []
        if isinstance(parsed, list):
            return parsed
        return [parsed] if parsed else []
    return []


_PHASE_MARKER = "===== PHASE: "
_RUN_MARKER = "========== RUN @ "
# Total conversation_log kept per request across all runs. Very high on purpose:
# each run is naturally bounded, so this only guards pathological growth.
_CONVERSATION_LOG_LIMIT = 1_500_000


def _format_conversation_log(steps, limit: int = 500000) -> str:
    """Render agent steps as clean, human-readable sectioned text (no JSON escaping)."""
    blocks = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        phase = step.get("phase", "Step")
        output = _message_content_to_str(step.get("output", "")).strip()
        blocks.append(f"{_PHASE_MARKER}{phase} =====\n{output}")
    text = "\n\n".join(blocks)
    if len(text) > limit:
        text = text[:limit] + "\n\n… [log truncated]"
    return text


def _append_conversation_log(request_name: str, new_block: str) -> str:
    """Append this run's log block to the existing conversation_log.

    The log accumulates across the request lifecycle (planning run -> execution
    run -> follow-up runs) so the full conversation is retained. start_agent
    clears it for a fresh from-scratch run.
    """
    existing = (frappe.db.get_value(DOCTYPE_NAME, request_name, "conversation_log") or "").rstrip()
    if not (new_block or "").strip():
        return existing

    header = f"{_RUN_MARKER}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} ==========\n\n"
    block = header + new_block
    combined = f"{existing}\n\n{block}" if existing else block

    if len(combined) > _CONVERSATION_LOG_LIMIT:
        # Keep the most recent content; drop oldest runs (understanding is also
        # persisted separately, so extraction is unaffected by this rare trim).
        combined = "… [older runs truncated]\n\n" + combined[-_CONVERSATION_LOG_LIMIT:]
    return combined


def _extract_understanding(doc) -> str:
    """Extract the Understanding section from the conversation log.

    Prefers the dedicated understanding_snapshot field (stable across runs and log
    trimming). Falls back to parsing the clean sectioned-text conversation log, then
    to the legacy JSON format for requests logged before the format change.
    """
    snapshot = (getattr(doc, "understanding_snapshot", "") or "").strip()
    if snapshot:
        return snapshot

    text = doc.conversation_log or ""
    try:
        if text.strip():
            marker = f"{_PHASE_MARKER}Understanding ====="
            idx = text.find(marker)
            if idx != -1:
                start = idx + len(marker)
                # Stop at the next phase or run boundary, whichever comes first.
                candidates = [
                    pos for pos in (
                        text.find(f"\n{_PHASE_MARKER}", start),
                        text.find(f"\n{_RUN_MARKER}", start),
                    ) if pos != -1
                ]
                next_idx = min(candidates) if candidates else -1
                section = text[start:] if next_idx == -1 else text[start:next_idx]
                return section.strip()

            # Legacy JSON-formatted logs.
            for step in as_json_list(text):
                if isinstance(step, dict) and step.get("phase") == "Understanding":
                    return _message_content_to_str(step.get("output", ""))
    except Exception as e:
        log_agent_error(
            "Agent Executor: extract understanding",
            f"request={getattr(doc, 'name', '')}\n{e}\n{frappe.get_traceback()}",
        )
    return ""
