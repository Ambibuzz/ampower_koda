# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Background jobs enqueued from api.py: planning, execution, bench + commit, deploy.

import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import uuid

import frappe
from ampower_koda.agent import koda_core
from ampower_koda.agent.errors import log_agent_error
from ampower_koda.agent.execution_contract import load_plan, read_snapshot
from ampower_koda.agent import session as koda_session
from ampower_koda.agent import tools as agent_tools
from ampower_koda.agent import render_check, verification
from ampower_koda.agent.checkpoint import ExecutionJournal, restore_checkpoint, cleanup_temporaries
from ampower_koda.agent.run_control import (
    managed_job, current_run, check_active, set_request_value, run_context, RunStopped,
)
from ampower_koda.agent.graph import _get_bench_env, _message_content_to_str
from ampower_koda.agent.git_ops import (
    KODA_CLEAN_EXCLUDES,
    branch_exists,
    generate_branch_name,
    get_pull_request,
    get_repo_root,
    get_current_branch,
    ignored_regression_tests,
    run_git,
    run_git_stdout,
    create_branch,
    commit_changes,
    push_branch,
    create_pull_request,
    worktree_signature,
)

DOCTYPE_NAME = "Agent Request"


#: Files under .koda/ that are the team's configuration, not Koda's generated state.
_KODA_CONFIG_FILES = {"config.toml", "verification.json", "verification.md"}


def _is_request_branch(branch: str, request_name: str, branch_prefix: str, recorded: str = "") -> bool:
    """True if ``branch`` is this request's own working branch.

    That is the branch recorded on the request, or a name this request's naming
    scheme produces (generate_branch_name: the base name or its _vN variants).
    A shared prefix alone never makes a branch this request's.
    """
    if not branch:
        return False
    if recorded and branch == recorded:
        return True
    if not request_name:
        return False
    own = generate_branch_name(request_name, branch_prefix)
    return branch == own or re.fullmatch(re.escape(own) + r"_v\d+", branch) is not None


def _uncommitted_paths(repo_root: str) -> list[str]:
    """Tracked changes and untracked files in the checkout, excluding Koda's own .koda/ state.

    Untracked files inside a .koda/ directory (index cache, render output,
    leftover or archived agent tests) are Koda's; its config files still count.
    """
    ok, out = run_git_stdout(["status", "--porcelain", "-z", "--untracked-files=all"], cwd=repo_root)
    if not ok:
        raise RuntimeError("Could not read the working-tree status of the target app checkout.")
    paths, entries = [], iter(out.split("\0"))
    for entry in entries:
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if code[0] in "RC":
            next(entries, None)  # the rename/copy source follows as its own entry
        parts = path.rstrip("/").split("/")
        if code == "??" and ".koda" in parts[:-1] and parts[-1] not in _KODA_CONFIG_FILES:
            continue
        paths.append(path)
    return paths


def bench_branch_refusal(app_name: str, branch_name: str, allowed=()) -> str:
    """Why bench commands must not run for this request now, or "" when they may.

    Checked immediately before the commands: they act on whatever is checked
    out, so the checkout must be the request's branch (or a branch in ``allowed``).
    """
    branch_name = (branch_name or "").strip()
    current = get_current_branch(app_name)
    if current and (current == branch_name or current in allowed) and current != "HEAD":
        return ""
    expected = " or ".join(f"'{b}'" for b in (branch_name, *allowed) if b) or "(none recorded)"
    return (f"Bench commands were not run: the checkout is on '{current or '(unknown)'}', not "
            f"{expected} of this request. Check out the request's branch, then run the commands again.")


def _revert_previous_changes(app_name: str, base_branch: str, request_name: str = "",
                              user: str = "", branch_prefix: str = "ai-agent/", *,
                              own_branch: str = "", archive_tests: bool = True):
    """Return the checkout to the base branch before a fresh run of this request.

    Only this request's own work is discarded. On this request's own branch
    (``own_branch`` or its naming scheme), uncommitted changes are reverted and
    the branch is deleted. Any other branch is left intact; the checkout only
    switches to the base, and uncommitted changes there (user work, another
    request's pending implementation) stop the run with the files named.
    """
    if not app_name:
        return

    repo_root = get_repo_root(app_name)
    reverted_items = []

    current = get_current_branch(app_name)
    base = (base_branch or "main").strip()
    is_own_branch = current != base and _is_request_branch(current, request_name, branch_prefix, own_branch)

    dirty = _uncommitted_paths(repo_root)
    if dirty and not is_own_branch:
        shown = ", ".join(dirty[:10]) + (f" (and {len(dirty) - 10} more)" if len(dirty) > 10 else "")
        raise ValueError(
            f"The '{app_name}' checkout has uncommitted changes on branch '{current or '(unknown)'}' "
            f"that do not belong to this request: {shown}. Commit or stash them, then start again. "
            "Nothing was reverted."
        )

    if is_own_branch:
        for cmd in (["reset", "--hard", "HEAD"], ["clean", "-fd", *KODA_CLEAN_EXCLUDES]):
            ok, out = run_git(cmd, cwd=repo_root)
            if not ok:
                raise RuntimeError(f"git {cmd[0]} failed while reverting '{current}': {out}")
        if dirty:
            reverted_items.append(f"discarded this request's uncommitted changes on {current}")
    if current != base:
        ok, out = run_git(["checkout", base], cwd=repo_root)
        if not ok:
            raise RuntimeError(f"Could not switch from '{current}' to '{base}': {out}")
        if is_own_branch:
            ok, out = run_git(["branch", "-D", current], cwd=repo_root)
            reverted_items.append(f"switched from {current} → {base} and deleted this request's earlier branch"
                                  if ok else f"switched from {current} → {base}; could not delete it: {out}")
        else:
            reverted_items.append(f"switched from {current} → {base} (branch '{current}' kept)")

    # Cleanup never removes `.koda/` (KODA_CLEAN_EXCLUDES), so an earlier
    # run's untracked tests would otherwise become this run's frozen contract.
    # Only planning archives: an approved plan was written against the tests
    # present when it was made, and executing it must not remove them.
    archived = []
    try:
        if archive_tests:
            archived = verification.archive_untracked_tests(app_name, label=request_name)
    except (OSError, ValueError):
        log_agent_error("Agent Executor: archive leftover tests", frappe.get_traceback())
    if archived:
        reverted_items.append(f"archived {len(archived)} leftover agent test file(s) to "
                              f"{verification.ARCHIVE_DIRECTORY}")

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


#: The variables LangChain reads to decide whether, and where, to trace, in both
#: spellings. The SDK reads ``LANGSMITH_*`` before ``LANGCHAIN_*``, so a worker
#: started with the newer one would otherwise override whatever is set here.
LANGSMITH_ENV = {
    "enabled": ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING"),
    "key": ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
    "project": ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
    "endpoint": ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"),
}


def _set_langsmith_env(role: str, value) -> None:
    """Set every spelling of one tracing variable, or remove them all for None."""
    for variable in LANGSMITH_ENV[role]:
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value


def _disable_langsmith() -> None:
    """Remove the trace destination and explicitly switch both enable flags off."""
    for role in LANGSMITH_ENV:
        _set_langsmith_env(role, None)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def _reset_langsmith_cache() -> None:
    """The SDK caches environment reads per process; a reused worker must re-read them."""
    try:
        from langsmith import utils as langsmith_utils
    except ImportError:
        return
    for name in ("get_env_var", "get_tracer_project"):
        cache_clear = getattr(getattr(langsmith_utils, name, None), "cache_clear", None)
        if cache_clear:
            cache_clear()


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
            _disable_langsmith()
            return

        api_key = (settings.get_password("langsmith_api_key", raise_exception=False) or "").strip()
        if not api_key:
            # Validation blocks this combination on save, but a row written
            # before the field existed can still reach here. Tracing without a
            # key silently drops every trace, so say so rather than pretend.
            _disable_langsmith()
            log_agent_error(
                "Agent LangSmith",
                "LangSmith tracing is enabled but no API key is set — tracing stays off.",
            )
            return

        _set_langsmith_env("enabled", "true")
        _set_langsmith_env("key", api_key)
        _set_langsmith_env("project", (
            getattr(settings, "langsmith_project", "") or "Koda"
        ).strip())
        _set_langsmith_env("endpoint", (
            getattr(settings, "langsmith_endpoint", "") or "https://api.smith.langchain.com"
        ).strip().rstrip("/"))
    except Exception:
        log_agent_error("Agent LangSmith", frappe.get_traceback())
    finally:
        _reset_langsmith_cache()


def validate_target_app(app_name: str) -> str:
    """Fail on an app that is not on this bench, before anything touches git.

    Checked against the bench's apps, not the site's installed list: the agent
    only reads and edits files.
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
    set_request_value(request_name, "status", status)
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
                set_request_value(request_name, k, v)
    frappe.db.commit()
    check_active()
    payload = {"request_name": request_name, "status": status, "message": message, **kwargs}
    frappe.publish_realtime("agent_progress", payload, user=user)


def _config_failed(request_name: str, title: str, error: Exception) -> None:
    """Fail a job whose configuration could not load, through the normal status path.

    Setting the fields without publishing left a request that was still Queued in
    an open form showing Queued until someone reloaded it.
    """
    log_agent_error(title, frappe.get_traceback())
    owner = frappe.db.get_value(DOCTYPE_NAME, request_name, "owner") or "Administrator"
    _update_status(request_name, owner, "Failed", f"Configuration error: {error}", error_log=str(error))


# Phase 1: Planning (investigate + plan, or revise the plan from the user's feedback)

@managed_job
def run_planning_phase(request_name: str, plan_feedback: str = "") -> None:
    """Investigate and plan in the request's session, then pause for plan approval.

    ``plan_feedback``, the user's answer to the plan, continues the same investigation.
    """
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        _config_failed(request_name, "Agent Planning Config Error", e)
        return

    user = config["user"]
    doc = config["doc"]
    plan_feedback = (plan_feedback or "").strip()

    try:
        if plan_feedback:
            # Nothing was implemented while the plan awaited approval.
            _update_status(request_name, user, "Understanding", "Revising the plan from your feedback...")
        else:
            revert_msg = _revert_previous_changes(
                config["target_app_name"], config["base_branch"],
                request_name=request_name, user=user,
                branch_prefix=config["branch_prefix"],
                own_branch=(doc.branch_name or "").strip(),
            )
            if revert_msg:
                _update_status(request_name, user, "Queued", revert_msg)
            gaps = render_check.missing_tooling()
            _update_status(request_name, user, "Understanding", "Exploring codebase..." + (
                " Some checks will end UNRESOLVED until this bench is set up: " + "; ".join(gaps) + "."
                if gaps else ""))

        graph = koda_session.build_planning_graph()
        initial = {
            "user_message": doc.user_message or "",
            "request_type": doc.request_type or "Improvement",
            "request_name": request_name,
            "target_app_name": config["target_app_name"],
            "ai_provider": config["ai_provider"],
            "ai_model": config["ai_model"],
            "github_repo_url": config["github_repo_url"],
            "base_branch": config["base_branch"],
            "branch_prefix": config["branch_prefix"],
            "git_user_name": config["git_user_name"],
            "git_user_email": config["git_user_email"],
            "intermediate_steps": [],
            "edits_made": [],
            "stage_log": [],
        }
        if plan_feedback:
            initial.update({
                "plan_feedback": plan_feedback,
                "plan_object": json.loads(doc.plan_json or "{}"),
                "stage_log": _parse_stage_log(doc.stage_log or ""),
                "tokens_used": int(doc.tokens_used or 0),
            })

        final_state = graph.invoke(initial)

        if final_state.get("error"):
            _save_logs(request_name, final_state)
            if plan_feedback and doc.plan_json:
                # The plan being answered is still valid: offer it again with the error.
                _update_status(request_name, user, "Awaiting Approval",
                    "The plan could not be revised; the previous plan is unchanged: " + final_state["error"],
                    error_log=final_state.get("error_log") or final_state["error"])
                return
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

        set_request_value(request_name, {
            "agent_plan": plan,
            "plan_json": json.dumps(plan_object, ensure_ascii=True),
            "approved_plan_json": "",
            "execution_results": "",
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


# Phase 2: Execution (Implement + Review)


def restore_execution_state(doc, plan_object: dict, app_name: str) -> dict:
    """Validate and restore one request checkpoint without mutating the checkout."""
    branch_name = (doc.branch_name or "").strip()
    if not branch_name or get_current_branch(app_name) != branch_name:
        raise ValueError("Resume requires the request's existing branch to be checked out. No checkout or cleanup was performed.")
    ok, head = run_git(["rev-parse", "HEAD"], cwd=get_repo_root(app_name))
    if not ok:
        raise ValueError("Could not verify HEAD for resume: " + head)
    restored = restore_checkpoint(
        doc.get("execution_checkpoint"),
        plan=plan_object,
        branch=branch_name,
        head=head.strip(),
        read_current=lambda path: read_snapshot(agent_tools._resolve_path(app_name, path)),
    )
    if restored.get("target_app_name") != app_name or restored.get("user_message", "") != (doc.user_message or ""):
        raise ValueError("Request configuration changed since the checkpoint; start a newly approved run.")
    if restored.get("is_follow_up") and restored.get("follow_up_message", "") != (doc.follow_up_message or "").strip():
        raise ValueError("Follow-up instructions changed since the checkpoint; submit a new follow-up.")
    return restored


@managed_job
def run_execution_phase(request_name: str, preserve_branch: int = 0, is_follow_up: int = 0,
                        resume: int = 0) -> None:
    """Create/reuse the working branch, run implement + review, then await bench approval.

    When preserve_branch is set, reuse the request's existing branch for a follow-up
    patch instead of creating a fresh one. is_follow_up enables patch-only mode
    without replacing the approved plan.
    """
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        _config_failed(request_name, "Agent Execution Config Error", e)
        return

    user = config["user"]
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)

    try:
        app_name = config["target_app_name"]
        keep_same_branch = bool(int(preserve_branch or 0))
        is_follow_up_mode = bool(int(is_follow_up or 0)) or keep_same_branch
        resume_mode = bool(int(resume or 0))
        restored = None

        # Validate the persisted approval before any checkout/reset or file mutation.
        approved = doc.get("approved_plan_json")
        plan_object = load_plan(approved)

        if resume_mode:
            branch_name = (doc.branch_name or "").strip()
            restored = restore_execution_state(doc, plan_object, app_name)
            check_active(reserve=5)
            cleanup_temporaries(restored.pop("_checkpoint_temporaries", []), lambda p: agent_tools._resolve_path(app_name, p))
            is_follow_up_mode = bool(restored.get("is_follow_up"))
            _update_status(request_name, user, "Implementing", "Resuming saved execution on the existing branch.")
        elif keep_same_branch:
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
                branch_prefix=config["branch_prefix"], archive_tests=False,
                own_branch=(doc.branch_name or "").strip(),
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
            # A new branch has no pull request yet: an earlier run's PR belongs
            # to its own branch and must not be reported as updated by this one.
            _update_status(request_name, user, "Implementing",
                f"Created branch '{branch_name}'. Starting implementation...",
                branch_name=branch_name, pr_url="", pr_number=0)

        prior_changed_paths = _prior_changed_paths(doc) if is_follow_up_mode else []
        implementation_memory = (
            (doc.implementation_snapshot or doc.change_summary or "").strip()
            if is_follow_up_mode else ""
        )
        follow_up_message = (doc.follow_up_message or "").strip() if is_follow_up_mode else ""

        # Carry the planning-phase stage log into execution for continuity.
        prev_stage_log = _parse_stage_log(doc.stage_log or "")

        repo_root = get_repo_root(app_name)
        worktree_before = worktree_signature(repo_root) if is_follow_up_mode else ""
        if restored:
            worktree_before = restored.get("follow_up_worktree_before", worktree_before)

        # The request's session: its investigation, then its implementation and follow-ups.
        graph = koda_session.build_execution_graph()
        initial = {
            "user_message": doc.user_message or "",
            "request_type": doc.request_type or "Improvement",
            "request_name": request_name,
            "plan_object": plan_object,
            "understanding_summary": _extract_understanding(doc),
            "target_app_name": config["target_app_name"],
            "ai_provider": config["ai_provider"],
            "ai_model": config["ai_model"],
            "github_repo_url": config["github_repo_url"],
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
            "follow_up_worktree_before": worktree_before,
            "prior_file_moves": (json.loads(doc.execution_results or "{}").get("file_moves", []) if is_follow_up_mode else []),
            "prior_deleted_paths": [e["path"] for e in as_json_list(doc.files_changed)
                                     if isinstance(e, dict) and e.get("path") and e.get("summary") == "Deleted"] if is_follow_up_mode else [],
        }
        if restored:
            initial = {**initial, **restored,
                       "ai_provider": config["ai_provider"], "ai_model": config["ai_model"]}
        ok, checkpoint_head = run_git(["rev-parse", "HEAD"], cwd=repo_root)
        if not ok:
            raise ValueError("Could not record execution HEAD: " + checkpoint_head)
        active_run = current_run()
        if active_run:
            active_run.journal = ExecutionJournal(request_name,
                lambda p: read_snapshot(agent_tools._resolve_path(app_name, p)),
                branch=branch_name, head=checkpoint_head.strip(), state=initial)

        final_state = graph.invoke(initial, config={"recursion_limit": 100})

        _save_logs(request_name, final_state)

        # Repair that stopped short of a pass still delivers its work, with the open findings shown.
        unresolved = _unresolved_warning(final_state)
        if final_state.get("error") or not (final_state.get("review_passed") or unresolved):
            final_state.setdefault("error", "Execution ended without a passing final review.")
            _update_status(request_name, user, "Failed",
                final_state["error"],
                error_log=final_state.get("error_log") or final_state["error"])
            return

        repo_root = get_repo_root(app_name)

        # Notes the model leaves behind (IMPLEMENTATION_DONE.md, notes.txt) never
        # belong in a patch. Only untracked .md/.txt files go, and never one the
        # approved plan names, so a reviewed patches.txt or README survives.
        executed_plan = final_state.get("plan_object") or plan_object
        planned = {path for task in executed_plan["tasks"] for path in task["files"]}
        stripped = _strip_stray_notes(repo_root, keep=planned)
        if stripped:
            _update_status(request_name, user, "Implementing",
                f"Removed {len(stripped)} stray note file(s): {', '.join(stripped[:5])}")

        worktree_after = worktree_signature(repo_root) if is_follow_up_mode else ""
        run_made_changes = (not is_follow_up_mode) or (worktree_before != worktree_after)

        # Porcelain status sees staged, unstaged and untracked work alike.
        ok_status, status_out = run_git(["status", "--porcelain"], cwd=repo_root)
        if not ok_status:
            raise RuntimeError("Could not verify the working tree after review: " + status_out)
        has_changes = bool((status_out or "").strip())

        change_summary = "\n\n".join(filter(None, [unresolved, (final_state.get("change_summary") or "").strip()]))
        if is_follow_up_mode and change_summary:
            prior_summary = (doc.change_summary or "").strip()
            if prior_summary:
                change_summary = f"{prior_summary}\n\n--- Follow-up #{doc.follow_up_count or 1} ---\n{change_summary}"

        # Execution never ends a request: every run, even one with no net changes, goes
        # through bench and push approval, and only the user-approved deploy completes it.
        if not has_changes:
            outcome = " with no net changes on the branch"
        elif not run_made_changes:
            outcome = " (no further changes; earlier work kept)"
        else:
            outcome = ""

        patch_diff = _generate_patch_diff(app_name)
        edits = [e for e in (final_state.get("edits_made") or [])
                 if not (e.get("path") and _same_file(e["path"], stripped))]
        if is_follow_up_mode:
            edits = _merge_file_edits(as_json_list(doc.files_changed), edits)
        bench_cmds = _compute_bench_commands(app_name, edits)

        _save_implementation_snapshot(request_name, doc, final_state, is_follow_up_mode)

        _update_status(
            request_name, user, "Awaiting Bench Approval",
            f"{'Follow-up patch' if is_follow_up_mode else 'Implementation'} complete"
            f"{outcome}"
            f"{' with unresolved review findings (see the change summary)' if unresolved else ''}. "
            f"{len(bench_cmds)} bench commands need approval.",
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


def _unresolved_warning(final_state: dict) -> str:
    """What the delivered work leaves open: a repair that stopped short of a pass, or tests still failing."""
    parts = []
    stopped = str(final_state.get("review_stopped") or "").strip()
    if stopped and not final_state.get("review_passed"):
        text = "UNRESOLVED - repair stopped before the review passed: " + stopped[:2000]
        notes = str(final_state.get("review_notes") or "").strip()
        if notes and notes[:200] not in stopped:
            text += "\nLast review findings:\n" + notes[:3000]
        parts.append(text)
    tests = str(final_state.get("tests_unresolved") or "").strip()
    if tests:
        parts.append("UNRESOLVED - tests still fail after repair; they were reported, not fixed:\n" + tests[:3000])
    return "\n\n".join(parts)


# Helpers: bench command computation

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
    set_request_value(request_name, {
        "implementation_snapshot": snapshot[:50000],
    })


def run_context_suggestion(request_name: str, query: str, token: str, user: str) -> None:
    """Rank code spans for a task with the shared retrieval/reranking pipeline.

    Runs as a job because indexing a large app takes longer than a web request
    should; the form matches the reply to its dialog by ``token``.
    """
    frappe.set_user("Administrator")
    payload = {"request_name": request_name, "token": token}
    try:
        doc = frappe.get_doc(DOCTYPE_NAME, request_name)
        payload["suggestions"] = koda_core.suggest_context(doc.target_app_name, query)
    except Exception as e:
        log_agent_error("Agent Task Suggestion Error", frappe.get_traceback())
        payload["error"] = str(e)
    frappe.publish_realtime("agent_task_suggestions", payload, user=user)


_NOTE_EXTENSIONS = (".md", ".txt")


def _strip_stray_notes(repo_root: str, keep: set[str]) -> list[str]:
    """Delete untracked .md/.txt files the run left behind, except planned ones.

    Returns the repo-relative paths removed. Only untracked files are listed,
    so tracked source and docs are never touched.
    """
    ok, untracked = run_git(["ls-files", "--others", "--exclude-standard"], cwd=repo_root)
    if not ok:
        return []
    removed = []
    for rel in (untracked or "").splitlines():
        rel = rel.strip()
        if (not rel.lower().endswith(_NOTE_EXTENSIONS) or _same_file(rel, keep)
                or '.koda' in rel.replace('\\', '/').split('/')):
            continue
        try:
            os.remove(os.path.join(repo_root, rel))
            removed.append(rel)
        except OSError:
            log_agent_error("Agent Execution: strip stray note",
                f"could not remove {rel}\n{frappe.get_traceback()}")
    return removed


def _same_file(path: str, others) -> bool:
    """Plan and edit paths are app-relative, git paths are repo-relative; match by suffix."""
    return any(path == other or path.endswith("/" + other) or other.endswith("/" + path) for other in others)


#: Module folders whose ``<folder>/<name>/<name>.json`` records `bench migrate`
#: syncs into the database, plus the app-level fixtures and customisations.
MIGRATED_METADATA_FOLDERS = (
    "doctype", "report", "page", "workspace", "workspace_sidebar", "dashboard",
    "dashboard_chart", "dashboard_chart_source", "number_card", "print_format",
    "web_form", "notification", "module_onboarding", "onboarding_step",
    "form_tour", "custom", "fixtures",
)

#: Front-end sources that `bench build` bundles: plain assets, templates and the
#: Vue/TypeScript/preprocessed-style sources of the esbuild bundles.
BUILT_ASSET_SUFFIXES = (
    ".js", ".jsx", ".ts", ".tsx", ".vue", ".css", ".scss", ".sass", ".less", ".html",
)


def _needs_migrate(path: str) -> bool:
    parts = path.replace("\\", "/").lower().split("/")
    if parts[-1] == "patches.txt":
        return True
    return parts[-1].endswith(".json") and any(folder in parts[:-1] for folder in MIGRATED_METADATA_FOLDERS)


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

    cmds = []
    if any(_needs_migrate(p) for p in edited_paths):
        cmds.append(f"bench --site {site_name} migrate")
    if any(p.lower().endswith(BUILT_ASSET_SUFFIXES) for p in edited_paths):
        cmds.append(f"bench build --app {app_name}")
    cmds.append(f"bench --site {site_name} clear-cache")
    cmds.append("supervisorctl restart all")
    return cmds


# Phase 2b: Bench + Commit — Run bench commands, then branch+commit

def _publish_bench_log(user, request_name, cmd, success, output_preview=""):
    """Broadcast a bench command start/result event via Frappe realtime."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frappe.publish_realtime("agent_log", {
        "request_name": request_name,
        "type": "bench_command",
        "command": cmd,
        "timestamp": ts,
        "event_id": uuid.uuid4().hex,
    }, user=user)
    frappe.publish_realtime("agent_log", {
        "request_name": request_name,
        "type": "bench_result",
        "success": success,
        "output_preview": (output_preview or "")[:180],
        "timestamp": ts,
        "event_id": uuid.uuid4().hex,
    }, user=user)


#: Longest one bench command may run, and the service restart after them (which
#: waits for every worker to stop first).
BENCH_COMMAND_TIMEOUT = 900
SERVICE_RESTART_TIMEOUT = 1800
_INVALID_SELECTION = "Invalid commands format: expected a JSON array of command strings."


def parse_bench_command(command) -> list[str]:
    """One edited command line as argv, quoted the way a POSIX shell would read it.

    The text is only split, never handed to a shell, so quoted paths and JSON
    arguments arrive as single arguments and nothing else is interpreted.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("A bench command must be non-empty text.")
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"Cannot parse bench command `{command.strip()}`: {exc}.") from None
    if not argv:
        raise ValueError("A bench command must be non-empty text.")
    return argv


def normalize_bench_selection(value):
    """The submitted command list, validated; None when nothing was submitted.

    An explicit empty list is returned as-is: the user unchecked every command,
    which means skip them all, not "fall back to the defaults".
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(_INVALID_SELECTION) from None
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(_INVALID_SELECTION)
    commands = [entry.strip() for entry in value if entry.strip()]
    for command in commands:
        parse_bench_command(command)
    return commands


def _is_service_restart(cmd: str) -> bool:
    return "supervisorctl" in cmd.lower()


def _bench_root() -> str:
    return os.path.normpath(os.path.join(frappe.get_app_path("frappe"), "..", "..", ".."))


def _append_receipt(request_name: str, block: str) -> None:
    # api.py imports this module, so the shared bench-log writer is imported late.
    from ampower_koda.agent.api import _append_bench_log
    _append_bench_log(request_name, block)


def _run_bench_command(user, request_name, cmd, bench_root, bench_env):
    """Run one command without a shell. Returns (receipt, failure); failure is "" on success."""
    _publish_bench_log(user, request_name, cmd, True, "Running...")
    try:
        result = subprocess.run(
            parse_bench_command(cmd),
            cwd=bench_root,
            capture_output=True,
            text=True,
            timeout=BENCH_COMMAND_TIMEOUT,
            env=bench_env,
        )
    except subprocess.TimeoutExpired:
        _publish_bench_log(user, request_name, cmd, False, f"TIMEOUT after {BENCH_COMMAND_TIMEOUT}s")
        log_agent_error("Agent Executor: bench command timeout", f"request={request_name}\ncmd={cmd}")
        return f"$ {cmd}\nTIMEOUT after {BENCH_COMMAND_TIMEOUT}s\n", f"{cmd} (timeout)"
    except Exception as e:
        _publish_bench_log(user, request_name, cmd, False, str(e))
        log_agent_error(
            "Agent Executor: bench command",
            f"request={request_name}\ncmd={cmd}\n{e}\n{frappe.get_traceback()}",
        )
        return f"$ {cmd}\nERROR: {e}\n", f"{cmd} ({e})"
    out = (result.stdout or "") + (result.stderr or "")
    ok = result.returncode == 0
    _publish_bench_log(user, request_name, cmd, ok, out[:180])
    status_str = "OK" if ok else f"FAILED (exit {result.returncode})"
    return f"$ {cmd}\n{status_str}\n{out.strip()}\n", "" if ok else f"{cmd} (exit {result.returncode})"


# A service restart stops the worker that asked for it, so it cannot be awaited
# in-process. This helper runs it in its own session, keeps its output, and
# reports the result back through `bench execute` once the services are up.
_RESTART_HELPER = """
import json, subprocess, sys
spec = json.loads(sys.argv[1])
try:
    done = subprocess.run(spec["argv"], cwd=spec["cwd"], capture_output=True, text=True,
                          timeout=spec["timeout"])
    code, out = done.returncode, (done.stdout or "") + (done.stderr or "")
except subprocess.TimeoutExpired:
    code, out = -1, "TIMEOUT after %ss" % spec["timeout"]
except Exception as exc:
    code, out = -1, "ERROR: %s" % exc
with open(spec["output_path"], "w", encoding="utf-8") as handle:
    handle.write(out[-20000:])
print("koda restart %r exited %s" % (spec["argv"], code), flush=True)
sys.exit(subprocess.run(["bench", "--site", spec["site"], "execute",
                         "ampower_koda.agent.executor.record_service_restart",
                         "--kwargs", json.dumps(dict(spec["record"], returncode=code))],
                        cwd=spec["cwd"]).returncode)
"""


def _restart_output_path(token: str) -> str:
    if uuid.UUID(hex=token).hex != token:
        raise ValueError("Invalid restart token.")
    return os.path.join(_bench_root(), "logs", f"koda-restart-{token}.out")


def _start_service_restart(request_name, user, cmd, bench_root, bench_env, promote: bool) -> None:
    """Launch the restart detached; its outcome arrives in record_service_restart."""
    try:
        token = uuid.uuid4().hex
        spec = {
            "argv": parse_bench_command(cmd),
            "cwd": bench_root,
            "timeout": SERVICE_RESTART_TIMEOUT,
            "site": frappe.local.site,
            "output_path": _restart_output_path(token),
            "record": {"request_name": request_name, "run_id": current_run().run_id,
                       "command": cmd, "token": token, "promote": int(promote)},
        }
        os.makedirs(os.path.dirname(spec["output_path"]), exist_ok=True)
        with open(os.path.join(bench_root, "logs", "koda-service-restart.log"), "a", encoding="utf-8") as log:
            subprocess.Popen(
                [sys.executable, "-c", _RESTART_HELPER, json.dumps(spec)],
                cwd=bench_root,
                env=bench_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # outlives the worker the restart stops
            )
    except Exception as e:
        log_agent_error(
            "Agent Executor: service restart",
            f"request={request_name}\ncmd={cmd}\n{e}\n{frappe.get_traceback()}",
        )
        _finish_service_restart(request_name, user, cmd, -1, f"ERROR: could not start: {e}", promote)


def _finish_service_restart(request_name, user, cmd, returncode: int, output: str, promote: bool) -> None:
    """Record a restart's receipt; only a success moves the approval flow on to push."""
    ok = returncode == 0
    status_str = "OK" if ok else f"FAILED (exit {returncode})"
    _append_receipt(request_name, f"$ {cmd}\n{status_str}\n{(output or '').strip()}\n")
    _publish_bench_log(user, request_name, cmd, ok, (output or "")[-180:])
    try:
        status = frappe.db.get_value(DOCTYPE_NAME, request_name, "status")
        if ok and promote and status == "Awaiting Bench Approval":
            _update_status(request_name, user, "Awaiting Push Approval",
                "Bench commands and service restart done. Test the changes, then approve push to commit and push.",
                error_log="")
        elif ok:
            _update_status(request_name, user, status, f"Service restart finished: {cmd}")
        else:
            _update_status(request_name, user, status,
                f"Service restart FAILED (exit {returncode}): the previous code may still be running. "
                "See the bench log, fix the restart, and rerun it.",
                error_log=f"{cmd} (exit {returncode})\n{(output or '').strip()[-2000:]}")
    except RunStopped:
        pass  # a newer run owns the request; the receipt above is still kept


def record_service_restart(request_name: str, run_id: str, command: str, token: str,
                           returncode: int, promote: int = 0) -> None:
    """Receive a deferred restart's result. Called by _RESTART_HELPER via `bench execute`."""
    frappe.set_user("Administrator")
    path = _restart_output_path(token)
    try:
        with open(path, encoding="utf-8") as handle:
            output = handle.read()
        os.remove(path)
    except OSError as e:
        output = f"(restart output unavailable: {e})"
    owner = frappe.db.get_value(DOCTYPE_NAME, request_name, "owner") or "Administrator"
    with run_context(request_name, run_id):
        _finish_service_restart(request_name, owner, command, int(returncode), output, bool(int(promote)))


@managed_job
def run_bench_and_commit(request_name: str) -> None:
    """Run the approved bench commands, then pause for push approval so the user can test."""
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        _config_failed(request_name, "Agent Bench Config Error", e)
        return

    user = config["user"]
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    app_name = config["target_app_name"]
    branch_name = (doc.branch_name or "").strip()

    try:
        # The commands build and migrate whatever is checked out: refuse unless
        # that is this request's branch, or another request's work is deployed
        # and then reported as this one's.
        refused = bench_branch_refusal(app_name, branch_name)
        if refused:
            _update_status(request_name, user, "Awaiting Bench Approval", refused, error_log=refused)
            return

        _update_status(request_name, user, "Building", "Running bench commands...")

        # Blank means no selection was ever saved, so the defaults apply. An
        # explicit empty list means every command was unchecked: skip the step.
        saved = (doc.pending_bench_commands or "").strip()
        try:
            cmds = normalize_bench_selection(saved)
        except ValueError as exc:
            _update_status(request_name, user, "Awaiting Bench Approval",
                f"The saved bench commands are invalid: {exc}", error_log=str(exc))
            return
        if cmds is None:
            cmds = _compute_bench_commands(app_name, [])
        if not cmds:
            skipped = "No bench commands selected: every bench command was skipped."
            _update_status(request_name, user, "Awaiting Push Approval",
                f"{skipped} Test the changes on branch '{branch_name}', then approve push to commit and push.",
                bench_log=skipped, error_log="")
            return

        bench_root = _bench_root()
        bench_env = _get_bench_env()

        deferred_cmds = [cmd for cmd in cmds if _is_service_restart(cmd)]
        immediate_cmds = [cmd for cmd in cmds if not _is_service_restart(cmd)]

        bench_output_parts = []
        failed_cmds = []
        for cmd in immediate_cmds:
            check_active(reserve=BENCH_COMMAND_TIMEOUT + 30)
            receipt, failure = _run_bench_command(user, request_name, cmd, bench_root, bench_env)
            check_active()
            bench_output_parts.append(receipt)
            if failure:
                failed_cmds.append(failure)

        for cmd in deferred_cmds:
            bench_output_parts.append(f"$ {cmd}\nNOT RUN: an earlier command failed\n" if failed_cmds else
                                      f"$ {cmd}\nPENDING: runs last; its result is appended when it finishes\n")

        bench_log = "\n".join(bench_output_parts)

        # Failed builds cannot be promoted to push approval. A later successful
        # retry must also clear the old failure, not leave a stale error behind.
        extra = {"bench_log": bench_log[:50000], "error_log": ""}
        if failed_cmds:
            message = (
                f"{len(failed_cmds)} of {len(immediate_cmds)} bench command(s) FAILED on "
                f"branch '{branch_name}': {', '.join(failed_cmds[:3])}"
                f"{'…' if len(failed_cmds) > 3 else ''}. "
                "Repair the failing command and rerun bench verification before push."
            )
            extra["error_log"] = "\n".join(failed_cmds)
        elif deferred_cmds:
            # Push approval waits for the restart's own result: a denied or
            # failed restart leaves the old code running.
            message = (
                f"Bench commands done on branch '{branch_name}'. Waiting for the service restart "
                f"({', '.join(deferred_cmds)}) to report; push approval opens once it succeeds."
            )
        else:
            message = (
                f"Bench commands done on branch '{branch_name}'. "
                "Test the changes, then approve push to commit and push."
            )

        waiting = failed_cmds or deferred_cmds
        _update_status(
            request_name, user, "Awaiting Bench Approval" if waiting else "Awaiting Push Approval", message, **extra
        )

        frappe.db.commit()

        for cmd in (() if failed_cmds else deferred_cmds):
            check_active(reserve=5)
            _start_service_restart(request_name, user, cmd, bench_root, bench_env, promote=True)

    except Exception as e:
        tb = frappe.get_traceback()
        log_agent_error("Agent Bench+Commit Error", tb)
        _update_status(request_name, user, "Failed", str(e), error_log=tb)


@managed_job
def run_selected_bench_commands(request_name: str, commands: list, previous_status: str) -> None:
    """Run commands chosen in the form or IDE outside the approval flow.

    Queued by api.run_selected_bench_commands so a long migrate or build never
    holds a web request open. Each command's receipt is appended to the bench log
    as it finishes; the request shows Building meanwhile and returns to
    ``previous_status`` afterwards. A service restart runs last and reports back
    through record_service_restart.
    """
    frappe.set_user("Administrator")
    user = frappe.db.get_value(DOCTYPE_NAME, request_name, "owner") or "Administrator"
    try:
        # The API checked the checkout when queuing; it may have moved since.
        doc = frappe.get_doc(DOCTYPE_NAME, request_name)
        refused = bench_branch_refusal(
            (doc.target_app_name or "").strip(), doc.branch_name,
            allowed=((doc.base_branch or "main").strip(),),
        )
        if refused:
            _append_receipt(request_name, refused)
            _update_status(request_name, user, previous_status, refused)
            return

        bench_root = _bench_root()
        bench_env = _get_bench_env()
        restarts = [cmd for cmd in commands if _is_service_restart(cmd)]
        failed = []
        for cmd in commands:
            if _is_service_restart(cmd):
                continue
            check_active(reserve=BENCH_COMMAND_TIMEOUT + 30)
            receipt, failure = _run_bench_command(user, request_name, cmd, bench_root, bench_env)
            _append_receipt(request_name, receipt)
            if failure:
                failed.append(failure)

        if failed:
            for cmd in restarts:
                _append_receipt(request_name, f"$ {cmd}\nNOT RUN: an earlier command failed\n")
            message = f"{len(failed)} bench command(s) FAILED: {', '.join(failed[:3])}. See the bench log."
        elif restarts:
            message = "Bench commands done. The service restart runs last; its result is added to the bench log."
        else:
            message = "Bench commands done. See the bench log."
        _update_status(request_name, user, previous_status, message)

        for cmd in (() if failed else restarts):
            check_active(reserve=5)
            _start_service_restart(request_name, user, cmd, bench_root, bench_env, promote=False)
    except Exception as e:
        tb = frappe.get_traceback()
        log_agent_error("Agent Selected Bench Commands Error", tb)
        _update_status(request_name, user, previous_status, f"Bench commands stopped: {e}", error_log=tb)


# Phase 3: Deployment (Push + Pull Request)

@managed_job
def run_deploy_phase(request_name: str, do_push: bool = True, do_pr: bool = True) -> None:
    """Commit the changes, then optionally push the branch and open a pull request."""
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        _config_failed(request_name, "Agent Deploy Config Error", e)
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

        # A follow-up after the PR was opened only pushes: the open PR picks up the new
        # commits, and asking GitHub for a second one fails with "already exists".
        existing_pr = (doc.pr_url or "").strip()
        stale_pr = _stale_pull_request(doc, config, branch_name) if existing_pr else ""
        if stale_pr:
            # Closed, merged or for another branch/base: it is not updated by this
            # push, so a new PR is created when requested.
            _update_status(request_name, user, "Pushing", stale_pr, pr_url="", pr_number=0)
            existing_pr = ""
        if existing_pr:
            do_pr = False
            if not do_push:
                _update_status(request_name, user, "Awaiting Push Approval",
                    f"A PR is already open ({existing_pr}); approve Push branch to update it.")
                return

        _update_status(request_name, user, "Pushing", "Committing changes...")
        user_msg = (doc.user_message or "")[:200]
        commit_msg = f"[AI Agent] {doc.request_type or 'Improvement'}: {request_name}\n\n{user_msg}"
        check_active(reserve=150)
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
                    status = "Completed" if existing_pr else "Awaiting Push Approval"
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
            check_active(reserve=150)
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
            check_active(reserve=150)
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
        elif existing_pr and do_push:
            summary_parts.append(f"existing PR updated ({existing_pr})")
        unshipped = ignored_regression_tests(app_name)
        if unshipped:
            summary_parts.append(f"{len(unshipped)} regression test(s) not committed because git ignores them: "
                                 f"{', '.join(unshipped[:5])}")

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


# Helpers

def _stale_pull_request(doc, config: dict, branch_name: str) -> str:
    """Why the request's recorded PR cannot be reused for ``branch_name``, or "".

    Reused only while GitHub reports it open with this head and base. When
    GitHub cannot be asked, the record is trusted: creating a second PR for the
    same head would fail with "already exists".
    """
    number = doc.pr_number or 0
    if not number:
        found = re.search(r"/pull/(\d+)", doc.pr_url or "")
        number = int(found.group(1)) if found else 0
    ok, pr = get_pull_request(config["github_repo_url"], config["github_token"], number)
    if not ok:
        return ""
    head = (pr.get("head") or {}).get("ref")
    base = (pr.get("base") or {}).get("ref")
    if pr.get("state") != "open":
        reason = "merged" if pr.get("merged") or pr.get("merged_at") else pr.get("state") or "not open"
    elif head != branch_name:
        reason = f"for branch '{head}'"
    elif base != config["base_branch"]:
        reason = f"against '{base}'"
    else:
        return ""
    return f"The recorded PR ({doc.pr_url}) is {reason}; it will not be reused for '{branch_name}'."


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
        set_request_value(request_name, {
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
    """The investigation's findings, saved with the plan when planning finished."""
    return (getattr(doc, "understanding_snapshot", "") or "").strip()
