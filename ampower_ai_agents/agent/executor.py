# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Two-phase executor: planning phase (understand+plan) and execution phase (implement+review+bench+deploy)

import json
import os

import frappe

from ampower_ai_agents.agent.graph import (
    build_planning_graph,
    build_execution_graph,
)
from ampower_ai_agents.agent.git_ops import (
    generate_branch_name,
    get_repo_root,
    get_current_branch,
    run_git,
    create_branch,
    commit_changes,
    push_branch,
    create_pull_request,
)

DOCTYPE_NAME = "AI Agent Request"


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
            pass
        return summary

    return ""


def _get_doc_config(request_name: str) -> dict:
    """Load the AI Agent Request document and return config needed for the graph state."""
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    settings = frappe.get_single("AI Agents Settings")

    if not settings.enable_ai_agent:
        frappe.throw("AI Agent is disabled in settings")

    provider = (doc.ai_provider or settings.default_ai_provider or "OpenAI").strip()

    provider_config = {
        "OpenAI": ("openai_api_key", "OPENAI_API_KEY", "OpenAI API key"),
        "Gemini": ("google_api_key", "GOOGLE_API_KEY", "Google API key"),
        "Claude": ("anthropic_api_key", "ANTHROPIC_API_KEY", "Anthropic API key"),
    }
    cfg = provider_config.get(provider, provider_config["OpenAI"])
    field_name, env_var, label = cfg
    api_key = settings.get_password(field_name) or ""
    if not api_key.strip():
        frappe.throw(f"{label} not set in AI Agents Settings")
    os.environ[env_var] = api_key.strip()

    return {
        "doc": doc,
        "user": doc.owner or frappe.session.user,
        "target_app_name": (doc.target_app_name or "").strip(),
        "ai_provider": provider,
        "ai_model": (doc.ai_model or settings.default_ai_model or "gpt-4o-mini").strip(),
        "github_repo_url": (doc.github_repo_url or "").strip(),
        "github_token": (doc.get_password("github_token") or "").strip(),
        "base_branch": (doc.base_branch or "main").strip(),
        "branch_prefix": (doc.branch_prefix or "ai-agent/").strip(),
        "git_user_name": (doc.git_user_name or "AI Agent").strip(),
        "git_user_email": (doc.git_user_email or "ai-agent@ampower.com").strip(),
    }


def _update_status(request_name: str, user: str, status: str, message: str = "", **kwargs):
    """Update request status, persist fields, and publish realtime event."""
    frappe.db.set_value(DOCTYPE_NAME, request_name, "status", status)
    allowed_fields = [
        "branch_name", "pr_url", "pr_number", "conversation_log",
        "agent_plan", "files_changed", "error_log", "tokens_used",
        "cost_estimate", "stage_log", "bench_log", "patch_diff",
        "pending_bench_commands",
    ]
    if kwargs:
        for k, v in kwargs.items():
            if k in allowed_fields:
                frappe.db.set_value(DOCTYPE_NAME, request_name, k, v)
    frappe.db.commit()
    payload = {"request_name": request_name, "status": status, "message": message, **kwargs}
    frappe.publish_realtime("agent_progress", payload, user=user)


# ---------------------------------------------------------------------------
# Phase 1: Planning — Understand + Plan, then pause for approval
# ---------------------------------------------------------------------------

def run_planning_phase(request_name: str) -> None:
    """Run understand and plan nodes, then set status to Awaiting Approval."""
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "Agent Planning Config Error")
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

        frappe.db.set_value(DOCTYPE_NAME, request_name, "agent_plan", plan[:50000])
        frappe.db.commit()

        _update_status(request_name, user, "Awaiting Approval",
            "Plan generated. Please review and approve.")

    except Exception as e:
        tb = frappe.get_traceback()
        frappe.log_error(tb, "Agent Planning Error")
        _update_status(request_name, user, "Failed", str(e), error_log=tb)


# ---------------------------------------------------------------------------
# Phase 2: Execution — Implement + Review + Bench + Deploy
# ---------------------------------------------------------------------------

def run_execution_phase(request_name: str) -> None:
    """Run implement, review, bench commands, and deploy nodes."""
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "Agent Execution Config Error")
        frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Failed")
        frappe.db.set_value(DOCTYPE_NAME, request_name, "error_log", str(e))
        frappe.db.commit()
        return

    user = config["user"]
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)

    try:
        revert_msg = _revert_previous_changes(
            config["target_app_name"], config["base_branch"],
            request_name=request_name, user=user,
            branch_prefix=config["branch_prefix"],
        )
        if revert_msg:
            _update_status(request_name, user, "Implementing", f"Cleaned up: {revert_msg}")

        app_name = config["target_app_name"]
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

        prev_stage_log = _parse_stage_log(doc.stage_log or "")

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
        }

        final_state = graph.invoke(initial)

        _save_logs(request_name, final_state)

        if final_state.get("error"):
            _update_status(request_name, user, "Failed",
                final_state["error"],
                error_log=final_state.get("error_log") or final_state["error"])
            return

        repo_root = get_repo_root(app_name)
        ok_diff, diff_out = run_git(["diff", "--stat"], cwd=repo_root)
        ok_ut, untracked = run_git(["ls-files", "--others", "--exclude-standard"], cwd=repo_root)
        has_changes = bool((diff_out or "").strip()) or bool((untracked or "").strip())

        if not has_changes:
            _update_status(request_name, user, "Failed",
                "No code changes were produced. The implement phase did not modify any files on disk.",
                error_log="No file changes detected after implementation.")
            return

        patch_diff = _generate_patch_diff(app_name)

        edits = final_state.get("edits_made") or []
        bench_cmds = _compute_bench_commands(app_name, edits)

        _update_status(
            request_name, user, "Awaiting Bench Approval",
            f"Implementation complete. {len(bench_cmds)} bench commands need approval.",
            patch_diff=patch_diff,
            files_changed=json.dumps(edits)[:50000],
            pending_bench_commands=json.dumps(bench_cmds),
        )

    except Exception as e:
        tb = frappe.get_traceback()
        frappe.log_error(tb, "Agent Execution Error")
        _update_status(request_name, user, "Failed", str(e), error_log=tb)


# ---------------------------------------------------------------------------
# Helpers: bench command computation
# ---------------------------------------------------------------------------

def _compute_bench_commands(app_name: str, edits: list) -> list[str]:
    """Determine which bench commands are needed based on edited file paths."""
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
    """Publish a realtime bench command log entry."""
    import datetime
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
    """Run approved bench commands, then set status to Awaiting Push Approval.
    Branch was already created at the start of execution phase.
    Commit happens only when the user approves the push.
    supervisorctl restart is deferred to run AFTER the status update so
    the worker is not killed before the critical DB writes complete."""
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "Agent Bench Config Error")
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

        import subprocess
        from ampower_ai_agents.agent.graph import _get_bench_env

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
            except subprocess.TimeoutExpired:
                bench_output_parts.append(f"$ {cmd}\nTIMEOUT after 900s\n")
                _publish_bench_log(user, request_name, cmd, False, "TIMEOUT after 900s")
            except Exception as e:
                bench_output_parts.append(f"$ {cmd}\nERROR: {e}\n")
                _publish_bench_log(user, request_name, cmd, False, str(e))

        if deferred_cmds:
            for cmd in deferred_cmds:
                bench_output_parts.append(f"$ {cmd}\n(deferred — runs after status update)\n")

        bench_log = "\n".join(bench_output_parts)

        _update_status(
            request_name, user, "Awaiting Push Approval",
            f"Bench commands done on branch '{branch_name}'. Test the changes, then approve push to commit and push.",
            bench_log=bench_log[:50000],
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
            except Exception:
                pass

    except Exception as e:
        tb = frappe.get_traceback()
        frappe.log_error(tb, "Agent Bench+Commit Error")
        _update_status(request_name, user, "Failed", str(e), error_log=tb)


# ---------------------------------------------------------------------------
# Phase 3: Deploy — Push branch and create PR (after user approval)
# ---------------------------------------------------------------------------

def run_deploy_phase(request_name: str, do_push: bool = True, do_pr: bool = True) -> None:
    """Commit changes, push the branch, and/or create a PR. Called after user approves push.
    Commit happens here so that changes are only committed on explicit user approval."""
    frappe.set_user("Administrator")
    try:
        config = _get_doc_config(request_name)
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "Agent Deploy Config Error")
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
            _update_status(request_name, user, "Failed",
                f"Failed to commit changes: {msg}",
                error_log=f"commit_changes failed: {msg}")
            return

        pr_url = ""
        pr_number = ""

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
        frappe.log_error(tb, "Agent Deploy Error")
        _update_status(request_name, user, "Failed", str(e), error_log=tb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
                except Exception:
                    parts.append(f"--- /dev/null\n+++ b/{fpath}\n+[binary or unreadable]")
        return "\n\n".join(parts)[:100000]
    except Exception as e:
        return f"(could not generate diff: {e})"


def _save_logs(request_name: str, final_state: dict):
    """Persist stage_log and conversation_log to the document."""
    stage_logs = final_state.get("stage_log") or []
    if isinstance(stage_logs, list):
        stage_text = "\n".join(
            f"[{l.get('timestamp', '')}] {l.get('stage', '')} - {l.get('status', '')}: {l.get('summary', '')}"
            for l in stage_logs
        )
    else:
        stage_text = str(stage_logs)

    frappe.db.set_value(DOCTYPE_NAME, request_name, {
        "stage_log": stage_text[:50000],
        "conversation_log": json.dumps(
            final_state.get("intermediate_steps", []), indent=2
        )[:50000],
    })
    frappe.db.commit()


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


def _extract_understanding(doc) -> str:
    """Extract understanding summary from conversation log if available."""
    try:
        steps = json.loads(doc.conversation_log or "[]")
        for step in steps:
            if step.get("phase") == "Understanding":
                return step.get("output", "")
    except (json.JSONDecodeError, TypeError):
        pass
    return ""
