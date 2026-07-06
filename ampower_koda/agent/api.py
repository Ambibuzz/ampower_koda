# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Whitelisted API for the AI Agent

import json
import os
import subprocess

import frappe
from frappe import _

from ampower_koda.agent.git_ops import checkout_base
from ampower_koda.agent.graph import _get_bench_env

DOCTYPE_NAME = "AI Agent Request"


def _validate_provider_key(doc):
    """
    Verifies that the AI provider and its API key are correctly configured in settings.
    This check ensures the agent has its 'brain' ready before it attempts any work.
    """
    settings = frappe.get_single("AI Agent Settings")
    if not settings.enable_ai_agent:
        frappe.throw(_("AI Coding Agent is disabled in AI Agent Settings."))
    provider = (doc.ai_provider or settings.default_ai_provider or "OpenAI").strip()
    key_checks = {
        "OpenAI": ("openai_api_key", "OpenAI API key"),
        "Gemini": ("google_api_key", "Google API key"),
        "Claude": ("anthropic_api_key", "Anthropic API key"),
    }
    field, label = key_checks.get(provider, key_checks["OpenAI"])
    if not getattr(settings, field, None):
        frappe.throw(_("{0} is not set in AI Agent Settings.").format(label))


@frappe.whitelist()
def start_agent(request_name: str):
    """
    Initiates the full workflow from scratch. 
    This guides the agent through exploring the codebase and drafting a detailed plan 
    for your review before any implementation begins.
    """
    if not request_name:
        frappe.throw(_("Request name is required."))

    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    
    # We only allow starting the agent if it is currently idle or in a terminal state.
    # This prevents accidental overlapping runs on the same request.
    restartable = ("Queued", "Failed", "Cancelled", "Completed", "Awaiting Approval", "Awaiting Push Approval")
    if doc.status not in restartable:
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
            if isinstance(cmd_list, list) and cmd_list:
                frappe.db.set_value(DOCTYPE_NAME, request_name,
                    "pending_bench_commands", json.dumps(cmd_list))
        except (ValueError, TypeError):
            pass

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
    except Exception:
        pass

    return {
        "status": "ok",
        "message": _("Running {0} bench commands...").format(len(cmds)),
    }


@frappe.whitelist()
def approve_push(request_name: str, push_branch: int = 1, create_pr: int = 1):
    """
    Approves pushing the feature branch to remote and/or opening a pull request.
    Set push_branch=1 to push, create_pr=1 to create a PR. At least one must be selected.
    """
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
def run_selected_bench_commands(request_name: str, commands: str = None):
    """Run user-selected bench commands. commands is a JSON array of command strings."""
    if not request_name:
        frappe.throw(_("Request name is required."))

    
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
        except Exception as e:
            output_parts.append(f"$ {cmd}\nERROR: {e}\n")

    bench_log = "\n".join(output_parts)
    frappe.db.set_value(DOCTYPE_NAME, request_name, "bench_log", bench_log[:50000])
    frappe.db.commit()

    return {"status": "ok", "log": bench_log}


@frappe.whitelist()
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


@frappe.whitelist()
def cancel_agent_request(request_name: str):
    """Cancel a queued or running request (best-effort)."""
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status in ("Completed", "Failed", "Cancelled"):
        frappe.msgprint(_("Request already finished."))
        return
    frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Cancelled")
    frappe.db.commit()
    return True
