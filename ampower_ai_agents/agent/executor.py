# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Entry point: run_agent(request_name) for background job with realtime updates

import json
import os

import frappe

from ampower_ai_agents.agent.graph import build_graph
from ampower_ai_agents.agent.git_ops import cleanup_and_checkout_base

DOCTYPE_NAME = "AI Agent Request"


def run_agent(request_name: str) -> None:
    """Run the AI coding agent for the given AI Agent Request. Called via frappe.enqueue."""
    frappe.set_user("Administrator")
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    user = doc.owner or frappe.session.user

    def update_status(status: str, message: str = "", **kwargs):
        frappe.db.set_value(DOCTYPE_NAME, request_name, "status", status)
        allowed_fields = [
            "branch_name", "pr_url", "pr_number", "conversation_log",
            "agent_plan", "files_changed", "error_log", "tokens_used",
            "cost_estimate", "stage_log",
        ]
        if kwargs:
            for k, v in kwargs.items():
                if k in allowed_fields:
                    frappe.db.set_value(DOCTYPE_NAME, request_name, k, v)
        frappe.db.commit()
        payload = {"request_name": request_name, "status": status, "message": message, **kwargs}
        frappe.publish_realtime("agent_progress", payload, user=user)

    def _safe_checkout_base():
        """Always return to base branch, swallowing errors."""
        try:
            ok, msg = cleanup_and_checkout_base()
            if ok:
                frappe.logger().info(f"Agent cleanup: {msg}")
            else:
                frappe.logger().warning(f"Agent cleanup failed: {msg}")
        except Exception as cleanup_err:
            frappe.logger().warning(f"Agent cleanup error: {cleanup_err}")

    try:
        settings = frappe.get_single("AI Agents Settings")
        if not settings.enable_ai_agent:
            update_status("Failed", "AI Agent is disabled in settings")
            return

        api_key = settings.get_password("openai_api_key") or ""
        if not api_key.strip():
            update_status("Failed", "OpenAI API key not set in settings")
            return

        os.environ["OPENAI_API_KEY"] = api_key.strip()

        update_status("Understanding", "Exploring codebase...")
        graph = build_graph()
        initial = {
            "user_message": doc.user_message or "",
            "request_type": doc.request_type or "Improvement",
            "request_name": request_name,
            "ai_model": (settings.ai_model or "gpt-4o-mini").strip(),
            "intermediate_steps": [],
            "edits_made": [],
            "stage_log": [],
        }

        final_state = None
        try:
            final_state = graph.invoke(initial)
        except Exception as graph_err:
            tb = frappe.get_traceback()
            frappe.log_error(tb, "Agent Graph Error")
            update_status("Failed", str(graph_err), error_log=tb)
            frappe.db.commit()
            return
        finally:
            _safe_checkout_base()

        if final_state.get("error"):
            update_status("Failed", final_state["error"], error_log=final_state.get("error_log") or final_state["error"])
            frappe.db.commit()
            return

        stage_logs = final_state.get("stage_log") or []
        stage_text = "\n".join(
            f"[{l.get('timestamp','')}] {l.get('stage','')} - {l.get('status','')}: {l.get('summary','')}"
            for l in stage_logs
        ) if isinstance(stage_logs, list) else str(stage_logs)

        update_status(
            "Completed",
            "PR created",
            branch_name=final_state.get("branch_name"),
            pr_url=final_state.get("pr_url"),
            pr_number=final_state.get("pr_number"),
            conversation_log=json.dumps(final_state.get("intermediate_steps", []), indent=2)[:50000],
            agent_plan=(final_state.get("plan") or "")[:50000],
            files_changed=(json.dumps(final_state.get("edits_made") or [])[:50000]),
            stage_log=stage_text[:50000],
        )
        frappe.db.commit()
    except Exception as e:
        tb = frappe.get_traceback()
        frappe.log_error(tb, "Agent Run Error")
        update_status("Failed", str(e), error_log=tb)
        frappe.db.commit()
        _safe_checkout_base()
