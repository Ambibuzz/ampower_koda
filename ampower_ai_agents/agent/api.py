# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Whitelisted API for the AI Agent page

import frappe

DOCTYPE_NAME = "AI Agent Request"


@frappe.whitelist()
def create_agent_request(message: str, request_type: str = "Improvement", title: str = ""):
    """Create an AI Agent Request and enqueue the agent. Returns request name."""
    if not message or not message.strip():
        frappe.throw("Message is required")
    request_type = (request_type or "Improvement").strip()
    if request_type not in ("Bug Fix", "Feature Request", "Improvement"):
        request_type = "Improvement"
    title = (title or "").strip() or message[:80]

    settings = frappe.get_single("AI Agents Settings")
    if not settings.enable_ai_agent:
        frappe.throw("AI Coding Agent is disabled in settings")
    if not settings.openai_api_key:
        frappe.throw("OpenAI API key is not set in settings")

    doc = frappe.get_doc(
        {
            "doctype": DOCTYPE_NAME,
            "request_title": title,
            "request_type": request_type,
            "user_message": message,
            "status": "Queued",
        }
    )
    doc.insert(ignore_permissions=True)
    frappe.db.commit()

    frappe.enqueue(
        "ampower_ai_agents.agent.executor.run_agent",
        queue="default",
        timeout=1800,
        request_name=doc.name,
    )
    return doc.name


@frappe.whitelist()
def get_agent_status(request_name: str):
    """Return current status and key fields for a request."""
    if not request_name:
        return None
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if not doc:
        return None
    return {
        "name": doc.name,
        "status": doc.status,
        "branch_name": doc.branch_name,
        "pr_url": doc.pr_url,
        "pr_number": doc.pr_number,
        "conversation_log": doc.conversation_log,
        "agent_plan": doc.agent_plan,
        "error_log": doc.error_log,
    }


@frappe.whitelist()
def get_agent_history(limit: int = 20):
    """Return recent agent requests for the current user."""
    limit = min(int(limit or 20), 50)
    return frappe.get_all(
        DOCTYPE_NAME,
        filters={"owner": frappe.session.user},
        fields=["name", "request_title", "request_type", "status", "creation", "pr_url", "branch_name"],
        order_by="creation desc",
        limit_page_length=limit,
    )


@frappe.whitelist()
def cancel_agent_request(request_name: str):
    """Cancel a queued or running request (best-effort)."""
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if doc.status in ("Completed", "Failed", "Cancelled"):
        frappe.msgprint("Request already finished.")
        return
    frappe.db.set_value(DOCTYPE_NAME, request_name, "status", "Cancelled")
    frappe.db.commit()
    return True
