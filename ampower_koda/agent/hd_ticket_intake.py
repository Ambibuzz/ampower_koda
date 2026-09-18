# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
"""Whitelisted entry points for the HD Ticket Intake page.

KB has no endpoint that returns a known repository's git_url/branch --
confirmed live: get_repository_status only returns indexing_status and
counters, never clone details. create_repository takes git_url as INPUT
only, for registering a new repo. So Koda keeps its own cache
(Agent Repository) of repo_name -> git_url/branch, populated the first
time a person goes through the manual-entry fallback for that repo. Any
later ticket naming the same repo takes the fast path automatically.

Two calls from the page:
  execute_from_ticket   -- always called first
  resolve_unknown_repo  -- called only if execute_from_ticket returns
                           status "unknown_repo"
"""

import re
import time

import frappe
from frappe import _

from ampower_koda.agent import kb_client
from ampower_koda.agent.kb_client import KBNotFoundError
from ampower_koda.agent.api import start_agent

DOCTYPE_NAME = "Agent Request"

TICKET_TYPE_TO_REQUEST_TYPE = {
    "bug": "Bug Fix",
    "newfeature": "Feature Request",
}

READY_INDEXING_STATUSES = {"Indexed", "Document Generated"}

INDEXING_POLL_INTERVAL_SECONDS = 20
INDEXING_MAX_POLL_SECONDS = 60 * 60


def _extract_relevant_kb_sections(comment_for_developer: str) -> str:
    """Trim KB's comment_for_developer down to the problem, affected files,
    and resolution steps (sections 1-3). Drops the testing plan (section 4)
    and any trailing notes -- Koda has its own Verify phase for that.

    Pattern-matches KB's numbered-section prose format; falls back to
    returning the text unchanged if the shape doesn't match.
    """
    if not comment_for_developer:
        return comment_for_developer

    parts = re.split(r'\n(?=\d\)\s)', comment_for_developer.strip())
    if len(parts) < 3:
        return comment_for_developer

    kept = []
    for part in parts:
        match = re.match(r'^(\d)\)', part)
        if match and int(match.group(1)) <= 3:
            kept.append(part.strip())

    return "\n\n".join(kept) if kept else comment_for_developer


def _get_cached_repo(repository_name: str):
    if not frappe.db.exists("Agent Repository", repository_name):
        return None
    doc = frappe.get_doc("Agent Repository", repository_name)
    return doc.github_repo_url, (doc.base_branch or "main")


def _cache_repo(repository_name: str, github_repo_url: str, base_branch: str):
    if frappe.db.exists("Agent Repository", repository_name):
        doc = frappe.get_doc("Agent Repository", repository_name)
        doc.github_repo_url = github_repo_url
        doc.base_branch = base_branch or "main"
        doc.save(ignore_permissions=True)
    else:
        frappe.get_doc({
            "doctype": "Agent Repository",
            "repository_name": repository_name,
            "github_repo_url": github_repo_url,
            "base_branch": base_branch or "main",
        }).insert(ignore_permissions=True)
    frappe.db.commit()


def _get_git_credential():
    user = frappe.session.user
    if not frappe.db.exists("Agent Git Credential", user):
        frappe.throw(
            _("You haven't connected a GitHub identity yet. Add one in "
              "Agent Git Credential before running an agent.")
        )

    cred = frappe.get_doc("Agent Git Credential", user)
    if not cred.is_active:
        frappe.throw(
            _("Your GitHub identity (Agent Git Credential) is marked inactive. "
              "Reactivate it before running an agent.")
        )

    token = cred.get_password("github_access_token")
    if not (cred.github_username and cred.github_email and token):
        frappe.throw(_("Your Agent Git Credential is incomplete. Please fill "
                        "in all fields before running an agent."))

    return cred.github_username, cred.github_email, token


def _finalize_agent_request(subject: str, description: str, request_type: str,
                             comment_for_developer: str, target_app_name: str,
                             github_repo_url: str, base_branch: str) -> dict:
    github_username, github_email, github_token = _get_git_credential()

    user_message = description
    if comment_for_developer:
        relevant_notes = _extract_relevant_kb_sections(comment_for_developer)
        user_message = (
            f"## Original Ticket\n\n{description}\n\n"
            f"## Knowledge Base Technical Notes\n\n{relevant_notes}"
        )

    settings = frappe.get_single("Agent Settings")

    new_request = frappe.get_doc({
        "doctype": DOCTYPE_NAME,
        "request_title": subject,
        "request_type": request_type,
        "user_message": user_message[:50000],
        "target_app_name": target_app_name,
        "github_repo_url": github_repo_url,
        "base_branch": base_branch or "main",
        "github_token": github_token,
        "git_user_name": github_username,
        "git_user_email": github_email,
        "ai_provider": settings.default_ai_provider,
        "ai_model": settings.default_ai_model,
    }).insert(ignore_permissions=True)
    frappe.db.commit()

    start_agent(new_request.name)

    return {
        "status": "ok",
        "request_name": new_request.name,
        "redirect_url": f"/app/agent-request/{new_request.name}",
    }


@frappe.whitelist()
def execute_from_ticket(subject: str, description: str):
    subject = (subject or "").strip()
    description = (description or "").strip()
    if not subject:
        frappe.throw(_("Subject is required."))
    if not description:
        frappe.throw(_("Description is required."))

    kb_response = kb_client.execute_task(user_prompt=f"{subject}\n\n{description}")
    result = kb_response.get("result") or {}
    ticket_type = (result.get("ticket_type") or "").strip()
    repository = (result.get("repository") or "").strip()
    reply_to_ticket = result.get("reply_to_ticket") or ""
    comment_for_developer = result.get("comment_for_developer") or ""

    request_type = TICKET_TYPE_TO_REQUEST_TYPE.get(ticket_type)
    if not request_type:
        return {
            "status": "not_actionable",
            "ticket_type": ticket_type,
            "reply_to_ticket": reply_to_ticket,
        }

    if not repository or repository == "unknown":
        return {
            "status": "unknown_repo",
            "ticket_type": ticket_type,
            "request_type": request_type,
            "reply_to_ticket": reply_to_ticket,
            "comment_for_developer": comment_for_developer,
        }

    cached = _get_cached_repo(repository)
    if not cached:
        return {
            "status": "unknown_repo",
            "ticket_type": ticket_type,
            "request_type": request_type,
            "reply_to_ticket": reply_to_ticket,
            "comment_for_developer": comment_for_developer,
            "suggested_app_name": repository,
        }

    git_url, branch = cached
    outcome = _finalize_agent_request(
        subject, description, request_type, comment_for_developer,
        repository, git_url, branch,
    )
    outcome["ticket_type"] = ticket_type
    outcome["reply_to_ticket"] = reply_to_ticket
    return outcome


@frappe.whitelist()
def resolve_unknown_repo(subject: str, description: str, request_type: str,
                          comment_for_developer: str, app_name: str,
                          repo_url: str, branch: str):
    subject = (subject or "").strip()
    description = (description or "").strip()
    app_name = (app_name or "").strip()
    repo_url = (repo_url or "").strip()
    branch = (branch or "main").strip()

    if request_type not in TICKET_TYPE_TO_REQUEST_TYPE.values():
        frappe.throw(_("Invalid request type."))
    if not (app_name and repo_url):
        frappe.throw(_("App Name and GitHub Repo URL are required."))

    _cache_repo(app_name, repo_url, branch)

    try:
        repo_status = kb_client.get_repository_status(app_name, allow_not_found=True)
    except KBNotFoundError:
        repo_status = None

    already_ready = bool(
        repo_status and repo_status.get("indexing_status") in READY_INDEXING_STATUSES
    )
    if already_ready:
        return _finalize_agent_request(
            subject, description, request_type, comment_for_developer,
            app_name, repo_url, branch,
        )

    if not repo_status:
        try:
            kb_client.create_repository(app_name, repo_url, branch)
        except Exception as e:
            if "already exists" not in str(e):
                raise
        kb_client.trigger_clone(app_name)

    frappe.enqueue(
        "ampower_koda.agent.hd_ticket_intake.wait_for_indexing_and_start",
        queue="default",
        timeout=3700,
        subject=subject,
        description=description,
        request_type=request_type,
        comment_for_developer=comment_for_developer,
        app_name=app_name,
        repo_url=repo_url,
        branch=branch,
        user=frappe.session.user,
    )

    return {
        "status": "indexing",
        "message": _("'{0}' is being indexed in Knowledge Base. You'll be "
                      "notified when it's ready.").format(app_name),
    }


def wait_for_indexing_and_start(subject: str, description: str, request_type: str,
                                  comment_for_developer: str, app_name: str,
                                  repo_url: str, branch: str, user: str):
    frappe.set_user(user)

    elapsed = 0
    final_status = None
    while elapsed < INDEXING_MAX_POLL_SECONDS:
        try:
            repo_status = kb_client.get_repository_status(app_name, allow_not_found=True)
        except KBNotFoundError:
            repo_status = {"indexing_status": "Pending"}

        status = repo_status.get("indexing_status")
        if status in READY_INDEXING_STATUSES:
            final_status = "indexed"
            break
        if status == "Failed":
            final_status = "failed"
            break

        time.sleep(INDEXING_POLL_INTERVAL_SECONDS)
        elapsed += INDEXING_POLL_INTERVAL_SECONDS
    else:
        final_status = "timed_out"

    if final_status == "indexed":
        try:
            kb_response = kb_client.execute_task(user_prompt=f"{subject}\n\n{description}")
            result = kb_response.get("result") or {}
            comment_for_developer = result.get("comment_for_developer") or comment_for_developer

            outcome = _finalize_agent_request(
                subject, description, request_type, comment_for_developer,
                app_name, repo_url, branch,
            )
            _notify_user(
                user, "success",
                _("'{0}' finished indexing and the agent has started.").format(app_name),
                outcome.get("redirect_url"),
            )
        except Exception:
            frappe.log_error(
                title="HD Ticket Intake: post-indexing finalize failed",
                message=frappe.get_traceback(),
            )
            _notify_user(
                user, "error",
                _("'{0}' finished indexing, but starting the agent failed. "
                  "Check the Error Log.").format(app_name),
                None,
            )
    else:
        reason = _("failed to index") if final_status == "failed" else _("did not finish indexing in time")
        _notify_user(
            user, "error",
            _("'{0}' {1}. No agent request was created.").format(app_name, reason),
            None,
        )


def _notify_user(user: str, kind: str, message: str, redirect_url):
    frappe.publish_realtime(
        "hd_ticket_intake_done",
        {"kind": kind, "message": message, "redirect_url": redirect_url},
        user=user,
    )
    try:
        frappe.get_doc({
            "doctype": "Notification Log",
            "for_user": user,
            "type": "Alert",
            "subject": message,
        }).insert(ignore_permissions=True)
        frappe.db.commit()
    except Exception:
        frappe.log_error(
            title="HD Ticket Intake: Notification Log insert failed",
            message=frappe.get_traceback(),
        )
