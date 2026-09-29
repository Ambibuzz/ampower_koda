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
from ampower_koda.agent import hd_client
from ampower_koda.agent.kb_client import KBNotFoundError
from ampower_koda.agent.hd_client import HDNotFoundError


DOCTYPE_NAME = "Agent Request"

TICKET_TYPE_TO_REQUEST_TYPE = {
    "bug": "Bug Fix",
    "newfeature": "ERPNext-flavored",
}

READY_INDEXING_STATUSES = {"Indexed", "Document Generated"}

INDEXING_POLL_INTERVAL_SECONDS = 20
INDEXING_MAX_POLL_SECONDS = 60 * 60


def _extract_relevant_kb_sections_html(comment_for_developer: str) -> str:
    """Condense KB's comment_for_developer into HTML: just the problem (one
    line), the affected files (a list), and the steps (titles only, not
    KB's per-step reasoning paragraphs). Drops the testing plan, root-cause
    narrative, and trailing notes -- Koda's own Explore phase reads the
    actual files itself and doesn't need KB's reasoning repeated.

    Returns HTML, not markdown -- Agent Request.user_message is a "Text
    Editor" fieldtype, which renders raw HTML, not markdown syntax or plain
    newlines. Every extracted fragment is escaped before insertion.

    Pattern-matches KB's prose format (backtick file paths, "- Step X:"
    lines, "1) Root Cause Analysis" section headers). Falls back to the
    unmodified text (escaped, newlines as <br>) if the shape doesn't match.
    """
    if not comment_for_developer:
        return comment_for_developer

    text = comment_for_developer.strip()

    def esc(s: str) -> str:
        return frappe.utils.escape_html(s.strip())

    problem = ""
    section1 = re.search(r'1\)[^\n]*\n(.*?)(?=\n\d\)|\Z)', text, re.DOTALL)
    if section1:
        for line in section1.group(1).splitlines():
            line = line.strip("- ").strip()
            if line and not line.lower().startswith(("classification", "repository:")):
                problem = line
                break

    files = list(dict.fromkeys(
        re.findall(r'`([\w./-]+\.(?:py|js|json|html|md))`', text)
    ))

    steps = re.findall(r'-\s*Step\s+\w+:\s*([^\n]+)', text)

    if not files and not steps:
        return f"<p>{esc(text).replace(chr(10), '<br>')}</p>"

    html = []
    if problem:
        html.append(f"<p><strong>Problem:</strong> {esc(problem)}</p>")
    if files:
        html.append("<p><strong>Files:</strong></p><ul>")
        html.extend(f"<li><code>{esc(f)}</code></li>" for f in files)
        html.append("</ul>")
    if steps:
        html.append("<p><strong>Steps:</strong></p><ol>")
        html.extend(f"<li>{esc(s)}</li>" for s in steps)
        html.append("</ol>")

    return "".join(html)


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


def _finalize_agent_request(subject: str, description: str, request_type: str,
                             comment_for_developer: str, target_app_name: str,
                             github_repo_url: str, base_branch: str,
                             source_hd_ticket: str = None) -> dict:

    description_html = frappe.utils.escape_html(description).replace("\n", "<br>")
    user_message = f"<h4>Original Ticket</h4><p>{description_html}</p>"
    if comment_for_developer:
        relevant_notes_html = _extract_relevant_kb_sections_html(comment_for_developer)
        user_message += f"<h4>Knowledge Base Technical Notes</h4>{relevant_notes_html}"

    settings = frappe.get_single("Agent Settings")

    new_request = frappe.get_doc({
        "doctype": DOCTYPE_NAME,
        "request_title": subject,
        "request_type": request_type,
        "user_message": user_message[:50000],
        "target_app_name": target_app_name,
        "github_repo_url": github_repo_url,
        "base_branch": base_branch or "main",
        "ai_provider": settings.default_ai_provider,
        "ai_model": settings.default_ai_model,
        "source_hd_ticket": source_hd_ticket or "",
    }).insert(ignore_permissions=True)
    frappe.db.commit()


    return {
        "status": "ok",
        "request_name": new_request.name,
        "redirect_url": f"/app/agent-request/{new_request.name}",
    }




@frappe.whitelist()
def execute_from_ticket(subject: str, description: str, source_hd_ticket: str = None):
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
        if source_hd_ticket and reply_to_ticket:
            try:
                hd_client.set_pending_draft_reply(source_hd_ticket, reply_to_ticket)
            except Exception:
                pass
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
            "source_hd_ticket": source_hd_ticket,
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
            "source_hd_ticket": source_hd_ticket,
        }

    git_url, branch = cached
    outcome = _finalize_agent_request(
        subject, description, request_type, comment_for_developer,
        repository, git_url, branch, source_hd_ticket,
    )
    outcome["ticket_type"] = ticket_type
    outcome["reply_to_ticket"] = reply_to_ticket
    return outcome

@frappe.whitelist()
def regenerate_send_to_koda_script():
    """(Re)build and push the 'Send to Koda' HD Form Script for the current
    HD Connection Mode. Run this after changing Connection Mode or Koda
    Public URL in Agent Settings.
    """
    script_text = hd_client.build_send_to_koda_script()
    hd_client.push_send_to_koda_script(script_text)
    return {"status": "ok"}


@frappe.whitelist()
def fetch_ticket_for_intake(hd_ticket: str):
    """Called by the intake page on load when it's opened with ?hd_ticket=<id>.
    Returns the ticket's subject/description so the page can auto-fill,
    instead of the person copy-pasting them by hand.
    """
    hd_ticket = (hd_ticket or "").strip()
    if not hd_ticket:
        frappe.throw(_("hd_ticket is required."))

    try:
        return hd_client.fetch_hd_ticket(hd_ticket)
    except HDNotFoundError:
        frappe.throw(_("Could not find HD Ticket '{0}'.").format(hd_ticket))


@frappe.whitelist()
def resolve_unknown_repo(subject: str, description: str, request_type: str,
                          comment_for_developer: str, app_name: str,
                          repo_url: str, branch: str, source_hd_ticket: str = None):
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
            app_name, repo_url, branch, source_hd_ticket,
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
        source_hd_ticket=source_hd_ticket,
    )

    return {
        "status": "indexing",
        "message": _("'{0}' is being indexed in Knowledge Base. You'll be "
                      "notified when it's ready.").format(app_name),
    }


def wait_for_indexing_and_start(subject: str, description: str, request_type: str,
                                  comment_for_developer: str, app_name: str,
                                  repo_url: str, branch: str, user: str,
                                  source_hd_ticket: str = None):
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
                app_name, repo_url, branch, source_hd_ticket,
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
        