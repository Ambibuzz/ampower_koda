# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
"""Client for fetching an HD Ticket's subject/description, branching by
Agent Settings.hd_connection_mode:

  Same Site  -- Helpdesk and Koda share one site; read the doc directly,
                no HTTP involved.
  Same Bench -- Helpdesk is a different site on the same bench; reached
                over HTTP with an API key/secret, same as Remote.
  Remote     -- Helpdesk is on a different server entirely; identical
                code path to Same Bench, just a different hd_base_url.

HD Ticket.description is a Text Editor field (raw HTML). This module always
returns plain text -- HTML is stripped here, once, so every caller (KB's
user_prompt, Agent Request.user_message) gets consistent plain text.
"""

import json
import re

import frappe
import requests
from frappe import _
from html import unescape


class HDNotFoundError(Exception):
    """Raised when the given HD Ticket doesn't exist."""


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def _get_hd_config():
    settings = frappe.get_single("Agent Settings")
    mode = settings.hd_connection_mode or "Same Site"
    base_url = (settings.hd_base_url or "").rstrip("/")
    api_key = (settings.hd_api_key or "").strip()
    api_secret = settings.get_password("hd_api_secret") if settings.hd_api_secret else ""
    return mode, base_url, api_key, api_secret


def fetch_hd_ticket(ticket_id: str) -> dict:
    """Returns {"subject": str, "description": str} for the given HD Ticket
    name, with description already stripped to plain text.

    Raises HDNotFoundError if the ticket doesn't exist.
    """
    mode, base_url, api_key, api_secret = _get_hd_config()

    if mode == "Same Site":
        if not frappe.db.exists("HD Ticket", ticket_id):
            raise HDNotFoundError(ticket_id)
        doc = frappe.get_doc("HD Ticket", ticket_id)
        subject, description = doc.subject, doc.description
    else:
        if not (base_url and api_key and api_secret):
            frappe.throw(
                _("Helpdesk connection is not configured. Set HD Base URL, "
                  "API Key and API Secret in Agent Settings.")
            )

        url = f"{base_url}/api/resource/HD Ticket/{ticket_id}"
        headers = {"Authorization": f"token {api_key}:{api_secret}"}

        try:
            response = requests.get(url, headers=headers, timeout=30)
        except requests.RequestException as e:
            frappe.throw(_("Could not reach Helpdesk: {0}").format(e))

        if response.status_code == 404:
            raise HDNotFoundError(ticket_id)
        if response.status_code != 200:
            frappe.throw(
                _("Helpdesk returned an error ({0}): {1}").format(
                    response.status_code, response.text[:500]
                )
            )

        try:
            data = response.json().get("data") or {}
        except ValueError:
            frappe.throw(_("Helpdesk returned a non-JSON response."))

        subject, description = data.get("subject"), data.get("description")

    return {
        "subject": subject or "",
        "description": _strip_html(description or ""),
    }

def get_hd_ticket_url(ticket_id: str) -> str:
    """Build the Helpdesk portal URL for a ticket, based on how HD is connected."""
    mode, base_url, _api_key, _api_secret = _get_hd_config()
    origin = base_url if mode != "Same Site" else frappe.utils.get_url()
    if not origin:
        frappe.throw(
            _("Helpdesk connection is not configured. Set HD Base URL in Agent Settings.")
        )
    return f"{origin.rstrip('/')}/helpdesk/tickets/{ticket_id}"

def build_send_to_koda_script() -> str:
    """Build the client JS for the 'Send to Koda' HD Ticket action.

    Same Site: the ticket and Koda share an origin, so the script uses the
    browser's own window.location.origin -- no configuration needed and it
    survives a domain change with no regeneration.
    Same Bench / Remote: Koda is a different site, so the base URL can't be
    read from the browser and is baked in from Agent Settings.koda_public_url
    at generation time instead.
    """
    settings = frappe.get_single("Agent Settings")
    mode = settings.hd_connection_mode or "Same Site"

    if mode == "Same Site":
        base_url_js = "window.location.origin"
    else:
        koda_url = (settings.koda_public_url or "").strip().rstrip("/")
        if not koda_url:
            frappe.throw(
                _("Set Koda Public URL in Agent Settings before generating the "
                  "'Send to Koda' button (HD Connection Mode is {0}).").format(mode)
            )
        base_url_js = json.dumps(koda_url)

    return (
        "function setupForm({ doc }) {\n"
        "    let actions = [];\n"
        '    if (doc.status === "Open") {\n'
        "        actions.push({\n"
        '            label: "Send to Koda",\n'
        "            onClick: () => {\n"
        f'                window.open({base_url_js} + "/app/hd-ticket-intake/" + doc.name, "_blank");\n'
        "            },\n"
        "        });\n"
        "    }\n"
        "    return { actions };\n"
        "}"
    )


def push_send_to_koda_script(script_text: str) -> None:
    """Write the generated 'Send to Koda' script to wherever HD Form Script lives."""
    mode, base_url, api_key, api_secret = _get_hd_config()

    if mode == "Same Site":
        if not frappe.db.exists("HD Form Script", "HD Ticket-Send to Koda"):
            frappe.throw(_("HD Form Script 'HD Ticket-Send to Koda' was not found."))
        frappe.db.set_value("HD Form Script", "HD Ticket-Send to Koda", "script", script_text)
        frappe.db.commit()
    else:
        if not (base_url and api_key and api_secret):
            frappe.throw(
                _("Helpdesk connection is not configured. Set HD Base URL, "
                  "API Key and API Secret in Agent Settings.")
            )

        url = f"{base_url}/api/resource/HD Form Script/HD Ticket-Send to Koda"
        headers = {"Authorization": f"token {api_key}:{api_secret}"}
        payload = {"script": script_text}

        try:
            response = requests.put(url, json=payload, headers=headers, timeout=30)
        except requests.RequestException as e:
            frappe.throw(_("Could not reach Helpdesk: {0}").format(e))

        if response.status_code != 200:
            frappe.throw(
                _("Helpdesk returned an error updating the Send to Koda script "
                  "({0}): {1}").format(response.status_code, response.text[:500])
            )


def set_pending_draft_reply(ticket_id: str, content: str) -> None:
    """Stages a suggested reply as HD Ticket.custom_koda_pending_draft, for
    the agent portal to surface as an 'Insert as draft' banner -- never
    posted or emailed automatically.
    """
    mode, base_url, api_key, api_secret = _get_hd_config()

    if mode == "Same Site":
        if not frappe.db.exists("HD Ticket", ticket_id):
            raise HDNotFoundError(ticket_id)
        frappe.db.set_value("HD Ticket", ticket_id, "custom_koda_pending_draft", content)
        frappe.db.commit()
    else:
        if not (base_url and api_key and api_secret):
            frappe.throw(
                _("Helpdesk connection is not configured. Set HD Base URL, "
                  "API Key and API Secret in Agent Settings.")
            )

        url = f"{base_url}/api/resource/HD Ticket/{ticket_id}"
        headers = {"Authorization": f"token {api_key}:{api_secret}"}
        payload = {"custom_koda_pending_draft": content}

        try:
            response = requests.put(url, json=payload, headers=headers, timeout=30)
        except requests.RequestException as e:
            frappe.throw(_("Could not reach Helpdesk: {0}").format(e))

        if response.status_code == 404:
            raise HDNotFoundError(ticket_id)
        if response.status_code != 200:
            frappe.throw(
                _("Helpdesk returned an error setting pending draft ({0}): {1}").format(
                    response.status_code, response.text[:500]
                )
            )

def post_ticket_comment(ticket_id: str, content: str) -> None:
    """Posts an internal HD Ticket Comment (never emailed to the customer).
    Used for staging a suggested reply an agent can review and choose to
    send, rather than auto-replying.
    """
    mode, base_url, api_key, api_secret = _get_hd_config()

    if mode == "Same Site":
        if not frappe.db.exists("HD Ticket", ticket_id):
            raise HDNotFoundError(ticket_id)
        frappe.get_doc({
            "doctype": "HD Ticket Comment",
            "reference_ticket": ticket_id,
            "content": content,
        }).insert(ignore_permissions=True)
        frappe.db.commit()
    else:
        if not (base_url and api_key and api_secret):
            frappe.throw(
                _("Helpdesk connection is not configured. Set HD Base URL, "
                  "API Key and API Secret in Agent Settings.")
            )

        url = f"{base_url}/api/resource/HD Ticket Comment"
        headers = {"Authorization": f"token {api_key}:{api_secret}"}
        payload = {"reference_ticket": ticket_id, "content": content}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
        except requests.RequestException as e:
            frappe.throw(_("Could not reach Helpdesk: {0}").format(e))

        if response.status_code == 404:
            raise HDNotFoundError(ticket_id)
        if response.status_code not in (200, 201):
            frappe.throw(
                _("Helpdesk returned an error posting comment ({0}): {1}").format(
                    response.status_code, response.text[:500]
                )
            )
