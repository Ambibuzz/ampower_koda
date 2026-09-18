# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
"""Thin HTTP client for calling ampower_knowledge_base's whitelisted API.

KB is not installed in this bench -- every call here is a real outbound HTTP
request, authenticated the same way any external Frappe site is: an API
key/secret pair sent as a token, not an internal frappe.client call.
"""

import requests
import frappe
from frappe import _


class KBNotFoundError(Exception):
    """Raised when KB reports a repository/record doesn't exist.

    Confirmed live: a bad repository_name surfaces as HTTP 417 with
    "not found" (lowercased-matched) in the response body -- KB uses
    ValidationError + frappe.throw for every failure case, not distinct
    exception types or HTTP status codes.
    """


def _get_kb_config():
    settings = frappe.get_single("Agent Settings")
    base_url = (settings.kb_base_url or "").rstrip("/")
    api_key = (settings.kb_api_key or "").strip()
    api_secret = settings.get_password("kb_api_secret") if settings.kb_api_secret else ""

    if not base_url or not api_key or not api_secret:
        frappe.throw(
            _("Knowledge Base connection is not configured. Set KB Base URL, "
              "API Key and API Secret in Agent Settings.")
        )
    return base_url, api_key, api_secret


def _kb_request(method_path: str, params: dict, timeout: int = 60,
                 allow_not_found: bool = False) -> dict:
    """Call one whitelisted KB method and return its parsed JSON `message`.

    allow_not_found=True turns a not-found response into KBNotFoundError
    instead of a generic frappe.throw, so callers can branch on "doesn't
    exist yet" vs. "KB is actually broken."
    """
    base_url, api_key, api_secret = _get_kb_config()
    url = f"{base_url}/api/method/{method_path}"
    headers = {"Authorization": f"token {api_key}:{api_secret}"}

    try:
        response = requests.post(url, data=params, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        frappe.throw(_("Could not reach Knowledge Base: {0}").format(e))

    if allow_not_found and response.status_code == 417 and "not found" in response.text.lower():
        raise KBNotFoundError(f"{method_path}: {params}")

    if response.status_code != 200:
        frappe.throw(
            _("Knowledge Base returned an error ({0}): {1}").format(
                response.status_code, response.text[:500]
            )
        )

    try:
        payload = response.json()
    except ValueError:
        frappe.throw(_("Knowledge Base returned a non-JSON response."))

    return payload.get("message", payload)


def execute_task(user_prompt: str, task_type: str = "triage",
                  output_format: str = None, include_details: int = 1) -> dict:
    params = {
        "task_type": task_type,
        "user_prompt": user_prompt,
        "include_details": include_details,
    }
    if output_format:
        params["output_format"] = output_format
    return _kb_request("ampower_knowledge_base.api.agent.execute_task", params)


def get_repository_status(repository_name: str, allow_not_found: bool = False) -> dict:
    return _kb_request(
        "ampower_knowledge_base.api.repository.get_repository_status",
        {"repository_name": repository_name},
        allow_not_found=allow_not_found,
    )


def create_repository(repository_name: str, git_url: str, branch: str) -> dict:
    return _kb_request(
        "ampower_knowledge_base.api.repository.create_repository",
        {"repository_name": repository_name, "git_url": git_url, "branch": branch},
    )


def trigger_clone(repository_name: str) -> dict:
    return _kb_request(
        "ampower_knowledge_base.api.repository.trigger_clone",
        {"repository_name": repository_name},
    )
