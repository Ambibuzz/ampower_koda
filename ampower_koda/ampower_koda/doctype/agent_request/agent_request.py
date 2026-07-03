# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# For license information, please see license.txt

import frappe
from frappe import _
from frappe.model.document import Document

from ampower_koda.agent.errors import log_agent_error


class AgentRequest(Document):
    def before_insert(self):
        if not self.owner:
            self.owner = frappe.session.user
        self._set_defaults_from_settings()

    def _set_defaults_from_settings(self):
        """Populate provider/model defaults from Agent Settings if not already set."""
        try:
            settings = frappe.get_single("Agent Settings")
            if not self.ai_provider:
                self.ai_provider = settings.default_ai_provider or "OpenAI"
            if not self.ai_model:
                self.ai_model = settings.default_ai_model or "gpt-4o-mini"
        except Exception:
            log_agent_error(
                "Agent Request: defaults from settings",
                frappe.get_traceback(),
            )

    def validate(self):
        if not (self.target_app_name or "").strip():
            frappe.throw(_("Target App Name is required"))
        if not (self.github_repo_url or "").strip():
            frappe.throw(_("GitHub Repo URL is required"))

        token = self.github_token
        if not self.is_new():
            token = self.get_password("github_token", raise_exception=False)
        if not token:
            frappe.throw(_("GitHub Token is required"))

        if not self.use_default_prompts:
            seen = set()
            for row in self.prompts:
                if row.prompt_key in seen:
                    frappe.throw(_("Duplicate prompt type not allowed: {0}").format(row.prompt_key))
                seen.add(row.prompt_key)
