# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# For license information, please see license.txt

import frappe
from frappe.model.document import Document


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
            pass

    def validate(self):
        if not self.target_app_name or not self.target_app_name.strip():
            frappe.throw("Target App Name is required")
        if not self.github_repo_url or not self.github_repo_url.strip():
            frappe.throw("GitHub Repo URL is required")

        token = self.github_token
        if not self.is_new():
            token = self.get_password("github_token", raise_exception=False)
        if not token:
            frappe.throw("GitHub Token is required")

        if not self.use_default_prompts:
            seen = set()
            for row in self.prompts:
                if row.prompt_key in seen:
                    frappe.throw(f"Duplicate prompt type not allowed: {row.prompt_key}")
                seen.add(row.prompt_key)
