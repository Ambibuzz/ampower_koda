# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# For license information, please see license.txt

import frappe
from frappe.model.document import Document


class AIAgentRequest(Document):
    def before_insert(self):
        if not self.owner:
            self.owner = frappe.session.user
        self._set_defaults_from_settings()

    def _set_defaults_from_settings(self):
        """Populate provider/model defaults from AI Agents Settings if not already set."""
        try:
            settings = frappe.get_single("AI Agents Settings")
            if not self.ai_provider:
                self.ai_provider = settings.default_ai_provider or "OpenAI"
            if not self.ai_model:
                self.ai_model = settings.default_ai_model or "gpt-4o-mini"
        except Exception:
            pass

    def validate(self):
        if not self.ai_provider:
            frappe.throw(_("AI Provider is required"))
        if not self.tokens_used:
            self.tokens_used = 0  # Default value for tokens used
        if not self.cost_per_request:
            self.cost_per_request = 0.0  # Default value for cost per request
        # Password fields need get_password() for value check on existing docs;
        # on new docs the value is in self.github_token directly
        token = self.github_token
        if not self.is_new():
            token = self.get_password("github_token", raise_exception=False)
        if not token:
            frappe.throw("GitHub Token is required")
