# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# For license information, please see license.txt

import frappe
from frappe import _
from frappe.model.document import Document


class AgentSettings(Document):
    def validate(self):
        if not self.enable_ai_agent:
            return

        provider = self.default_ai_provider or "OpenAI"
        key_map = {
            "OpenAI": ("openai_api_key", "OpenAI API Key"),
            "Gemini": ("google_api_key", "Google API Key"),
            "Claude": ("anthropic_api_key", "Anthropic API Key"),
        }
        field, label = key_map.get(provider, key_map["OpenAI"])
        if not getattr(self, field, None):
            frappe.throw(
                _("{0} is required when {1} is the default provider").format(label, provider)
            )
