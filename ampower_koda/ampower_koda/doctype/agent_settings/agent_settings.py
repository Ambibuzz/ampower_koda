# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# For license information, please see license.txt

import frappe
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
            frappe.throw(f"{label} is required when {provider} is the default provider")
