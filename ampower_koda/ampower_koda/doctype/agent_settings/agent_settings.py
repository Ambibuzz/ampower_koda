# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# For license information, please see license.txt

import frappe
from frappe import _
from frappe.model.document import Document


PROVIDER_KEY_FIELDS = {
    "OpenAI": ("openai_api_key", "OpenAI API Key"),
    "Gemini": ("google_api_key", "Google API Key"),
    "Claude": ("anthropic_api_key", "Anthropic API Key"),
    "OpenRouter": ("openrouter_api_key", "OpenRouter API Key"),
}


class AgentSettings(Document):
    def validate(self):
        self._validate_provider_key()
        self._validate_langsmith()

    def _validate_provider_key(self):
        if not self.enable_ai_agent:
            return

        provider = self.default_ai_provider or "OpenAI"
        field, label = PROVIDER_KEY_FIELDS.get(provider, PROVIDER_KEY_FIELDS["OpenAI"])
        if not getattr(self, field, None):
            frappe.throw(
                _("{0} is required when {1} is the default provider").format(label, provider)
            )

    def _validate_langsmith(self):
        """Tracing on with no key is the failure worth catching here.

        LangChain does not raise when its tracer cannot authenticate — it drops
        the trace and carries on — so the symptom is an empty project rather than
        an error, which is the kind of thing nobody notices for weeks.
        """
        if not self.enable_langsmith:
            return

        if not self.langsmith_api_key:
            frappe.throw(_("LangSmith API Key is required when LangSmith tracing is enabled"))

        self.langsmith_endpoint = (
            self.langsmith_endpoint or "https://api.smith.langchain.com"
        ).strip().rstrip("/")
        self.langsmith_project = (self.langsmith_project or "Koda").strip()
