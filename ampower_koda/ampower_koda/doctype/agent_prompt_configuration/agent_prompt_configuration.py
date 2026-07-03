# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# For license information, please see license.txt

import frappe
from frappe import _
from frappe.model.document import Document


class AgentPromptConfiguration(Document):
    def validate(self):
        existing = frappe.get_all(
            "Agent Prompt Configuration",
            filters={
                "parent": self.parent,
                "parenttype": self.parenttype,
                "prompt_key": self.prompt_key,
                "name": ["!=", self.name],
            },
            limit=1,
        )
        if existing:
            frappe.throw(
                _("Prompt Type '{0}' already exists for this document.").format(self.prompt_key)
            )
