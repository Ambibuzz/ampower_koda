import frappe
from frappe.model.document import Document


class AIAgentPromptConfiguration(Document):

    def validate(self):
        existing = frappe.get_all(
            "AI Agent Prompt Configuration",
            filters={
                "parent": self.parent,
                "parenttype": self.parenttype,
                "prompt_key": self.prompt_key,
                "name": ["!=", self.name]
            },
            limit=1
        )
        if existing:
            frappe.throw(
                f"Prompt Type '{self.prompt_key}' already exists for this document."
            )