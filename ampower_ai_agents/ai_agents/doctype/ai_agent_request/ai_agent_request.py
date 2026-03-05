# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# For license information, please see license.txt

import frappe
from frappe.model.document import Document


class AIAgentRequest(Document):
    def before_insert(self):
        if not self.owner:
            self.owner = frappe.session.user
