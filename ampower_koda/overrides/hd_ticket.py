# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
"""Overrides HD Ticket's mark_seen to fix a type error in the official
helpdesk app: clear_notifications() expects ticket as str, but self.name
is an int on this bench's autoname, causing a 500 on every ticket view.
"""

import frappe
from helpdesk.helpdesk.doctype.hd_ticket.hd_ticket import HDTicket
from helpdesk.helpdesk.doctype.hd_notification.utils import clear as clear_notifications


class CustomHDTicket(HDTicket):
    @frappe.whitelist()
    def mark_seen(self):
        self.add_viewed(unique_views=True, force=True)
        self.add_seen()
        clear_notifications(ticket=str(self.name))
