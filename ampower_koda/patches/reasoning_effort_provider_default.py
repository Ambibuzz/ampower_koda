# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
import frappe


def execute():
    """Move Agent Settings off the removed "Provider default" Reasoning Effort onto high, the new default.

    Until then the agent already runs such a site at high, but Agent Settings cannot be saved while the
    value is outside the field's options.
    """
    if frappe.db.get_single_value("Agent Settings", "reasoning_effort") == "Provider default":
        frappe.db.set_single_value("Agent Settings", "reasoning_effort", "high")
