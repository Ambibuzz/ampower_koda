# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors

import frappe


def log_agent_error(title: str, message: str | None = None) -> None:
    """Write an entry to Frappe Error Log. Never raises."""
    try:
        frappe.log_error(message or frappe.get_traceback(), title)
    except Exception:
        pass
