# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
import frappe


RENAME_MAP = [
    ("AI Agent Prompt Configuration", "Agent Prompt Configuration"),
    ("AI Agent Settings", "Agent Settings"),
    ("AI Agent Request", "Agent Request"),
]


def execute():
    """Rename legacy AI Agent DocTypes to shorter names."""
    for old_name, new_name in RENAME_MAP:
        if not frappe.db.exists("DocType", old_name):
            continue
        if frappe.db.exists("DocType", new_name):
            continue
        frappe.rename_doc("DocType", old_name, new_name, force=True)

    _migrate_user_settings()
    frappe.db.commit()


def _migrate_user_settings():
    """Copy per-user form settings from old DocType name to new."""
    rows = frappe.db.sql(
        """SELECT `user`, `data` FROM `__UserSettings`
        WHERE doctype = %s""",
        "AI Agent Request",
        as_dict=True,
    )
    for row in rows:
        exists = frappe.db.exists(
            "__UserSettings",
            {"user": row.user, "doctype": "Agent Request"},
        )
        if exists:
            continue
        frappe.db.sql(
            """INSERT INTO `__UserSettings` (`user`, `doctype`, `data`)
            VALUES (%s, %s, %s)""",
            (row.user, "Agent Request", row.data),
        )
        frappe.cache.hdel("_user_settings", f"Agent Request::{row.user}")
