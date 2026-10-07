# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
import frappe


RENAME_MAP = [
    ("AI Agent Prompt Configuration", "Agent Prompt Configuration"),
    ("AI Agent Settings", "Agent Settings"),
    ("AI Agent Request", "Agent Request"),
]


def execute():
    """Move legacy AI Agent DocTypes and their data to the shorter names.

    Runs before model sync, so on an upgrade the legacy DocType is renamed in
    place (table, links and child parenttypes) before sync creates an empty
    replacement. A site where an earlier migrate already created the new names
    gets its legacy records and settings copied into them instead, and only when
    the new ones are still empty. Nothing legacy is dropped; re-running is a no-op.
    """
    for old_name, new_name in RENAME_MAP:
        if not frappe.db.exists("DocType", old_name):
            continue
        if not frappe.db.exists("DocType", new_name):
            frappe.rename_doc("DocType", old_name, new_name, force=True)
        elif not _copy_legacy_records(old_name, new_name):
            continue
        # Neither the rename nor the table copy carries Password fields (__Auth).
        _copy_legacy_passwords(old_name, new_name)

    _migrate_user_settings()
    frappe.db.commit()


def _is_single(doctype: str) -> bool:
    return bool(frappe.db.get_value("DocType", doctype, "issingle"))


def _copy_legacy_records(old_name: str, new_name: str) -> bool:
    """Copy legacy rows into the new DocType when it has none; True when copied."""
    if _is_single(old_name):
        if frappe.db.sql("SELECT 1 FROM `tabSingles` WHERE `doctype` = %s LIMIT 1", new_name):
            return False
        frappe.db.sql(
            """INSERT INTO `tabSingles` (`doctype`, `field`, `value`)
            SELECT %s, `field`, `value` FROM `tabSingles` WHERE `doctype` = %s""",
            (new_name, old_name),
        )
        frappe.db.sql(
            "UPDATE `tabSingles` SET `value` = %s WHERE `doctype` = %s AND `field` = 'name'",
            (new_name, new_name),
        )
        return True

    if not (frappe.db.table_exists(old_name) and frappe.db.table_exists(new_name)):
        return False
    if frappe.db.sql(f"SELECT 1 FROM `tab{new_name}` LIMIT 1"):
        return False
    if not frappe.db.sql(f"SELECT 1 FROM `tab{old_name}` LIMIT 1"):
        return False
    new_columns = set(frappe.db.get_table_columns(new_name))
    columns = ", ".join(f"`{c}`" for c in frappe.db.get_table_columns(old_name) if c in new_columns)
    frappe.db.sql(f"INSERT INTO `tab{new_name}` ({columns}) SELECT {columns} FROM `tab{old_name}`")
    # Child rows point at their parent DocType by name.
    if "parenttype" in new_columns:
        for old_parent, new_parent in RENAME_MAP:
            frappe.db.sql(
                f"UPDATE `tab{new_name}` SET `parenttype` = %s WHERE `parenttype` = %s",
                (new_parent, old_parent),
            )
    return True


def _copy_legacy_passwords(old_name: str, new_name: str) -> None:
    """Copy stored secrets (API keys, tokens) to the new DocType, never overwriting one."""
    single = _is_single(new_name)
    rows = frappe.db.sql(
        "SELECT `name`, `fieldname`, `password`, `encrypted` FROM `__Auth` WHERE `doctype` = %s",
        old_name, as_dict=True,
    )
    for row in rows:
        name = new_name if single else row.name
        if frappe.db.sql(
            "SELECT 1 FROM `__Auth` WHERE `doctype` = %s AND `name` = %s AND `fieldname` = %s",
            (new_name, name, row.fieldname),
        ):
            continue
        frappe.db.sql(
            """INSERT INTO `__Auth` (`doctype`, `name`, `fieldname`, `password`, `encrypted`)
            VALUES (%s, %s, %s, %s, %s)""",
            (new_name, name, row.fieldname, row.password, row.encrypted),
        )


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
