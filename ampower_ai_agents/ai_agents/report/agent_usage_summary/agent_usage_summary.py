import frappe
from frappe import _

def execute(filters=None):
    columns = [
        {"label": _("AI Provider"), "fieldname": "ai_provider", "fieldtype": "Data", "width": 150},
        {"label": _("Total Requests"), "fieldname": "total_requests", "fieldtype": "Int", "width": 150},
        {"label": _("Total Tokens Used"), "fieldname": "total_tokens_used", "fieldtype": "Int", "width": 150},
        {"label": _("Average Cost per Request"), "fieldname": "average_cost", "fieldtype": "Currency", "width": 150},
    ]

    data = get_report_data(filters)

    return columns, data

def get_report_data(filters):
    conditions = ""
    if filters.get("from_date"):
        conditions += " and creation >= %(from_date)s"
    if filters.get("to_date"):
        conditions += " and creation <= %(to_date)s"

    query = f"""
        SELECT ai_provider,
               COUNT(*) AS total_requests,
               SUM(tokens_used) AS total_tokens_used,
               AVG(cost_per_request) AS average_cost
        FROM `tabAI Agent Request` AS a
        WHERE docstatus < 2 {conditions}
        GROUP BY ai_provider
    """
    return frappe.db.sql(query, filters, as_dict=True)