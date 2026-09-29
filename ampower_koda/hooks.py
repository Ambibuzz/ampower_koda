app_name = "ampower_koda"
app_title = "Ampower Koda"
app_publisher = "Ambibuzz Technologies LLP"
app_description = "AI Coding Agent for Frappe apps"
app_email = "buzz@ambibuzz.com"
app_license = "MIT"
required_apps = ["frappe"]

override_doctype_class = {
    "HD Ticket": "ampower_koda.overrides.hd_ticket.CustomHDTicket"
}

fixtures = [
    {
        "doctype": "Custom Field",
        "filters": [["name", "=", "HD Ticket-custom_koda_pending_draft"]],
    },
    {
        "doctype": "HD Form Script",
        "filters": [["name", "=", "HD Ticket-Koda Update"]],
    },
]

doctype_js = {
    "Agent Request": "ampower_koda/doctype/agent_request/agent_request.js",
    "Agent Settings": "ampower_koda/doctype/agent_settings/agent_settings.js",
}

app_include_js = [
    "/assets/ampower_koda/js/hd_ticket_intake_listener.js",
]
