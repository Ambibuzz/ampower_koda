app_name = "ampower_koda"
app_title = "Ampower Koda"
app_publisher = "Ambibuzz Technologies LLP"
app_description = "AI Coding Agent for Frappe apps"
app_email = "buzz@ambibuzz.com"
app_license = "MIT"
required_apps = ["frappe"]

# Page checks need Playwright's Chromium; migrate also covers benches that installed Koda earlier.
after_install = "ampower_koda.install.ensure_browser"
after_migrate = "ampower_koda.install.ensure_browser"

# No doctype_js: Frappe already loads each DocType's own colocated form script,
# and listing it here as well registered every form handler twice.
