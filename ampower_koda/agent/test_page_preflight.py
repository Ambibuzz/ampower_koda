"""The browser-free route check that runs before check_page starts Chromium.

Run with: bench --site <site> run-tests --module ampower_koda.agent.test_page_preflight
"""

import unittest
from unittest import mock

import frappe

from ampower_koda.agent import page_preflight, render_check, render_runner
from ampower_koda.agent.page_preflight import called_methods, preflight


def _with_doctype_layouts(layouts):
    """frappe.get_all that answers DocType Layout with ``layouts`` and passes every other call through."""
    real_get_all = frappe.get_all

    def get_all(doctype, *args, **kwargs):
        if doctype == "DocType Layout":
            return [frappe._dict(layout) for layout in layouts]
        return real_get_all(doctype, *args, **kwargs)

    return mock.patch("frappe.get_all", side_effect=get_all)


def _with_workspaces(names):
    return mock.patch("frappe.desk.desktop.get_workspace_sidebar_items",
                      return_value={"pages": [{"name": name} for name in names]})


class CalledMethodsTest(unittest.TestCase):
    def test_finds_each_literal_form_once(self):
        script = """
            frappe.call({method: "app.module.api.load", args: {}});
            frappe.call('app.module.api.save', {name});
            frappe.xcall(`app.module.api.load`);
            frappe.call({"method": "app.module.api.quoted"});
        """
        self.assertEqual(called_methods(script),
                         ["app.module.api.load", "app.module.api.quoted", "app.module.api.save"])

    def test_skips_document_methods_and_runtime_paths(self):
        script = 'frm.call({method: "make_copy"}); frappe.call({method: base + ".load"});'
        self.assertEqual(called_methods(script), [])

    def test_positional_name_without_a_dot_is_a_server_script_api(self):
        self.assertEqual(called_methods('frappe.xcall("koda_api", {})'), ["koda_api"])


class RouteTest(unittest.TestCase):
    def test_route_outside_desk_is_not_checked(self):
        self.assertTrue(preflight("/login")["ok"])
        self.assertEqual(preflight("/login")["kind"], "other")

    def test_app_prefix_query_and_encoding(self):
        result = preflight("/app/query-report/No%20Such%20Report?x=1")
        self.assertEqual(result["kind"], "report")
        self.assertIn("no Report named 'No Such Report'", result["problem"])

    def test_unknown_user(self):
        self.assertIn("no User", preflight("/desk/user", "nobody@koda.invalid")["problem"])


class SiteRouteTest(unittest.TestCase):
    def test_doctype_list_and_form(self):
        self.assertEqual(preflight("/desk/user"), {"ok": True, "kind": "list", "target": "User", "problem": "",
                                                 "call_problems": []})
        self.assertTrue(preflight("/desk/user/Administrator")["ok"])
        missing = preflight("/desk/user/nobody@koda.invalid")
        self.assertEqual(missing["kind"], "form")
        self.assertIn("no User named", missing["problem"])

    def test_single_doctype_is_a_form(self):
        self.assertEqual(preflight("/desk/system-settings")["kind"], "form")

    def test_view_factory_routes_are_left_to_the_browser(self):
        result = preflight("/desk/List/ToDo")
        self.assertTrue(result["ok"])
        self.assertEqual(result["kind"], "view")

    def test_new_document_routes(self):
        self.assertTrue(preflight("/desk/user/new")["ok"])
        self.assertTrue(preflight("/desk/user/new-user-1")["ok"])

    def test_doctype_the_user_cannot_read(self):
        self.assertIn("cannot read DocType 'User'", preflight("/desk/user", "Guest")["problem"])

    def test_doctype_layout_route(self):
        with _with_doctype_layouts([{"name": "Koda Staff", "document_type": "User"}]):
            self.assertEqual(preflight("/desk/koda-staff")["target"], "User")
            self.assertTrue(preflight("/desk/koda-staff")["ok"])
            self.assertIn("DocType Layout route 'koda-staff'", preflight("/desk/koda-staff", "Guest")["problem"])

    def test_workspace_the_user_sees(self):
        with _with_workspaces(["Koda Space"]):
            self.assertEqual(preflight("/desk/koda-space"),
                             {"ok": True, "kind": "workspace", "target": "Koda Space", "problem": "",
                              "call_problems": []})

    def test_workspace_the_user_does_not_see(self):
        with _with_workspaces([]):
            self.assertIn("no Page named 'koda-space'", preflight("/desk/koda-space")["problem"])

    def test_private_workspace(self):
        with _with_workspaces(["Mine-administrator"]):
            self.assertTrue(preflight("/desk/private/mine")["ok"])
            self.assertIn("no private workspace 'other'", preflight("/desk/private/other")["problem"])

    def test_missing_page(self):
        self.assertIn("no Page named 'koda_no_such_page'", preflight("/desk/koda_no_such_page")["problem"])

    def test_page_roles(self):
        self.assertTrue(preflight("/desk/permission-manager")["ok"])
        self.assertIn("lacks the roles", preflight("/desk/permission-manager", "Guest")["problem"])
        self.assertNotIn("403", frappe.response)

    def test_page_without_script(self):
        with mock.patch("frappe.desk.desk_page.get", return_value={"script": ""}):
            self.assertIn("has no script", preflight("/desk/koda_any_page")["problem"])

    def test_page_with_only_what_load_assets_adds(self):
        script = 'frappe.templates["row"] = \'<div></div>\';\n\n\n//# sourceURL=koda_any_page.js'
        with mock.patch("frappe.desk.desk_page.get", return_value={"script": script}):
            self.assertIn("has no script", preflight("/desk/koda_any_page")["problem"])

    def test_any_script_counts_without_parsing_it(self):
        script = "/* written later */\n\n//# sourceURL=koda_any_page.js"
        with mock.patch("frappe.desk.desk_page.get", return_value={"script": script}):
            self.assertTrue(preflight("/desk/koda_any_page")["ok"])

    def test_page_calling_broken_methods(self):
        script = ('frappe.call({method: "frappe.client.get_list"}); frappe.xcall("frappe.client.koda_missing"); '
                  'frappe.call("frappe.utils.now");')
        with mock.patch("frappe.desk.desk_page.get", return_value={"script": script}):
            result = preflight("/desk/koda_any_page")
        self.assertTrue(result["ok"])
        problem = "; ".join(result["call_problems"])
        self.assertNotIn("get_list", problem)
        self.assertIn("frappe.client.koda_missing does not resolve", problem)
        self.assertIn("frappe.utils.now is not whitelisted", problem)

    def test_method_closed_to_guest(self):
        with mock.patch("frappe.desk.desk_page.get", return_value={"script": 'frappe.xcall("frappe.client.get_list")'}):
            problem = "; ".join(preflight("/desk/koda_any_page", "Guest")["call_problems"])
        self.assertIn("not open to Guest", problem)

    def test_server_script_api(self):
        script_map = {"_api": {"koda_api": "Koda API"}}
        previous_user = frappe.session.user
        with mock.patch("frappe.core.doctype.server_script.server_script_utils.get_server_script_map",
                        return_value=script_map):
            self.assertEqual(page_preflight._method_problem("koda_api"), "")
            frappe.set_user("Guest")
            try:
                # frappe.db is a proxy; patching its class reaches the connection the check uses.
                with mock.patch.object(frappe.db.__class__, "get_value", return_value=0):
                    self.assertIn("not open to Guest", page_preflight._method_problem("koda_api"))
            finally:
                frappe.set_user(previous_user)

    def test_session_user_is_restored(self):
        before = frappe.session.user
        preflight("/desk/user", "Guest")
        self.assertEqual(frappe.session.user, before)


class FrappeApiTest(unittest.TestCase):
    """Unmocked calls into Frappe, so an upgrade that changes them fails here instead of skipping every preflight."""

    def test_workspace_sidebar_lists_workspaces(self):
        workspaces = page_preflight._visible_workspaces()
        self.assertTrue(workspaces)
        self.assertTrue(all(isinstance(name, str) for name in workspaces.values()))

    def test_standard_page_has_no_call_problems(self):
        result = preflight("/desk/permission-manager")
        self.assertTrue(result["ok"])
        self.assertEqual(result["call_problems"], [])


class RunnerTest(unittest.TestCase):
    def test_route_that_opens(self):
        report = {}
        self.assertTrue(render_runner.route_opens("/desk/user", "Administrator", report))
        self.assertEqual(report, {})

    def test_route_that_does_not_open_stops_the_browser(self):
        report = {}
        self.assertFalse(render_runner.route_opens("/desk/koda_no_such_page", "Administrator", report))
        self.assertIn("no Page named 'koda_no_such_page'", report["preflight"])

    def test_call_problems_are_noted_and_the_browser_runs(self):
        result = {"ok": True, "problem": "", "call_problems": ["a.b.c is not whitelisted"]}
        report = {}
        with mock.patch("ampower_koda.agent.page_preflight.preflight", return_value=result):
            self.assertTrue(render_runner.route_opens("/desk/x", "Administrator", report))
        self.assertEqual(report, {"preflight_calls": ["a.b.c is not whitelisted"]})

    def test_broken_preflight_leaves_the_check_to_the_browser(self):
        report = {}
        with mock.patch("ampower_koda.agent.page_preflight.preflight", side_effect=RuntimeError("gone")), \
                mock.patch.object(frappe.db.__class__, "rollback") as rollback:
            self.assertTrue(render_runner.route_opens("/desk/x", "Administrator", report))
        rollback.assert_called_once()
        self.assertEqual(report, {"preflight_skipped": "RuntimeError: gone"})


class ReportTest(unittest.TestCase):
    def test_failed_preflight_reports_without_a_page(self):
        text = render_check._format({"preflight": "there is no Page named 'x'"})
        self.assertEqual(text, "RENDER_FAILED: there is no Page named 'x'. The browser was not started.")

    def test_failed_preflight_keeps_the_read_only_note(self):
        text = render_check._format({"preflight": "there is no Page named 'x'", "unregistered": "x/x.json",
                                     "user": "someone@example.com"})
        self.assertIn("not registered: x/x.json", text)
        self.assertIn("signed in as someone@example.com", text)

    def test_call_problems_are_a_warning_not_a_failure(self):
        text = render_check._format({"preflight_calls": ["a.b.c is not whitelisted (add @frappe.whitelist())"],
                                     "url": "/desk/x", "status": 200})
        self.assertTrue(text.startswith("RENDER_OK"))
        self.assertIn("warning: its script names server methods that would fail when called: a.b.c", text)

    def test_call_problems_are_shown_beside_a_browser_failure(self):
        text = render_check._format({"preflight_calls": ["a.b.c is not whitelisted (add @frappe.whitelist())"],
                                     "url": "/desk/x", "status": 500})
        self.assertTrue(text.startswith("RENDER_FAILED: HTTP 500"))
        self.assertNotIn("a.b.c", text.splitlines()[0])
        self.assertIn("warning: its script names server methods", text)

    def test_skipped_preflight_is_reported(self):
        text = render_check._format({"preflight_skipped": "OperationalError: gone", "url": "/desk/x", "status": 200})
        self.assertIn("route check before the browser did not run (OperationalError: gone)", text)
