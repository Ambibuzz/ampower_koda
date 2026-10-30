"""Check that a Desk route opens for a user, without a browser.

Desk resolves /desk/<first>/... in the browser (router.js convert_to_standard_route): a workspace the user
sees, then /private/<workspace>, then a DocType the user can read or a DocType Layout, then a Page named
<first>. v16.50 first takes a leading sidebar off the route (/desk/stock/item). This module asks the site the
same questions, as the same user, so a route that would show "Page not found" or "No permission" is caught
before Chromium starts. Needs a connected site (frappe.init and frappe.connect). It writes nothing, but
like Desk it imports the Python modules the route reaches: a Page's get_context (for a jinja template) and
the server methods its script names, so their module-level code runs.

Where router.js changed between releases (private workspace URLs, sidebars in routes), the installed
release decides. A development build could carry any of them, so there a route any of them opens passes.

Only verdicts the site answers for certain stop the browser. Server methods found by scanning the script
are reported as call_problems, and the browser still runs.

From a console:
    bench --site <site> execute ampower_koda.agent.page_preflight.preflight \
        --kwargs "{'route': '/desk/my_page', 'user': 'someone@example.com'}"
"""

from __future__ import annotations

import re
from functools import cached_property
from urllib.parse import unquote

import frappe
from frappe.utils.user import UserPermissions

DESK_PREFIXES = ("desk", "app")
# router.js render_page: /desk/<view>/<more> with a frappe.views.<View>Factory opens that view, not a Page.
VIEW_FACTORY_ROUTES = ("List", "Form", "Tree", "Report")
# Pages Desk registers in its own JavaScript (frappe.standard_pages), which have no Page document.
STANDARD_PAGES = ("query-report", "Workspaces")
# Server methods named literally in a Page's script: frappe.call({method: "a.b.c"}), frappe.call("a.b.c"),
# frappe.xcall("a.b.c"). A path built at runtime is not seen. A method key without a dot is a document
# method (frm.call), which this does not check; a positional name without a dot is a Server Script API.
# Comments are not told apart from code, so a commented-out call is checked too; the result is only a warning.
CALLED_METHOD_PATTERNS = (
    re.compile(r"""\bmethod["']?\s*:\s*["'`]([\w.]+\.\w+)["'`]"""),
    re.compile(r"""\bfrappe\.(?:call|xcall)\(\s*["'`]([\w.]+)["'`]"""),
)
# Page.load_assets puts each .html file of the Page before its script as one frappe.templates line, and a
# sourceURL line after the script read from <page>.js.
LOAD_ASSETS_LINE = re.compile(r"^(?:frappe\.templates\[.*|//# sourceURL=.*)$", re.MULTILINE)


def _slug(name: str) -> str:
    """The URL segment Desk makes from a DocType or Workspace name (frappe.router.slug)."""
    return name.lower().replace(" ", "-")


def _result(kind: str, target: str, problem: str = "", call_problems: list[str] | None = None) -> dict:
    return {"ok": not problem, "kind": kind, "target": target, "problem": problem,
            "call_problems": call_problems or []}


def _error_text(exc: Exception) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:300]


def _route_parts(route: str) -> list[str] | None:
    """The path segments after /desk or /app, or None when the route is not a Desk route."""
    path = str(route or "").split("?", 1)[0].split("#", 1)[0]
    parts = [unquote(part) for part in path.strip("/").split("/") if part]
    if not parts or parts[0] not in DESK_PREFIXES:
        return None
    return parts[1:]


def _shell_slug(name: str) -> str:
    """The URL segment v16 Desk makes from a sidebar or module name (router.js shell_slug)."""
    return "-".join(name.lower().replace("&", " and ").split())


def _visible_workspaces() -> dict[str, dict]:
    """The workspaces Desk lists for the session user, by URL segment (desk.js setup_workspaces).

    The call applies the workspace roles and domains, and includes the user's private workspaces. It is
    get_workspace_sidebar_items on v15 and get_workspaces on v16.
    """
    from frappe.desk import desktop
    list_workspaces = getattr(desktop, "get_workspaces", None) or desktop.get_workspace_sidebar_items
    pages = list_workspaces()["pages"]
    return {_slug(page["name"]): page for page in pages if page}


def _frappe_release() -> tuple[int, int] | None:
    """The installed Frappe as (major, minor), or None for a development build, whose router could be any."""
    match = re.match(r"(\d+)\.(\d+)", frappe.__version__)
    if not match or "dev" in frappe.__version__:
        return None
    return int(match.group(1)), int(match.group(2))


def _desk_has_shells() -> bool:
    """Whether router.js takes sidebars off routes and has the Private sidebar, which v16.50 added together."""
    release = _frappe_release()
    if release is None:
        import frappe.boot
        return hasattr(frappe.boot, "get_module_sidebars")
    return release >= (16, 50)


def _shell_slugs() -> set[str]:
    """The segments v16 Desk takes off the front of a route as a sidebar: /desk/stock/item is the Item list.

    router.js setup_shell_routes reads them from boot.module_sidebars, the session user's sidebars, and
    every installed module whether or not the user sees it. private is left out, as router.js does: it
    reads that one itself.
    """
    if not _desk_has_shells():
        return set()
    import frappe.boot
    names = list(frappe.boot.get_module_sidebars()) + frappe.get_all("Module Def", pluck="name")
    return {_shell_slug(name) for name in names} - {"private"}


class DeskTables:
    """What router.js reads a route against, for the session user; each table is read when first needed."""

    def __init__(self, user: str):
        self.user = user

    @cached_property
    def workspaces(self) -> dict[str, dict]:
        return _visible_workspaces()

    @cached_property
    def readable(self) -> list[str]:
        return UserPermissions(self.user).get_can_read()

    @cached_property
    def doctype_routes(self) -> dict[str, str]:
        return _doctype_routes(self.readable)

    @cached_property
    def shells(self) -> set[str]:
        return _shell_slugs()

    @cached_property
    def has_shells(self) -> bool:
        return _desk_has_shells()


def _segment_kind(segment: str, desk: DeskTables) -> str:
    """What a segment names to v16 Desk (router.js segment_kind), or "" when it names nothing."""
    if segment in desk.workspaces:
        return "workspace"
    if segment == "private":
        return "private"
    if segment in desk.doctype_routes:
        return "doctype"
    if segment in STANDARD_PAGES or frappe.db.exists("Page", segment):
        return "page"
    if segment.capitalize() in VIEW_FACTORY_ROUTES:
        return "view"
    return ""


def _without_shell(parts: list[str], desk: DeskTables) -> list[str]:
    """The route without a leading sidebar segment, as router.js take_shell_from drops it on v16."""
    if len(parts) < 2 or parts[0] not in desk.shells:
        return parts
    # A sidebar named like a DocType stays that DocType, and the segment after it must name something.
    if _segment_kind(parts[0], desk) == "doctype" or not _segment_kind(parts[1], desk):
        return parts
    return parts[1:]


def _private_lookup(segment: str, user: str) -> tuple[set[str], bool]:
    """How this release's router.js finds /desk/private/<segment>: the workspace names it matches, and
    whether a title matches too.

    Each release is the first v16 tag holding that change to router.js; v15 has looked up "<segment>-<user>"
    as typed since 2024. A development build could hold any of them, so it accepts every one.
    """
    as_typed = f"{segment}-{user.lower()}"
    release = _frappe_release()
    if release is None:
        return {as_typed, _slug(as_typed), _slug(segment)}, True
    if release >= (16, 50):
        return {_slug(segment)}, True
    if release >= (16, 14):
        return {_slug(segment)}, False
    if release >= (16, 5):
        return {_slug(as_typed)}, False
    return {as_typed}, False


def _own_private_workspace(segment: str, user: str, workspaces: dict[str, dict]) -> dict | None:
    """The user's private workspace that ``segment`` names, as this release's router.js looks it up."""
    names, title_matches = _private_lookup(segment, user)
    for page in workspaces.values():
        if page.get("public") or page.get("for_user") != user:
            continue
        if _slug(page["name"]) in names:
            return page
        if title_matches and _slug(page.get("title") or "") in names:
            return page
    return None


def _check_private_workspace(rest: list[str], user: str, desk: DeskTables) -> dict:
    if not rest and desk.has_shells:
        # The Private sidebar opens its first page, or its empty state when it has none.
        return _result("workspace", "private")
    if not rest:
        return _result("workspace", "private", "/desk/private needs a workspace after it; Desk would say "
                                               "the workspace does not exist")
    page = _own_private_workspace(rest[0], user, desk.workspaces)
    if page:
        return _result("workspace", page["name"])
    if desk.has_shells and _segment_kind(rest[0], desk):
        # v16 then reads private as the Private sidebar with an ordinary route inside it.
        return _resolve(rest, user, desk)
    return _result("workspace", rest[0], f"{user} has no private workspace {rest[0]!r}; Desk would say "
                                         "the workspace does not exist")


def _doctype_routes(readable: list[str]) -> dict[str, str]:
    """Desk's route table (router.js setup): readable DocTypes, then DocType Layouts, which replace them."""
    routes = {_slug(name): name for name in readable}
    for layout in frappe.get_all("DocType Layout", fields=["name", "document_type"]):
        routes[_slug(layout.name)] = layout.document_type
    return routes


def _doctype_named(segment: str) -> str:
    names = frappe.get_all("DocType", pluck="name")
    return next((name for name in names if _slug(name) == segment), "")


def _check_doctype(doctype: str, rest: list[str], user: str) -> dict:
    """A list view (/desk/<doctype>, /desk/<doctype>/view/...) or a form (/desk/<doctype>/<name>)."""
    meta = frappe.get_meta(doctype)
    if meta.issingle:
        return _result("form", doctype)
    if not rest or rest[0] == "view":
        return _result("list", doctype)
    docname = "/".join(rest)
    target = f"{doctype} {docname}"
    if meta.is_virtual:
        # A virtual DocType's records live outside the database, so there is nothing to look up here.
        return _result("form", target)
    if not frappe.db.exists(doctype, docname):
        if docname.startswith("new"):
            # formview.js opens a missing name that starts with "new" (new, new-sales-invoice-1) as a new document.
            return _result("form", target)
        return _result("form", target, f"there is no {doctype} named {docname!r}; the form would not open")
    if not frappe.has_permission(doctype, "read", doc=docname, user=user):
        return _result("form", target, f"{user} may not read {doctype} {docname!r}")
    return _result("form", target)


def _check_report(rest: list[str]) -> dict:
    if not rest:
        return _result("report", "", "/desk/query-report needs a report name after it")
    report_name = rest[0]
    if not frappe.db.exists("Report", report_name):
        return _result("report", report_name, f"there is no Report named {report_name!r}")
    # get_report_doc applies the same role check as the page's own get_script call.
    from frappe.desk.query_report import get_report_doc
    try:
        get_report_doc(report_name)
    except frappe.PermissionError:
        return _result("report", report_name, f"{frappe.session.user} lacks the roles for Report {report_name!r}")
    except Exception as exc:
        return _result("report", report_name, f"loading Report {report_name!r} raised {_error_text(exc)}")
    finally:
        frappe.clear_messages()
    return _result("report", report_name)


def called_methods(script: str) -> list[str]:
    """The server method paths a script names literally, in first-seen order."""
    found = []
    for pattern in CALLED_METHOD_PATTERNS:
        for match in pattern.finditer(script or ""):
            if match.group(1) not in found:
                found.append(match.group(1))
    return found


def _server_script_problem(path: str, script_name: str) -> str:
    """Why an API Server Script would refuse the call (server_script.execute_api_server_script)."""
    if frappe.session.user != "Guest":
        return ""
    if frappe.db.get_value("Server Script", script_name, "allow_guest"):
        return ""
    return f"{path} is a Server Script API not open to Guest (Allow Guest)"


def _method_problem(path: str) -> str:
    """Why Desk's call to ``path`` would fail before the method runs, as frappe.handler.execute_cmd checks."""
    from frappe.core.doctype.server_script.server_script_utils import get_server_script_map
    resolved = frappe.override_whitelisted_method(path)
    script_name = get_server_script_map().get("_api", {}).get(resolved)
    if script_name:
        return _server_script_problem(path, script_name)
    # get_attr refuses modules outside the installed apps, so this imports only what Desk's call would.
    try:
        function = frappe.get_attr(resolved)
    except Exception as exc:
        return f"{path} does not resolve to a function or Server Script API ({_error_text(exc)})"
    try:
        frappe.is_whitelisted(function)
    except frappe.PermissionError:
        if frappe.session.user == "Guest":
            return f"{path} is not open to Guest (allow_guest)"
        return f"{path} is not whitelisted (add @frappe.whitelist())"
    return ""


def _checked_method_problem(path: str) -> str:
    try:
        return _method_problem(path)
    finally:
        # get_attr and is_whitelisted report through msgprint; this is a check, not a request, so nothing is shown.
        frappe.clear_messages()


def _has_script(script: str) -> bool:
    """Whether load_assets found any script for the Page, in its own .js file or through a page_js hook.

    An empty result stops the browser, so it must be certain: only the lines load_assets adds are removed,
    without parsing JavaScript, and a hook's script counts because it may be what renders the Page.
    """
    return bool(LOAD_ASSETS_LINE.sub("", script).strip())


def _check_page(page_name: str) -> dict:
    # desk_page.get is what getpage runs when Desk opens the route; it also loads the Page's script.
    from frappe.desk.desk_page import get as get_page
    try:
        page = get_page(page_name)
    except frappe.DoesNotExistError:
        return _result("page", page_name, f"there is no Page named {page_name!r} on the site (Frappe would show "
                                          "Page not found; register the Page or check the route)")
    except frappe.PermissionError:
        # get sets the 403 flag on the response it would send; this check sends none.
        frappe.response.pop("403", None)
        return _result("page", page_name, f"{frappe.session.user} lacks the roles of Page {page_name!r} "
                                          "(its Has Role table)")
    except Exception as exc:
        return _result("page", page_name, f"loading Page {page_name!r} raised {_error_text(exc)}")
    finally:
        frappe.clear_messages()
    script = page.get("script") or ""
    if not _has_script(script):
        return _result("page", page_name, f"Page {page_name!r} has no script ({frappe.scrub(page_name)}.js in its "
                                          "folder, or a page_js hook), so Desk would show an empty page")
    # Found by scanning the script, so a mistaken match must not keep the browser from showing the page.
    # A page_js hook's calls are included: they run on this page, so a broken one fails here too.
    call_problems = []
    for method_path in called_methods(script):
        problem = _checked_method_problem(method_path)
        if problem:
            call_problems.append(problem)
    return _result("page", page_name, call_problems=call_problems)


def _resolve(parts: list[str], user: str, desk: DeskTables) -> dict:
    if not parts:
        return _result("home", "")
    first, rest = parts[0], parts[1:]
    if first in desk.workspaces:
        return _result("workspace", desk.workspaces[first]["name"])
    if first == "private":
        return _check_private_workspace(rest, user, desk)
    doctype = desk.doctype_routes.get(first, "")
    if doctype and doctype not in desk.readable:
        # A DocType Layout is routed for every user, but its DocType still has to load for this one.
        return _result("list", doctype, f"{user} cannot read DocType {doctype!r}, which the DocType Layout "
                                        f"route {first!r} opens")
    if doctype:
        return _check_doctype(doctype, rest, user)
    unreadable = _doctype_named(first)
    if unreadable:
        # Desk only maps DocTypes the user can read, so the route falls through to a Page lookup and fails.
        return _result("list", unreadable, f"{user} cannot read DocType {unreadable!r}, so Desk would show "
                                           "Page not found")
    if rest and first.capitalize() in VIEW_FACTORY_ROUTES:
        # /desk/List/ToDo and the like: the view resolves its DocType in the browser, which reports what it finds.
        return _result("view", "/".join(parts))
    if first == "query-report":
        return _check_report(rest)
    if first == "Workspaces":
        # Desk's own workspace view, which has no Page document to look up.
        return _result("workspace", "/".join(rest))
    return _check_page(first)


def preflight(route: str, user: str = "Administrator") -> dict:
    """What the Desk route opens for ``user``: {ok, kind, target, problem, call_problems}.

    ``problem`` is empty when the route opens. ``call_problems`` lists server methods a Page's script names that
    would fail; they do not make ``ok`` false.
    """
    parts = _route_parts(route)
    if parts is None:
        return _result("other", str(route), "")
    # What this request cached was worked out for the caller (v16 caches get_workspaces this way), so the
    # check starts without it and leaves the caller's values as they were.
    request_cache = getattr(frappe.local, "request_cache", None)
    callers_request_cache = dict(request_cache or {})
    if request_cache:
        request_cache.clear()
    try:
        stored_user = frappe.db.get_value("User", user, "name")
        if not stored_user:
            return _result("other", str(route), f"there is no User {user!r} to sign in as")
        # The lookup ignores case, but Desk compares user names exactly (a private workspace's for_user).
        return _resolve_as(parts, stored_user)
    finally:
        if request_cache is not None:
            request_cache.clear()
            request_cache.update(callers_request_cache)


def _resolve_as(parts: list[str], user: str) -> dict:
    previous_user = frappe.session.user
    frappe.set_user(user)
    try:
        desk = DeskTables(user)
        return _resolve(_without_shell(parts, desk), user, desk)
    finally:
        frappe.set_user(previous_user)
