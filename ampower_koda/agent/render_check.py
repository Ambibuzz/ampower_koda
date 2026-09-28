"""Open a Desk page in a headless browser and report what the user would see.

The model cannot otherwise see the pages it builds: a client script that never loads, a route that
is not found, a control whose handler throws, or a whitelisted method that fails only when the page
calls it. Each check runs in its own process (the worker never imports Playwright), signs in as
Administrator with a one-time login key, and can drive the page with a short list of steps.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from ampower_koda.agent import verification
from ampower_koda.agent.tools import _app_root

RENDER_TIMEOUT = 180  # each step waits for its calls and animations to settle
MAX_STEPS = 20
MAX_SNAPSHOT_CHARS = 6000
MAX_LAYOUT_LABELS = 60
# A report stays in the conversation for every later call (AGENT-0038 kept eight of 10-29k characters), so
# what the steps already show is cut: a step whose expects all passed keeps its first changed lines, and
# after steps the final snapshot and label list are halved. Failed steps and every non-diff line stay whole.
PASSED_STEP_DIFF_LINES = 8
STEPS_SNAPSHOT_CHARS = 3000
STEPS_LAYOUT_LABELS = 30
_DIFF_LINE = re.compile(r"^  (?:[+-] -|… \d+ more changed lines)")
MIN_READABLE_PX = 9
# Browser checks need the optional extra: pip install ampower_koda[browser] && playwright install chromium.
# A separate Playwright install is used only when named: KODA_PLAYWRIGHT_PYTHONPATH and
# PLAYWRIGHT_BROWSERS_PATH, or KODA_VERIFICATION_LAB holding python/ and browsers/.
LAB_ENV = "KODA_VERIFICATION_LAB"
RENDER_DIRECTORY = ".koda/render"

RENDER_TOOL_DESCRIPTION = (
    "Open a Desk page of the live site in a fresh headless browser, signed in as Administrator, run steps "
    "on it, and report what the user saw. route is the Desk path, e.g. '/desk/warehouse_traceability' (a "
    "Page's name; '/app/<page-name>' on older sites). steps is a JSON list run in order after the page "
    "loads: {\"click\": \"<button, link or option name, text or CSS>\"}, {\"fill\": \"<label, placeholder or "
    "CSS>\", \"value\": \"...\"}, {\"select\": \"<drop-down label>\", \"value\": \"<option>\"}, {\"press\": "
    "\"Enter\"}, {\"wait_ms\": 800}, {\"viewport\": [390, 844]}. Each step waits for the server calls it "
    "triggers. Any step may add \"expect\" (one or a list), checked after it: \"<text>\" (visible somewhere), "
    "{\"order\": [\"A\", \"B\"], \"axis\": \"x\"} (each visible, centres left to right; \"y\": top to bottom), "
    "{\"text\": \"X\", \"within\": \"<region's aria label, heading or CSS>\"}, {\"absent\": \"X\"} (not "
    "visible), {\"count\": \"<CSS or text>\", \"min\": 1, \"max\": 9}. A text expect proves only presence, not "
    "side, order, region or that something went away: check such claims with the matching form. A step of "
    "only expect and/or {\"layout\": true} looks at the page as it is; layout lists label positions there.\n"
    "The report gives, for the load and for EVERY step: the server calls made with their arguments and "
    "response; what changed on the page, in dialogs and alerts (accessibility-tree lines '-' removed, '+' "
    "added) or that nothing changed; each changed table's row count and whether each numeric or date "
    "column is ascending, descending or unsorted; what happened while the page settled (text shown only "
    "briefly, text changes with their times, animations with where they moved, animations still running, "
    "layout shifts, the page freezing with the script that was busy, and each element that grew, shrank or "
    "moved: over how long, where it stalled, whether it reversed); after a viewport step, content that is "
    "cut off and whether it can be scrolled to. "
    "Then errors, failed calls, the final snapshot and label positions (what is drawn left of what).\n"
    "Read every step against what the request asks, not only for errors: recompute numbers the page "
    "derives (totals, counts, order), check that a panel shows the record you opened, and treat a click "
    "that should change something but shows no change, a flash of wrong or empty content, a freeze, a "
    "stall, a bounce or a slow animation, or an animation that never stops as a defect. State carried "
    "between views often goes stale: after changing page, filter, sort or record, reopen what you opened "
    "before. Open and close every toggle (expand and collapse, show and hide). Exercise each control the "
    "task changed on at least two real records, and check phone width. Check once after building, fix everything it shows in "
    "one batch, confirm with at most one more check, then stop. During implementation a Page of this app is "
    "registered from its JSON before each check (a live-site change); in planning and review it is not, and "
    "the site shows the Page as last registered. Steps run the page's real server calls on the live site "
    "and are NOT rolled back: do not click actions that create, submit, cancel or delete records."
)

RUNNER = Path(__file__).with_name("render_runner.py")


def _lab_path(part: str) -> str:
    lab = os.environ.get(LAB_ENV, "").strip()
    return os.path.join(lab, part) if lab and os.path.isdir(os.path.join(lab, part)) else ""


def _playwright_path() -> str:
    return os.environ.get("KODA_PLAYWRIGHT_PYTHONPATH") or _lab_path("python")


def _browsers_path() -> str:
    return os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or _lab_path("browsers")


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[:limit * 2 // 3], text[-(limit // 3):]
    return f"{head}\n… [{len(text) - len(head) - len(tail):,} chars of the snapshot omitted] …\n{tail}"


def _step_lines(lines: list[str], outcome: str) -> list[str]:
    """A step's report lines, its accessibility changes cut to the first few when everything it checked passed."""
    if outcome != "ok" or any(line.startswith("  expect ") and ": FAIL" in line for line in lines):
        return lines
    diff = [n for n, line in enumerate(lines) if _DIFF_LINE.match(line)]
    if len(diff) <= PASSED_STEP_DIFF_LINES:
        return lines
    cut = set(diff[PASSED_STEP_DIFF_LINES:])
    # The runner's own "… N more" line counts lines it already dropped; the note replaces it and counts both.
    hidden = sum(int(m.group(1)) if (m := re.match(r"^  … (\d+) more changed lines", lines[n])) else 1 for n in cut)
    out = [line for n, line in enumerate(lines) if n not in cut]
    out.insert(diff[PASSED_STEP_DIFF_LINES], f"  … {hidden} more changed lines not shown (every check in this "
                                             "step passed; add an expect to verify something else here)")
    return out


def _layout(items: list, viewport: dict, limit: int = MAX_LAYOUT_LABELS) -> str:
    """Where the visible labels are: enough to check what is drawn left of what, and what is too small."""
    if not items:
        return ""
    graph = [i for i in items if i[4]]
    chosen = (graph + [i for i in items if not i[4]])[:limit]
    chosen.sort(key=lambda i: (round(i[2] / 12), i[1]))
    size = f"{viewport.get('width')}x{viewport.get('height')}" if viewport else "the"
    rows = [f"- {text!r} x={x} y={y} h={h}{' [graph]' if in_graph else ''}" for text, x, y, h, in_graph in chosen]
    notes = []
    tiny = [i for i in graph if i[3] < MIN_READABLE_PX]
    if tiny:
        notes.append(f"{len(tiny)} of {len(graph)} graph labels are under {MIN_READABLE_PX}px tall (unreadable).")
    if len(items) > len(chosen):
        notes.append(f"{len(items) - len(chosen)} more labels not listed.")
    return (f"layout (visible labels in a {size} viewport; x,y = centre in px, h = height):\n"
            + "\n".join(rows) + ("\n" + " ".join(notes) if notes else ""))


def _format(report: dict) -> str:
    problems = []
    if report.get("login_page"):
        problems.append("the browser was sent to the login page")
    if report.get("not_found"):
        problems.append("Frappe reports the page as not found (register the Page, or check the route)")
    if report.get("status") and report["status"] >= 400:
        problems.append(f"HTTP {report['status']}")
    if report.get("console"):
        problems.append(f"{len(report['console'])} console/page error(s)")
    if report.get("api_failures"):
        problems.append(f"{len(report['api_failures'])} failed API call(s)")
    failed_steps = [s for s in report.get("steps", []) if s["outcome"] != "ok"]
    if failed_steps:
        problems.append(f"{len(failed_steps)} step(s) failed")
    failed_expects = sum(1 for s in report.get("steps", []) for line in s.get("lines", [])
                         if line.startswith("  expect ") and ": FAIL" in line)
    if failed_expects:
        problems.append(f"{failed_expects} expect(s) failed")
    if report.get("registration_error"):
        problems.insert(0, "the Page JSON could not be registered: " + report["registration_error"])
    if report.get("runner_error"):
        problems.append(report["runner_error"])
    lines = [("RENDER_FAILED: " + "; ".join(problems)) if problems else "RENDER_OK",
             f"url: {report.get('url')} (HTTP {report.get('status')})"]
    if report.get("registered"):
        lines.append("registered the Page from its JSON before loading (as bench migrate would)")
    if report.get("unregistered"):
        lines.append(f"not registered: {report['unregistered']} was not imported, since this phase is read-only; "
                     "the site shows the Page as last registered (or not found if it never was). "
                     "Implementation registers it before its checks.")
    if report.get("load"):
        lines += ["first two seconds after load:", *report["load"]]
    steps = report.get("steps", [])
    for step in steps:
        lines += _step_lines(step["lines"], step["outcome"])
        layout = _layout(step.get("layout") or [], step.get("viewport") or {})
        if layout:
            lines += ["  " + line for line in layout.splitlines()]
    if report.get("console"):
        lines.append("console errors:\n" + "\n".join(f"- {e}" for e in report["console"]))
    if report.get("api_failures"):
        lines.append("failed API calls:\n" + "\n".join(f"- {e}" for e in report["api_failures"]))
    lines.append("page after the last step (accessibility snapshot, with open dialogs and alerts):\n"
                 + _bounded(str(report.get("snapshot") or ""), STEPS_SNAPSHOT_CHARS if steps else MAX_SNAPSHOT_CHARS))
    layout = _layout(report.get("layout") or [], report.get("viewport") or {},
                     STEPS_LAYOUT_LABELS if steps else MAX_LAYOUT_LABELS)
    if layout:
        lines.append(layout)
    if report.get("reach"):
        lines.append(report["reach"])
    if report.get("screenshot"):
        lines.append(f"screenshot for the user: {report['screenshot']}")
    return "\n".join(lines)


def check_page(app_name: str, route: str, steps: list | None = None, env: dict | None = None, *,
               register: bool = True) -> str:
    """``register=False`` (planning, review) never imports Page JSON into the live site or commits."""
    route = "/" + str(route or "").strip().lstrip("/")
    if not re.fullmatch(r"/[A-Za-z0-9_\-./?=&%]*", route) or route.startswith("//"):
        return "RENDER_FAILED: route must be a site path such as /desk/my_page."
    if steps is not None and not isinstance(steps, list):
        return "RENDER_FAILED: steps must be a JSON list of step objects."
    steps = list(steps or [])[:MAX_STEPS]
    root = Path(_app_root(app_name)).resolve()
    command_env = verification._runner_environment(root, env)
    if not command_env.get("KODA_SITE"):
        return "RENDER_UNAVAILABLE: no Frappe site is connected to this worker."
    extra = _playwright_path()
    if extra:
        command_env["PYTHONPATH"] = os.pathsep.join(filter(None, [command_env.get("PYTHONPATH", ""), extra]))
    browsers = _browsers_path()
    if browsers:
        command_env["PLAYWRIGHT_BROWSERS_PATH"] = browsers
    shots = root / RENDER_DIRECTORY
    shots.mkdir(parents=True, exist_ok=True)
    shot = shots / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', route).strip('_') or 'page'}-{time.strftime('%H%M%S')}.png"
    name = re.fullmatch(r"/(?:desk|app)/([A-Za-z0-9_\-]+)/?", route.split("?", 1)[0])
    page_json = ""
    if name:
        slug = name.group(1).replace("-", "_")
        found = sorted(root.glob(f"**/page/{slug}/{slug}.json"))
        page_json = str(found[0]) if found else ""
    unregistered = ""
    if page_json and not register:
        unregistered, page_json = Path(page_json).relative_to(root).as_posix(), ""
    report = None
    try:
        with tempfile.TemporaryDirectory(prefix="koda-render-", ignore_cleanup_errors=True) as scratch:
            # The report goes through a file: _execute keeps only a bounded head and tail of stdout.
            report_file = Path(scratch) / "report.json"
            argv = [sys.executable, str(RUNNER), route, json.dumps(steps, default=str), str(shot), page_json,
                    str(shots / ".session.json"), str(report_file)]
            code, output, timed_out = verification._execute(
                argv, Path(command_env["KODA_SITES_PATH"]),
                {**command_env, "TMPDIR": scratch, "TEMP": scratch, "TMP": scratch}, RENDER_TIMEOUT)
            if report_file.exists():
                try:
                    report = json.loads(report_file.read_text(encoding="utf-8"))
                except ValueError:
                    report = None
            if isinstance(report, dict) and unregistered:
                report["unregistered"] = unregistered
    except (OSError, ValueError) as exc:
        return f"RENDER_UNAVAILABLE: could not start the browser runner: {type(exc).__name__}: {exc}"
    if "KODA_RENDER_UNAVAILABLE" in output:
        return "RENDER_UNAVAILABLE: " + output.split("KODA_RENDER_UNAVAILABLE", 1)[1].strip()[:400]
    if verification.RUNNER_ERROR in output:
        return "RENDER_UNAVAILABLE: " + _bounded(output, 1500) + verification.ENVIRONMENT_NOTE
    if timed_out:
        return f"RENDER_FAILED: the page check did not finish within {RENDER_TIMEOUT}s.\n" + _bounded(output, 2000)
    if report is not None:
        return _format(report)
    marker = output.rfind("KODA_RENDER_REPORT ")
    if marker < 0:
        return f"RENDER_FAILED: the browser runner exited {code} without a report.\n" + _bounded(output, 3000)
    try:
        report = json.loads(output[marker + len("KODA_RENDER_REPORT "):].strip().splitlines()[0])
    except (ValueError, IndexError):
        return "RENDER_FAILED: unreadable browser report.\n" + _bounded(output, 3000)
    if isinstance(report, dict) and unregistered:
        report["unregistered"] = unregistered
    return _format(report)
