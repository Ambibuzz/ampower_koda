"""Open a Desk page in a headless browser and report what the user would see.

The model cannot otherwise see the pages it builds: a client script that never loads, a route that
is not found, a control whose handler throws, or a whitelisted method that fails only when the page
calls it. Each check runs in its own process (the worker never imports Playwright), signs in as
Administrator with a one-time login key, and can drive the page with a short list of steps.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
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


