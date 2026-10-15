"""Record a check_page run so the user can replay it: the page as rrweb events, and the check's steps.

Used by render_runner when KODA_RECORD_DIR is set. Nothing here calls a model.
"""

import time
from pathlib import Path

from . import bundle

HERE = Path(__file__).parent
RRWEB_FILE = Path("node_modules", "@rrweb", "record", "dist", "record.umd.min.cjs")
RRWEB_MISSING = ("the recorder library @rrweb/record is not installed; run `bench setup requirements --node` "
                 "(the check itself is not affected)")

# Enough of a step's report to say what the step did; the agent got the full report already.
STEP_LINES_KEPT = 30


class RecordingUnavailable(Exception):
    pass


def node_module_file(relative):
    """A file from the app's node_modules, found the way node would: here or in a parent folder."""
    for folder in (HERE, *HERE.parents):
        candidate = folder / relative
        if candidate.is_file():
            return candidate
    return None


def init_script():
    """rrweb, then capture.js: the script Playwright runs in every page before the page's own scripts."""
    library = node_module_file(RRWEB_FILE)
    if library is None:
        raise RecordingUnavailable(RRWEB_MISSING)
    rrweb = library.read_text(encoding="utf-8")
    capture = (HERE / "capture.js").read_text(encoding="utf-8")
    # Separated by ";" so the two scripts cannot run into each other.
    return rrweb + "\n;\n" + capture


class Recorder:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.page = None
        self.events = []
        self.steps = []

    def attach(self, context, page):
        """Call right after sign-in, so the login redirect is not part of the recording."""
        context.add_init_script(init_script())
        self.page = page

    def drain(self):
        """Collect what capture.js recorded since the last call: after the load and after every step."""
        try:
            collected = self.page.evaluate("window.__kodaRec ? window.__kodaRec.drain() : null")
        except Exception:
            return
        if collected:
            self.events += collected["events"]

    def stamp(self, name, payload):
        """Put a named moment into the recording, so the replay timeline can show it."""
        try:
            self.page.evaluate("([name, payload]) => window.__kodaRec && window.__kodaRec.stamp(name, payload)",
                               [name, payload])
        except Exception:
            pass

    def step_start(self, index, action):
        self.stamp("koda-step", {"index": index, "action": action})
        self.steps.append({"index": index, "action": action})

    def step_end(self, report):
        step = self.steps[-1]
        step["outcome"] = report.get("outcome")
        step["lines"] = report.get("lines", [])[:STEP_LINES_KEPT]
        self.stamp("koda-step-end", {"index": step["index"], "outcome": step["outcome"]})
        self.drain()

    def finish(self, route):
        """Write recording.json.gz, and summary.json for the list of recordings."""
        self.drain()
        record = {
            "version": 1,
            "meta": {"route": route, "started": self.started},
            "events": self.events,
            "steps": self.steps,
        }
        bundle.write(self.out_dir, record)
        bundle.write_summary(self.out_dir, {
            "route": route,
            "started": self.started,
            "steps": [{"index": step["index"], "outcome": step.get("outcome")} for step in self.steps],
        })
