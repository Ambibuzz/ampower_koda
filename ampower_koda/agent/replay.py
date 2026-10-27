"""Browser-check replays: where recordings are kept, and the methods the Koda Replay page calls.

Every check_page run during a request is recorded into the target app's .koda/render/recordings/<request>/
(Koda never commits .koda/). The replay page lists those recordings and plays one.
"""

import re
import secrets
import time
from pathlib import Path

import frappe
from frappe import _

from ampower_koda.agent.recording import bundle, cleanup, recorder
from ampower_koda.agent.tools import _app_root

DOCTYPE_NAME = "Agent Request"
RECORDINGS_DIRECTORY = Path(".koda", "render", "recordings")
# <time>-<random part>-<route>, as new_recording_dir names the folder.
RECORDING_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}-[A-Za-z0-9_-]{1,60}$")
REQUEST_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
# Served at run time rather than bundled, so `bench build` works on a bench where node_modules is not installed.
REPLAYER_FILE = Path("node_modules", "@rrweb", "replay", "dist", "replay.umd.min.cjs")


def recordings_root(app_name):
    return Path(_app_root(app_name)).resolve() / RECORDINGS_DIRECTORY


def new_recording_dir(app_name, request_name, route):
    """A fresh folder for one check's recording; the runner fills it. Old recordings are deleted first."""
    if not request_name or not REQUEST_NAME.match(request_name):
        return None
    root = recordings_root(app_name)
    try:
        cleanup.prune(root)
    except OSError:
        # A file in use or a permission problem: the next check tries again; this one still records.
        pass
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", route).strip("_")[:60] or "page"
    # The time comes first because prune sorts folders by name; the random part keeps two checks of the
    # same route in the same second apart.
    folder = root / request_name / f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}-{slug}"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Disk full or not writable: the check still runs, unrecorded.
        return None
    return folder


def has_recording(folder):
    return bool(folder) and (Path(folder) / bundle.RECORDING_FILE).is_file()


def recording_error(folder):
    """Why the runner could not record a check; empty when nothing went wrong."""
    return bundle.read_error(folder)


def _request(request_name):
    if not request_name or not REQUEST_NAME.match(request_name):
        frappe.throw(_("Invalid request name."))
    # No permission check: anyone signed in with the link can watch a replay.
    doc = frappe.get_doc(DOCTYPE_NAME, request_name)
    if not doc.target_app_name:
        frappe.throw(_("This request has no target app, so it has no recordings."))
    return doc


def _recording_dir(doc, recording):
    if not RECORDING_ID.match(recording or ""):
        frappe.throw(_("Invalid recording id."))
    folder = recordings_root(doc.target_app_name) / doc.name / recording
    if not has_recording(folder):
        frappe.throw(_("Recording {0} was not found.").format(recording), frappe.DoesNotExistError)
    return folder


@frappe.whitelist()
def list_recordings(request_name):
    """The request's recorded checks, newest first."""
    doc = _request(request_name)
    folder = recordings_root(doc.target_app_name) / doc.name
    if not folder.is_dir():
        return []
    recordings = []
    for path in sorted(folder.iterdir(), reverse=True):
        if not RECORDING_ID.match(path.name) or not has_recording(path):
            continue
        summary = bundle.read_summary(path) or {}
        steps = summary.get("steps") or []
        recordings.append({
            "id": path.name,
            "route": summary.get("route"),
            "started": summary.get("started"),
            "truncated": bool(summary.get("truncated")),
            "steps": len(steps),
            "failed_steps": sum(1 for step in steps if str(step.get("outcome") or "").startswith("FAILED")),
        })
    return recordings


@frappe.whitelist()
def recording_file(request_name, recording):
    """The gzipped recording; the player unpacks it in the browser."""
    doc = _request(request_name)
    folder = _recording_dir(doc, recording)
    frappe.local.response.filename = bundle.RECORDING_FILE
    frappe.local.response.filecontent = (folder / bundle.RECORDING_FILE).read_bytes()
    frappe.local.response.type = "binary"


@frappe.whitelist()
def replayer_script():
    """rrweb's player for the browser; it sets window.rrwebReplay."""
    path = recorder.node_module_file(REPLAYER_FILE)
    if path is None:
        frappe.throw(_("The replay library @rrweb/replay is not installed; run `bench setup requirements --node`."),
                     frappe.DoesNotExistError)
    frappe.local.response.filename = "rrweb-replay.js"
    frappe.local.response.filecontent = path.read_bytes()
    # "download" sends content_type as is; "binary" would send octet-stream, which a <script> tag refuses.
    frappe.local.response.type = "download"
    frappe.local.response.content_type = "text/javascript"
    frappe.local.response.display_content_as = "inline"
