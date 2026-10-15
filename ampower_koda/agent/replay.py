"""Browser-check recordings: where they are kept for each request.

Every check_page run during a request is recorded into the target app's .koda/render/recordings/<request>/
(Koda never commits .koda/).
"""

import re
import time
from pathlib import Path

from ampower_koda.agent.recording import bundle, cleanup
from ampower_koda.agent.tools import _app_root

RECORDINGS_DIRECTORY = Path(".koda", "render", "recordings")
REQUEST_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


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
    folder = root / request_name / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
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
