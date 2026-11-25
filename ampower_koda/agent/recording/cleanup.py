"""Delete old check recordings, so .koda/render/recordings does not grow without end.

Runs each time a new recording starts. A recording is kept while it is one of its request's newest
KEEP_PER_REQUEST and younger than MAX_AGE_DAYS.
"""

import shutil
import time
from pathlib import Path

from . import bundle

KEEP_PER_REQUEST = 20
MAX_AGE_DAYS = 14
# A folder without a finished recording may belong to a check that is still running.
UNFINISHED_GRACE_SECONDS = 3600


def _age_seconds(path, now):
    return now - path.stat().st_mtime


def prune(root, keep_per_request=KEEP_PER_REQUEST, max_age_days=MAX_AGE_DAYS, now=None):
    """Delete what is past its keep; returns how many recordings were deleted."""
    root = Path(root)
    if not root.is_dir():
        return 0
    now = time.time() if now is None else now
    max_age_seconds = max_age_days * 86400
    deleted = 0

    for request_folder in root.iterdir():
        if not request_folder.is_dir():
            continue
        # Folder names start with the time they were made, so the newest sort first.
        recordings = sorted((folder for folder in request_folder.iterdir() if folder.is_dir()), reverse=True)
        finished = 0
        for folder in recordings:
            recording_file = folder / bundle.RECORDING_FILE
            if not recording_file.is_file():
                if _age_seconds(folder, now) > UNFINISHED_GRACE_SECONDS:
                    shutil.rmtree(folder, ignore_errors=True)
                continue
            finished += 1
            too_many = finished > keep_per_request
            too_old = _age_seconds(recording_file, now) > max_age_seconds
            if too_many or too_old:
                shutil.rmtree(folder, ignore_errors=True)
                deleted += 1
        if not any(request_folder.iterdir()):
            request_folder.rmdir()
    return deleted
