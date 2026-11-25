"""Where a recording's files live, and saving and reading them.

Recordings are kept as <recordings>/<request>/<recording>/recording.json.gz, with a small summary.json beside
it.
"""

import gzip
import json
from pathlib import Path

RECORDING_FILE = "recording.json.gz"
# What the list of recordings shows, readable without unpacking the recording.
SUMMARY_FILE = "summary.json"
# Why a check was not recorded, for the worker to show the user. render_runner.write_unrecorded writes it.
ERROR_FILE = "error.txt"


def write(out_dir, record):
    text = json.dumps(record, separators=(",", ":"))
    (Path(out_dir) / RECORDING_FILE).write_bytes(gzip.compress(text.encode("utf-8"), 9))


def load(out_dir):
    return json.loads(gzip.decompress((Path(out_dir) / RECORDING_FILE).read_bytes()))


def write_summary(out_dir, summary):
    (Path(out_dir) / SUMMARY_FILE).write_text(json.dumps(summary, default=str), encoding="utf-8")


def read_summary(out_dir):
    try:
        return json.loads((Path(out_dir) / SUMMARY_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_error(out_dir):
    try:
        return (Path(out_dir) / ERROR_FILE).read_text(encoding="utf-8")
    except OSError:
        return ""
