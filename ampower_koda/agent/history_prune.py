"""Retire retained tool history the model can no longer use.

A tool loop keeps every round so later rounds can build on earlier reads. The
model's view of a file is the newest full copy it has seen (a whole-file read
or a body it wrote) plus the edits it made since, each returned with its edited
region. That view is kept: editing by text anchor needs no fresh read, and
retiring it after every edit only forced the file to be read again.

What does stop being useful:

- An older copy of a file the model has since read again or rewritten whole.
- The body of a write or edit that such a newer copy already contains.
- An outline or search of a path written afterwards: its line numbers moved.
- A template read, once the copy made from it has been worked on.
- A failed write's body, after the reply that could correct it.
- Encrypted reasoning replayed with every earlier assistant turn.

Replacing any of these is not free: a changed message invalidates the prompt
cache from that point on. So they are retired in batches: at a turn boundary,
or mid-turn only near the end of the history, where the rewrite is cheap.

Reasoning is the exception: it is retired only at a turn boundary or a full
context. Its encrypted payload is large in characters but small in tokens, so
it tripped the mid-turn batch 19 times in one run, and each strip rewrote the
whole cached suffix (54% of that run's cache writes) to save ~2k tokens.
"""

from __future__ import annotations

import json
import re

READ_TOOLS = frozenset({"read_file", "get_file_outline", "search_code"})
WRITE_OK_PREFIXES = ("WRITE_OK:", "EDIT_OK:", "COPY_OK:", "RENAME_OK:", "DELETE_OK:")
SUPERSEDED = "[superseded]"
# Shorter bodies stay: they cost little and show the model exactly what it did.
PAYLOAD_MIN_CHARS = 300
REASONING_KEYS = ("reasoning_details", "reasoning")
FULL = None  # a view of the whole file
_RANGE_HEADER = re.compile(r"^\[[^\]\n]*\] lines (\d+)-(\d+) of \d+")
_LINE_REFERENCE = re.compile(r"^\S+:\d+", re.M)  # search_code content mode: "path:line" headers
_WHOLE_HEADER = re.compile(r"^\[[^\]\n]*\] \d+ lines")


def write_succeeded(result) -> bool:
    return str(result).startswith(WRITE_OK_PREFIXES)


def _clean(path) -> str:
    return str(path or "").strip().rstrip("/")


def _written_paths(name: str, args: dict) -> list[str]:
    if name == "rename_file":
        paths = [args.get("source_path"), args.get("destination_path")]
    elif name == "copy_file":
        paths = [args.get("destination_path")]
    else:
        paths = [args.get("path")]
    return [_clean(p) for p in paths if p]


def _covers(read_path: str, written: str) -> bool:
    """Whether a write to ``written`` changes what a read of ``read_path`` showed."""
    scope = _clean(read_path)
    return not scope or written == scope or written.startswith(scope + "/")


def _read_span(content: str):
    """FULL for a whole-file read, (start, end) for a range, "" when it showed no source."""
    header = content.split("\n", 1)[0]
    match = _RANGE_HEADER.match(header)
    if match:
        return int(match.group(1)), int(match.group(2))
    return FULL if _WHOLE_HEADER.match(header) else ""


def _contains(newer, older) -> bool:
    if newer is FULL:
        return True
    return older is not FULL and newer[0] <= older[0] and older[1] <= newer[1]


def _elide_args(value, note: str):
    if isinstance(value, str):
        if len(value) <= PAYLOAD_MIN_CHARS:
            return value
        return f"<{len(value)} chars, {value.count(chr(10)) + 1} lines elided: {note}>"
    if isinstance(value, dict):
        return {key: _elide_args(item, note) for key, item in value.items()}
    if isinstance(value, list):
        return [_elide_args(item, note) for item in value]
    return value


def _message_chars(message) -> int:
    size = len(str(getattr(message, "content", "") or ""))
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        size += len(json.dumps(calls, sort_keys=True, default=repr))
    extra = getattr(message, "additional_kwargs", None) or {}
    for key in REASONING_KEYS:
        if key in extra:
            size += len(json.dumps(extra[key], default=repr))
    return size


def _elide_calls(ai, notes: dict):
    """Return ``ai`` with the long arguments of the calls in ``notes`` (id -> note) elided."""
    calls, changed = [], False
    for call in getattr(ai, "tool_calls", None) or []:
        note = notes.get(call.get("id"))
        if note:
            args = _elide_args(call.get("args") or {}, note)
            if args != (call.get("args") or {}):
                call, changed = {**call, "args": args}, True
        calls.append(call)
    if not changed:
        return ai
    # ChatOpenAI prefers ``tool_calls`` over a raw ``additional_kwargs`` copy,
    # but drop the raw copy so the full body cannot be serialized instead.
    kwargs = {k: v for k, v in (ai.additional_kwargs or {}).items() if k != "tool_calls"}
    return ai.model_copy(update={"tool_calls": calls, "additional_kwargs": kwargs})


def strip_reasoning(ai):
    extra = getattr(ai, "additional_kwargs", None) or {}
    if not any(key in extra for key in REASONING_KEYS):
        return ai
    kwargs = {k: v for k, v in extra.items() if k not in REASONING_KEYS}
    return ai.model_copy(update={"additional_kwargs": kwargs})


def describe_call(name: str, args: dict) -> str:
    path = args.get("path") or ""
    if name == "read_file" and args.get("ranges"):
        return f"{path} lines {args['ranges']}"
    if name == "read_file" and (args.get("start_line") or args.get("end_line")):
        return f"{path} lines {args.get('start_line') or 1}-{args.get('end_line') or 'end'}"
    if name == "search_code":
        return f"{str(args.get('pattern', ''))[:60]!r} in {path or 'app'}"
    if name == "call_method":
        return str(args.get("method") or "")
    return path


def drop_followups(history: dict, kind: str) -> int:
    """Remove every queued follow-up of ``kind``; returns how many were removed."""
    followups = history.get("followups") or []
    kept = [f for f in followups if f.get("kind") != kind]
    history["followups"] = kept
    return len(followups) - len(kept)
