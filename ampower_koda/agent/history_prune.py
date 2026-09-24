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


def prune_rounds(rounds: list[dict], write_tools, *, reasoning: bool = True,
                 tail_chars: int | None = None) -> int:
    """Retire what ``rounds`` no longer need; return the characters removed.

    Rounds are ``{"number", "ai", "tools", ...}`` entries, oldest first. The
    newest round keeps its reasoning (it may continue the current chain) and
    any failed write body (the next reply may correct it). ``reasoning=False``
    keeps all reasoning, for mid-turn batches. ``tail_chars`` touches
    only messages with at most that much history after them, since a rewrite uncaches what follows.
    """
    return _prune(rounds, write_tools, dry_run=False, reasoning=reasoning, tail_chars=tail_chars)[0]


def prune_price(rounds: list[dict], write_tools, *, reasoning: bool = True, tail_chars: int | None = None,
                followups=()) -> tuple[int, int]:
    """``(removed, rewritten)`` for a prune; ``rewritten`` counts every message from the first change
    on, including later follow-ups, since all of it is resent uncached.
    """
    freed, first, sizes, after_round = _prune(rounds, write_tools, dry_run=True, reasoning=reasoning,
                                              tail_chars=tail_chars)
    if first is None:
        return freed, 0
    r_index, slot = first
    ai_size, tool_sizes = sizes[r_index]
    suffix = (ai_size if slot == 0 else 0) + sum(tool_sizes[max(0, slot - 1):]) + after_round[r_index]
    later = {entry["number"] for entry in rounds[r_index:]}
    suffix += sum(_message_chars(m) for f in followups if f.get("after") in later for m in f.get("messages", ()))
    return freed, max(0, suffix - freed)


def _prune(rounds: list[dict], write_tools, *, dry_run: bool, reasoning: bool, tail_chars: int | None):
    """``prune_rounds`` with what ``prune_price`` needs: (freed, first change, sizes, chars after each round)."""
    sizes = [[_message_chars(entry["ai"]), [_message_chars(m) for m in entry["tools"]]] for entry in rounds]
    after_round, running = [0] * len(rounds), 0  # chars of every round after this one
    for r_index in range(len(rounds) - 1, -1, -1):
        after_round[r_index] = running
        running += sizes[r_index][0] + sum(sizes[r_index][1])

    def near_end(r_index: int, t_index: int | None = None) -> bool:
        """Whether a rewrite of this round's AI message (t_index None) or tool result is cheap."""
        if tail_chars is None:
            return True
        tools = sizes[r_index][1]
        after = sum(tools if t_index is None else tools[t_index + 1:])
        return after_round[r_index] + after <= tail_chars

    entries, writes, views, copies, touches, order = [], [], [], [], [], 0
    for r_index, entry in enumerate(rounds):
        calls = {c.get("id"): c for c in getattr(entry["ai"], "tool_calls", None) or []}
        for t_index, message in enumerate(entry["tools"]):
            call = calls.get(getattr(message, "tool_call_id", None)) or {}
            name, args = call.get("name", ""), call.get("args") or {}
            content = str(message.content)
            succeeded = write_succeeded(content)
            entries.append((order, r_index, t_index, name, args, message, call.get("id"), succeeded))
            if name in write_tools and succeeded:
                written = _written_paths(name, args)
                writes.extend((order, path) for path in written)
                touches.extend((order, path) for path in written)
                if name == "write_file" and written:
                    views.append((order, written[0], FULL, "rewritten whole by write_file"))
                if name == "copy_file" and args.get("source_path") and written:
                    copies.append((order, _clean(args["source_path"]), written[0]))
            elif name in READ_TOOLS and args.get("path"):
                path = _clean(args["path"])
                touches.append((order, path))
                span = _read_span(content) if name == "read_file" else ""
                if span != "":
                    views.append((order, path, span, "read again"))
            order += 1

    def newer_view(position: int, path: str, span):
        """Why a later copy of ``path`` makes a view at ``position`` redundant, or ""."""
        for at, other, other_span, why in views:
            if at > position and other == path and _contains(other_span, span):
                return why
        return ""

    def copied_away(position: int, read_path: str) -> str:
        """The copy of ``read_path`` the model has since worked on, if any.

        A template read before ``copy_file`` duplicated it is the same bytes as
        the copy. Once the copy has been read or edited, the model works from
        that, and the template read is 20k+ tokens re-sent every call (12.8% of
        one run). Until then it is the model's only view of the copy's lines.
        """
        for at, source, destination in copies:
            if at > position and source == _clean(read_path) and any(
                    when > at and _covers(path, destination) for when, path in touches):
                return destination
        return ""

    freed = 0
    first = None  # (round index, slot) of the earliest change; slot 0 is the AI message, 1 + t tool t
    elide = {}  # (r_index, call id) -> note
    newest = len(rounds) - 1

    def changed(r_index: int, slot: int) -> None:
        nonlocal first
        if first is None or (r_index, slot) < first:
            first = (r_index, slot)

    for position, r_index, t_index, name, args, message, call_id, succeeded in entries:
        content = str(message.content)
        if name in write_tools:
            path = _clean(args.get("path"))
            if not near_end(r_index):
                continue
            if not succeeded:
                if r_index != newest:
                    elide[(r_index, call_id)] = "this write was not applied"
            elif name in ("write_file", "edit_file") and path and newer_view(position, path, FULL):
                elide[(r_index, call_id)] = "applied; a newer copy of this file is in the conversation"
            continue
        if name not in READ_TOOLS or content.startswith(SUPERSEDED) or not near_end(r_index, t_index):
            continue
        read_path = _clean(args.get("path"))
        where = f"{name}({describe_call(name, args)}) from round {rounds[r_index]['number']}"
        if name == "read_file":
            span = _read_span(content)
            why = newer_view(position, read_path, span) if read_path and span != "" else ""
            if why:
                stub = f"{SUPERSEDED} {where}: {read_path} was {why} later in this conversation; use that copy."
            else:
                copy = copied_away(position, read_path) if read_path else ""
                if not copy:
                    continue
                stub = (f"{SUPERSEDED} {where}: this source was copied to {copy}, which you have worked on "
                        f"since. Read {read_path} again only if you need the original.")
        else:
            later = next((path for at, path in writes if at > position and _covers(read_path, path)), None)
            if later is None or (name == "search_code" and not _LINE_REFERENCE.search(content)):
                continue  # a file list without line numbers stays valid after an edit
            stub = (f"{SUPERSEDED} {where}: {later} was edited afterwards, so these line numbers are out of "
                    "date. Search or outline again if you still need it.")
        if len(stub) >= len(content):
            continue
        freed += len(content) - len(stub)
        changed(r_index, 1 + t_index)
        if not dry_run:
            rounds[r_index]["tools"][t_index] = message.model_copy(update={"content": stub})

    for r_index, entry in enumerate(rounds):
        notes = {call_id: note for (index, call_id), note in elide.items() if index == r_index}
        ai = _elide_calls(entry["ai"], notes)
        if reasoning and r_index != newest and near_end(r_index):
            ai = strip_reasoning(ai)
        if ai is not entry["ai"]:
            freed += _message_chars(entry["ai"]) - _message_chars(ai)
            changed(r_index, 0)
            if not dry_run:
                entry["ai"] = ai
    return freed, first, sizes, after_round


def drop_followups(history: dict, kind: str) -> int:
    """Remove every queued follow-up of ``kind``; returns how many were removed."""
    followups = history.get("followups") or []
    kept = [f for f in followups if f.get("kind") != kind]
    history["followups"] = kept
    return len(followups) - len(kept)
