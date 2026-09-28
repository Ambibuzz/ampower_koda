"""A second reader of the implementation, on its own append-only conversation.

It reads the rounds that changed files, a few at a time, so defects are fixed during
implementation instead of after a full review pass. Its conversation only grows, so each call
reuses a cached prefix.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

MAX_CALLS = 12          # per implementation pass
# Write rounds read per call. Advising every round spent 16 calls on one run, 12 with no note;
# reading three rounds together catches the same defects a little later for a third of the calls.
ADVISE_EVERY = 3
MAX_NOTES = 3
DELTA_CHARS = 12_000    # of one round's changes
BODY_CHARS = 4_000      # of one written body or edit
RESULT_CHARS = 1_500    # of one edit receipt
WRITE_OK = ("WRITE_OK:", "EDIT_OK:", "COPY_OK:", "RENAME_OK:", "DELETE_OK:")
SEVERITIES = ("blocker", "concern")

SYSTEM = """You advise an implementation agent working in a Frappe app. After each round in which it
changed files you see those changes and their automatic check receipts.

Flag only what will make the task fail or leave an acceptance criterion unmet: a wrong field,
DocType or method name; broken client/server wiring (a frappe.call path that does not exist, a
method that is not whitelisted); wrong query logic or data; an edit that breaks code it did not
read; an unhandled error path the criteria cover; a test that patches the code it tests.
Do not restate what it did, suggest style, add scope, or repeat a note you already gave.
Unsure is not a note: say nothing unless you can name the line and the concrete failure.

Reply with JSON only:
{"notes": [{"severity": "blocker|concern", "where": "path:line", "note": "what fails and why"}]}
at most 3 notes; {"notes": []} when nothing is wrong."""


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def _bounded(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more chars)"


def render_round(round_entry: dict) -> str:
    """The files one round changed, with their receipts; "" when it changed none."""
    ai = round_entry["ai"]
    results = {getattr(m, "tool_call_id", None): _text(m.content) for m in round_entry["tools"]}
    blocks = []
    for call in getattr(ai, "tool_calls", None) or []:
        result = results.get(call.get("id"), "")
        if not result.startswith(WRITE_OK):
            continue
        args = call.get("args") or {}
        target = args.get("path") or args.get("destination_path") or ""
        body = {key: value for key, value in args.items() if key not in ("path", "destination_path")}
        blocks.append(f"### {call.get('name')} {target}\n"
                      + _bounded(json.dumps(body, ensure_ascii=False, indent=1), BODY_CHARS)
                      + "\nReceipt:\n" + _bounded(result, RESULT_CHARS))
    if not blocks:
        return ""
    said = _text(ai.content).strip()
    head = f"## ROUND {round_entry.get('number')}\n" + (f"Agent: {_bounded(said, 500)}\n" if said else "")
    return _bounded(head + "\n\n".join(blocks), DELTA_CHARS)


def parse_notes(text: str) -> list[dict]:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        payload = json.loads(match.group(0)) if match else {}
    except ValueError:
        return []
    notes = payload.get("notes") if isinstance(payload, dict) else None
    kept = []
    for item in notes if isinstance(notes, list) else []:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "").lower()
        note = str(item.get("note") or "").strip()
        if severity in SEVERITIES and note:
            kept.append({"severity": severity, "where": str(item.get("where") or "").strip(), "note": note})
    return kept[:MAX_NOTES]


def directive(notes: list[dict]) -> str:
    lines = [f"- [{n['severity']}] {n['where'] + ': ' if n['where'] else ''}{n['note']}" for n in notes]
    return ("## ADVISOR NOTES\nA second reviewer read your last changes. Check each note against the source: "
            "fix a real defect now, and move on from a note you can show is wrong.\n" + "\n".join(lines))


class Advisor:
    def __init__(self, llm, contract: str, *, max_calls: int = MAX_CALLS, every: int = ADVISE_EVERY):
        self.llm = llm
        self.max_calls = max_calls
        self.every = max(1, every)
        self.pending: list[str] = []
        self.calls = 0
        self.messages = [SystemMessage(content=SYSTEM), HumanMessage(content=contract)]
        self.given: set[str] = set()

    def advise(self, round_entry: dict) -> tuple[list[dict], object]:
        """New notes on this round's changes, and the reply (None without a call) for the caller to charge."""
        if self.calls >= self.max_calls:
            return [], None
        delta = render_round(round_entry)
        if not delta:
            return [], None
        self.pending.append(delta)
        if len(self.pending) < self.every:
            return [], None
        self.calls += 1
        messages = [*self.messages, HumanMessage(content="\n\n".join(self.pending))]
        self.pending = []
        reply = self.llm.invoke(messages)
        text = _text(reply.content)
        self.messages = [*messages, AIMessage(content=text)]
        fresh = [note for note in parse_notes(text) if note["note"] not in self.given]
        self.given.update(note["note"] for note in fresh)
        return fresh, reply
