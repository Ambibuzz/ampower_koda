"""JSON execution checkpoints with revision-checked resume and write intents.

These are private request fields, not repository files. Credentials and model
objects are never serialized. A write intent is committed before mutation, so a
worker lost after a successful write can reconcile either observed outcome.
"""
import copy
import json
from pathlib import Path

import frappe

from .execution_contract import revision
from .run_control import check_active, current_run, set_request_value

#: Checkpoint format; a checkpoint of another version is refused rather than resumed wrongly.
VERSION = 2
# Message objects are never serialized; the conversation lives in the session file.
_PRIVATE_KEYS = {"github_token", "messages", "review_history"}


class ExecutionJournal:
    def __init__(self, request_name, read_current, *, branch, head, state=None):
        self.request_name = request_name
        self.read_current = read_current
        self.branch = branch
        self.head = head
        self.state = dict(state or {})
        self.node = "prepare"
        self.revisions = {}
        self.pending = None
        self.temporary_files = []

    def paths(self):
        return set(self.state.get("execution_baseline") or {}) | {
            p for t in (self.state.get("plan_object") or {}).get("tasks", []) for p in t["files"]
        } | set(self.state.get("prior_changed_paths") or [])

    def payload(self):
        return {"version": VERSION, "branch": self.branch, "head": self.head, "node": self.node,
                "state": {k: v for k, v in self.state.items() if k not in _PRIVATE_KEYS and not k.startswith("_")},
                "revisions": self.revisions, "pending": self.pending, "temporary_files": self.temporary_files}

    def save(self):
        set_request_value(self.request_name, "execution_checkpoint", json.dumps(self.payload(), ensure_ascii=True))
        frappe.db.commit()
        check_active()

    def begin(self, state, node):
        self.state = copy.deepcopy({k: v for k, v in state.items() if k not in _PRIVATE_KEYS})
        self.node = node
        self.pending = None
        self.refresh()
        self.save()

    def refresh(self):
        self.revisions = {p: revision(self.read_current(p)) for p in self.paths()}

    def update(self, **updates):
        self.state.update(copy.deepcopy(updates))
        # Token/call accounting must not re-read every approved file. Existing
        # revisions change only at a node boundary or a recorded filesystem write.
        for path in self.paths() - self.revisions.keys():
            self.revisions[path] = revision(self.read_current(path))
        self.save()

    def intent(self, before, after, *, move=None):
        self.pending = {"before": {p: revision(c) for p, c in before.items()},
                        "after": {p: revision(c) for p, c in after.items()}, "move": move}
        self.save()

    def finish_write(self, file_moves=None):
        if file_moves is not None:
            self.state["file_moves"] = copy.deepcopy(file_moves)
        for path in (self.pending or {}).get("after", {}):
            self.revisions[path] = revision(self.read_current(path))
        if self.temporary_files:
            from .tools import _resolve_path
            cleanup_temporaries(self.temporary_files,
                                lambda p: _resolve_path(self.state["target_app_name"], p))
        self.pending = None
        self.temporary_files = []
        self.save()


def journal():
    run = current_run()
    return run.journal if run else None


def update(**updates):
    active = journal()
    if active:
        active.update(**updates)


def write_intent(before, after, *, move=None):
    active = journal()
    if active:
        active.intent(before, after, move=move)


def temporary_file(path, destination):
    active = journal()
    if active:
        active.temporary_files.append({"path": path, "destination": destination})
        active.save()


def cleanup_temporaries(entries, resolve_path):
    """Remove only this checkpoint's named sibling temporary files after validation."""
    checked = []
    for entry in entries:
        temporary = Path(entry["path"])
        destination = Path(resolve_path(entry["destination"]))
        if (temporary.is_symlink() or temporary.resolve().parent != destination.resolve().parent
                or not temporary.name.startswith(".koda-write-")):
            raise ValueError("Invalid checkpoint temporary path; no cleanup performed.")
        checked.append(temporary)
    for path in checked:
        if path.exists():
            path.chmod(0o600)
        path.unlink(missing_ok=True)


def restore_checkpoint(value, *, plan, branch, head, read_current):
    payload = json.loads(value) if isinstance(value, str) else copy.deepcopy(value)
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("No compatible execution checkpoint is available.")
    if payload.get("branch") != branch or payload.get("head") != head:
        raise ValueError("Branch or HEAD changed since the checkpoint. Resume was refused without changing files.")
    state = payload["state"]
    if state.get("plan_object") != plan:
        raise ValueError("Approved plan changed since the checkpoint. Review the plan before starting a new run.")
    pending = payload.get("pending") or {}
    revisions = dict(payload.get("revisions") or {})
    revisions.update(pending.get("before") or {})
    for path, expected in revisions.items():
        current = revision(read_current(path))
        allowed = {expected}
        if path in (pending.get("after") or {}):
            allowed.add(pending["after"][path])
        if current not in allowed:
            raise ValueError(f"{path} changed outside the saved execution. Resume was refused without changing files.")
    move = pending.get("move")
    if move and read_current(move["source"]) is None and revision(read_current(move["destination"])) == move["sha256"]:
        moves = list(state.get("file_moves") or [])
        if move not in moves:
            moves.append(move)
        state["file_moves"] = moves
    node = payload.get("node")
    if node not in {"prepare", "implement", "review", "done"}:
        raise ValueError("Invalid checkpoint node.")
    if node in {"review", "implement"} and state.get("error") and state.get("review_repairable"):
        # Send the saved finding back to implementation; attempt history is kept.
        node = "implement"
        state["review_repairable"] = False
    state.pop("error", None)
    state.pop("error_log", None)
    if state.get("current_stage") == "Reviewing":
        state["turn_exhausted"] = False
    state["resume_node"] = node
    state["resuming"] = True
    state["_checkpoint_temporaries"] = payload.get("temporary_files") or []
    return state
