"""Rollback and revert — as ordinary edits, never as direct writes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .apply import Edit
from .checkpoints import ABSENT, Store

EMPTIED_NOTICE = (
    "revert emptied files it could not delete — their contents are gone but the "
    "files remain. Delete them manually: {paths}"
)

PARTIAL_NOTICE = "(the rollback did not fully apply — check {paths})"


@dataclass(frozen=True, slots=True)
class Unwind:
    """Edits that would restore an earlier state, and what they cannot restore."""

    edits: tuple[Edit, ...] = ()
    emptied: tuple[str, ...] = ()
    """Files the unwind can only blank, because an edit has no deletion form."""

    missing: tuple[str, ...] = ()
    """Files whose recorded content is no longer available — a blob the store
    never held, or a checkpoint that recorded a file it could not read."""

    notices: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.edits


def rollback(store: Store, patch_id: str, current: Mapping[str, str]) -> Unwind:
    """Undo one attempt, using that attempt's own checkpoint as the target."""
    point = store.for_patch(patch_id)
    if point is None:
        return Unwind(notices=(f"no checkpoint for {patch_id}",))
    return _restore(store, {path: point.files[path] for path in point.paths}, current)


def revert(store: Store, to_sequence: int, current: Mapping[str, str]) -> Unwind:
    """Undo everything after ``to_sequence``."""
    targets: dict[str, str] = {}
    for point in store.after(to_sequence):
        for path in point.paths:
            targets.setdefault(path, point.files[path])
    return _restore(store, targets, current)


def _restore(store: Store, targets: Mapping[str, str], current: Mapping[str, str]) -> Unwind:
    """Turn a path → sha map into edits against the current text."""
    edits: list[Edit] = []
    emptied: list[str] = []
    missing: list[str] = []

    for path in sorted(targets):
        sha = targets[path]
        now = current.get(path, "")
        wanted = store.content(sha)

        if not store.holds(sha) and now:
            missing.append(path)
            continue

        if wanted == now:
            continue

        if sha == ABSENT:
            emptied.append(path)

        edits.append(
            Edit(
                path=path,
                old_string=now,
                new_string=wanted,
                reason="restore checkpoint",
            )
        )

    notices: list[str] = []
    if emptied:
        notices.append(EMPTIED_NOTICE.format(paths=", ".join(emptied)))
    if missing:
        notices.append(f"could not restore {', '.join(missing)} — the recorded content is gone")

    return Unwind(
        edits=tuple(edits),
        emptied=tuple(emptied),
        missing=tuple(missing),
        notices=tuple(notices),
    )


def partial_notice(failed: Sequence[str]) -> str:
    """The annotation a partly-applied rollback leaves on its attempt record."""
    return PARTIAL_NOTICE.format(paths=", ".join(failed)) if failed else ""
