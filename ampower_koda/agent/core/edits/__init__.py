"""§16 — the write path: locate, dry-apply, checkpoint, describe."""

from __future__ import annotations

from .apply import AnchorCheck, ChangedRegion, DryRun, Edit, Patch, changed_regions, dry_apply
from .checkpoints import ABSENT, Checkpoint, Store, blob_id, capture_before, snapshot
from .locate import Located, Match, dominant_eol, locate
from .unwind import EMPTIED_NOTICE, Unwind, partial_notice, revert, rollback

__all__ = [
    "ABSENT",
    "EMPTIED_NOTICE",
    "AnchorCheck",
    "ChangedRegion",
    "Checkpoint",
    "DryRun",
    "Edit",
    "Located",
    "Match",
    "Patch",
    "Store",
    "Unwind",
    "blob_id",
    "capture_before",
    "changed_regions",
    "dominant_eol",
    "dry_apply",
    "locate",
    "partial_notice",
    "revert",
    "rollback",
    "snapshot",
]
