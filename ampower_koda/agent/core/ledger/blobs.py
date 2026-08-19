"""Git blob shas, and the rule that the core computes them itself."""

from __future__ import annotations

import hashlib


def blob_sha(content: bytes) -> str:
    """Return the git blob sha1 of ``content``."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def short(sha: str, length: int = 8) -> str:
    """The display form. Empty stays empty rather than becoming ``'        '``."""
    return sha[:length] if sha else ""
