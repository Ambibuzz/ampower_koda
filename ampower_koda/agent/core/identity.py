"""Content-addressed identity for everything the index holds."""

from __future__ import annotations

import hashlib

from .constants import ANCHOR_DIGEST

_SEPARATOR = "\0"


def _digest(*parts: object) -> str:
    """Return the full hex sha256 of ``parts`` joined unambiguously."""
    payload = _SEPARATOR.join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint(*parts: object, length: int = 16) -> str:
    """Return a short digest over an arbitrary tuple of values."""
    return _digest(*parts)[:length]


def content_hash(
    path: str,
    start_line: int,
    end_line: int,
    identity: str,
    body: str,
) -> str:
    """Return the full-length hash naming one chunk of one file."""
    return _digest(path, start_line, end_line, identity, body)


def anchor_id(path: str, snippet: str) -> str:
    """Return the short id of a content anchor."""
    return _digest(path, snippet)[:ANCHOR_DIGEST]


def source_hash(content: bytes) -> str:
    """Return the hash of a whole file's bytes."""
    return hashlib.sha256(content).hexdigest()


def cache_key(relative_path: str) -> str:
    """Return the on-disk filename stem for one file's cache entry."""
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
