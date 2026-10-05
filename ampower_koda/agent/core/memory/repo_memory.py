"""Reading the instructions a repository writes for its agent."""

from __future__ import annotations

from ..constants import MEMORY_FILENAMES
from ..contracts.session import RepoMemory
from ..errors import WorkspaceError
from ..tokens import estimate_tokens, truncate_to_tokens
from ..workspace.ports import Workspace

MIN_TRUNCATED_TOKENS = 32
"""A cut memory file shorter than this is dropped rather than kept as a stub.
A file that fits whole is kept at any length."""


def read_repo_memory(
    workspace: Workspace,
    *,
    max_tokens: int,
    filenames: tuple[str, ...] = MEMORY_FILENAMES,
) -> RepoMemory:
    """Read the repository's memory files into one budgeted block.

    ``max_tokens`` 0 switches memory off: nothing is read and nothing reported
    as truncated, so "off" and "too small to fit" stay distinguishable.
    """
    if max_tokens <= 0:
        return RepoMemory()

    sections: list[str] = []
    sources: list[str] = []
    truncated = False
    remaining = max_tokens

    for filename in filenames:
        text = _read_text(workspace, filename)
        if text is None or not text.strip():
            continue

        if remaining <= 0:
            truncated = True
            continue

        header = f"# {filename}"
        body = truncate_to_tokens(text.strip(), remaining - estimate_tokens(header) - 1)
        if not body:
            truncated = True
            remaining = 0
            continue

        cut = len(body) < len(text.strip())
        if cut and estimate_tokens(body) < MIN_TRUNCATED_TOKENS:
            # The leftover of an earlier file's budget: a header and a word or
            # two instruct nothing, so the file is dropped, not stubbed.
            truncated = True
            continue

        truncated = truncated or cut
        section = f"{header}\n\n{body}"
        sections.append(section)
        sources.append(filename)
        remaining -= estimate_tokens(section) + 1

    return RepoMemory(
        text="\n\n".join(sections),
        sources=tuple(sources),
        truncated=truncated,
    )


def _read_text(workspace: Workspace, path: str) -> str | None:
    """Read one file as UTF-8, or return ``None`` if it is missing or unreadable."""
    if workspace.stat(path) is None:
        return None
    try:
        return workspace.read_bytes(path).decode("utf-8-sig", errors="replace")
    except WorkspaceError:
        return None
