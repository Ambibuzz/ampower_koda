"""Reading the instructions a repository writes for its agent."""

from __future__ import annotations

from ..constants import MEMORY_FILENAMES
from ..contracts.session import RepoMemory
from ..errors import WorkspaceError
from ..tokens import estimate_tokens, truncate_to_tokens
from ..workspace.ports import Workspace


def read_repo_memory(
    workspace: Workspace,
    *,
    max_tokens: int,
    filenames: tuple[str, ...] = MEMORY_FILENAMES,
) -> RepoMemory:
    """Read the repository's memory files into one budgeted block."""
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

        truncated = truncated or len(body) < len(text.strip())
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
