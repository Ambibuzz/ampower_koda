"""Co-change memory: which files this repository changes together."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from ..config.schema import HistoryConfig
from ..contracts.session import CoChangeMemory

_HEADER = "\x1f"

GIT_LOG_FORMAT = f"--pretty=format:{_HEADER}%H{_HEADER}%ct"

_SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class Commit:
    """One commit, reduced to what co-change needs."""

    sha: str
    timestamp: float
    paths: tuple[str, ...]


def git_log_arguments(config: HistoryConfig) -> tuple[str, ...]:
    """The exact read-only git invocation this module can parse."""
    return (
        "log",
        f"-n{config.max_commits}",
        GIT_LOG_FORMAT,
        "--name-only",
        "--no-merges",
        "--no-renames",
    )


def parse_git_log(text: str) -> tuple[Commit, ...]:
    """Parse ``git log`` output produced with :data:`GIT_LOG_FORMAT`."""
    commits: list[Commit] = []
    sha = ""
    timestamp = 0.0
    paths: list[str] = []

    def flush() -> None:
        if sha and paths:
            commits.append(Commit(sha=sha, timestamp=timestamp, paths=tuple(sorted(set(paths)))))

    for line in text.splitlines():
        if line.startswith(_HEADER):
            flush()
            _, _, header = line.partition(_HEADER)
            sha, _, stamp = header.partition(_HEADER)
            timestamp = _to_float(stamp)
            paths = []
        elif line.strip():
            paths.append(line.strip().replace("\\", "/"))

    flush()
    return tuple(commits)


def build_cochange(
    commits: Iterable[Commit],
    config: HistoryConfig,
    *,
    now: float,
) -> CoChangeMemory:
    """Accumulate weighted co-change pairs into a neighbour table."""
    weights: dict[str, dict[str, float]] = {}
    counted = 0

    for commit in commits:
        paths = commit.paths
        if not 2 <= len(paths) <= config.max_files_per_commit:
            continue

        counted += 1
        contribution = _recency(commit.timestamp, now, config.half_life_days) / (len(paths) - 1)

        for left, right in combinations(paths, 2):
            _add(weights, left, right, contribution)
            _add(weights, right, left, contribution)

    return CoChangeMemory(
        neighbours={
            path: _rank(neighbours, config.max_neighbours)
            for path, neighbours in weights.items()
        },
        commits_read=counted,
    )


def empty_memory() -> CoChangeMemory:
    """The memory a repository with no usable history has."""
    return CoChangeMemory()


def _recency(timestamp: float, now: float, half_life_days: float) -> float:
    """Exponential decay in ``[0, 1]``, 1.0 for a commit made right now."""
    age_days = max(0.0, (now - timestamp) / _SECONDS_PER_DAY)
    return 0.5 ** (age_days / half_life_days)


def _rank(neighbours: dict[str, float], limit: int) -> tuple[tuple[str, float], ...]:
    """Strongest neighbours first; ties broken by path so the order is total."""
    ordered = sorted(neighbours.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ordered[:limit]) if limit else ()


def _add(weights: dict[str, dict[str, float]], source: str, target: str, amount: float) -> None:
    """Accumulate one directed half of a pair. The caller adds both halves."""
    row = weights.setdefault(source, {})
    row[target] = row.get(target, 0.0) + amount


def _to_float(value: str) -> float:
    try:
        return float(value.strip())
    except ValueError:
        return 0.0
