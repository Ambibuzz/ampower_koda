"""The repository index: every analysed file, and the pure queries over it."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from .analysis import FileAnalysis, SkippedFile
from .chunks import Chunk
from .symbols import Definition


@dataclass(frozen=True, slots=True)
class RepositoryIndex:
    """Every file the cold start analysed, keyed by workspace-relative path."""

    files: Mapping[str, FileAnalysis] = field(default_factory=dict)
    skipped: Mapping[str, SkippedFile] = field(default_factory=dict)

    fingerprint: str = ""
    """The schema fingerprint the entries were produced under. An index built
    under a different fingerprint cannot be merged with this one."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        object.__setattr__(self, "skipped", MappingProxyType(dict(self.skipped)))

    def __len__(self) -> int:
        return len(self.files)

    def __contains__(self, path: object) -> bool:
        return path in self.files

    @property
    def paths(self) -> tuple[str, ...]:
        """Analysed paths in codepoint order."""
        return tuple(sorted(self.files))


def with_file(index: RepositoryIndex, analysis: FileAnalysis) -> RepositoryIndex:
    """Return a new index with ``analysis`` replacing any entry for its path."""
    files = dict(index.files)
    files[analysis.path] = analysis
    skipped = {path: entry for path, entry in index.skipped.items() if path != analysis.path}
    return replace(index, files=files, skipped=skipped)


def with_skip(index: RepositoryIndex, skip: SkippedFile) -> RepositoryIndex:
    """Return a new index recording ``skip`` and dropping any analysis for it."""
    files = {path: entry for path, entry in index.files.items() if path != skip.path}
    skipped = dict(index.skipped)
    skipped[skip.path] = skip
    return replace(index, files=files, skipped=skipped)


def without_file(index: RepositoryIndex, path: str) -> RepositoryIndex:
    """Return a new index with ``path`` removed entirely — analysed or skipped."""
    if path not in index.files and path not in index.skipped:
        return index
    files = {key: entry for key, entry in index.files.items() if key != path}
    skipped = {key: entry for key, entry in index.skipped.items() if key != path}
    return replace(index, files=files, skipped=skipped)


def iter_chunks(index: RepositoryIndex) -> Iterator[Chunk]:
    """Yield every chunk in the index, files in codepoint order."""
    for path in index.paths:
        yield from index.files[path].chunks


def iter_indexable_chunks(index: RepositoryIndex) -> Iterator[Chunk]:
    """Yield only chunks that contribute to corpus-wide statistics."""
    for chunk in iter_chunks(index):
        if chunk.indexable:
            yield chunk


def iter_definitions(index: RepositoryIndex) -> Iterator[tuple[str, Definition]]:
    """Yield ``(path, definition)`` for every definition, in path order."""
    for path in index.paths:
        for definition in index.files[path].definitions:
            yield path, definition


def definitions_by_name(index: RepositoryIndex) -> Mapping[str, tuple[tuple[str, Definition], ...]]:
    """Map every bare *and* qualified name to the definitions that carry it."""
    table: dict[str, list[tuple[str, Definition]]] = {}
    for path, definition in iter_definitions(index):
        entry = (path, definition)
        table.setdefault(definition.name, []).append(entry)
        qualified = definition.qualified_name
        if qualified != definition.name:
            table.setdefault(qualified, []).append(entry)
    return MappingProxyType({name: tuple(entries) for name, entries in table.items()})


def files_referencing(index: RepositoryIndex, name: str) -> tuple[str, ...]:
    """Return paths whose references mention ``name``, in codepoint order."""
    return tuple(
        path
        for path in index.paths
        if any(reference.name == name for reference in index.files[path].references)
    )


def chunk_by_digest(index: RepositoryIndex, digest: str) -> Chunk | None:
    """Resolve a chunk by its content hash."""
    for chunk in iter_chunks(index):
        if chunk.digest == digest:
            return chunk
    return None
