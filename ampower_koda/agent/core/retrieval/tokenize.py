"""Turning source code and questions into the same vocabulary."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from types import MappingProxyType

MIN_TOKEN_LENGTH = 2
MAX_TOKEN_LENGTH = 64

_ACRONYM_BOUNDARY = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NON_WORD = re.compile(r"[^0-9A-Za-z]+")

STOP_WORDS: frozenset[str] = frozenset(
    (
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were", "be",
        "been", "being", "for", "on", "at", "by", "with", "from", "as", "it", "its", "this",
        "that", "these", "those", "then", "than", "so", "such",
    )
)

DISCOURSE_WORDS: frozenset[str] = frozenset(
    (
        "how", "what", "where", "when", "why", "who", "which", "does", "do", "did", "done",
        "can", "could", "should", "would", "will", "shall", "may", "might", "must", "have",
        "has", "had", "please", "help", "need", "want", "using", "use", "used", "make",
        "made", "get", "got", "give", "given", "show", "tell", "find", "look", "looking",
        "see", "seeing", "know", "think", "about", "into", "onto", "over", "under",
        "between", "during", "through", "above", "below", "again", "further", "here",
        "there", "all", "both", "few", "other", "some", "own", "too", "very", "just", "now",
        "also", "like", "trying", "try", "tried", "working", "work", "works", "issue",
        "problem", "bug", "error", "something", "anything", "everything", "nothing",
        "someone", "anyone", "everyone",
    )
)

RESCUED_WORDS: frozenset[str] = frozenset(
    (
        "after", "all", "always", "any", "before", "each", "every", "least", "less", "more",
        "most", "never", "no", "not", "only", "same", "until", "without",
    )
)

CONCEPT_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "auth": ("auth", "authenticate", "authentication", "login", "logout", "signin",
                 "credential", "credentials", "token", "session", "password", "permission",
                 "permissions", "role", "roles"),
        "config": ("config", "configuration", "configure", "setting", "settings", "option",
                   "options", "preference", "preferences", "env", "environment"),
        "error": ("error", "errors", "exception", "exceptions", "fail", "failed", "failure",
                  "raise", "raised", "throw", "thrown", "traceback", "crash", "bug"),
        "test": ("test", "tests", "testing", "spec", "specs", "fixture", "fixtures",
                 "assert", "assertion", "mock", "stub"),
        "database": ("database", "db", "sql", "query", "queries", "table", "tables", "column",
                     "columns", "row", "rows", "migration", "migrate", "schema", "index",
                     "doctype", "docfield", "record", "records"),
        "http": ("http", "https", "request", "requests", "response", "responses", "api",
                 "endpoint", "endpoints", "route", "routes", "url", "rest", "webhook",
                 "whitelist", "whitelisted"),
        "cache": ("cache", "cached", "caching", "invalidate", "invalidation", "memo",
                  "memoize", "memoized", "stale", "ttl", "expire", "expiry"),
        "log": ("log", "logs", "logger", "logging", "trace", "tracing", "debug", "audit",
                "monitor", "monitoring", "metric", "metrics"),
        "file": ("file", "files", "path", "paths", "directory", "directories", "folder",
                 "read", "write", "upload", "download", "attachment", "attachments"),
        "queue": ("queue", "queued", "job", "jobs", "worker", "workers", "background",
                  "schedule", "scheduled", "scheduler", "cron", "enqueue", "task", "tasks"),
        "notify": ("notify", "notification", "notifications", "email", "emails", "mail",
                   "alert", "alerts", "reminder", "reminders", "message", "messages"),
    }
)

_CONCEPT_OF: Mapping[str, str] = MappingProxyType(
    {word: head for head, words in CONCEPT_GROUPS.items() for word in words}
)

_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ies", "y"),
    ("ing", ""),
    ("es", ""),
    ("ed", ""),
    ("s", ""),
)

MIN_STEM_LENGTH = 4


def split_words(text: str) -> tuple[str, ...]:
    """Normalise and split ``text`` into raw lowercase words."""
    spaced = _CAMEL_BOUNDARY.sub(r"\1 \2", _ACRONYM_BOUNDARY.sub(r"\1 \2", text))
    folded = unicodedata.normalize("NFKD", spaced).lower()
    return tuple(word for word in _NON_WORD.split(folded) if word)


def stem(word: str) -> str:
    """Return ``word``'s suffix stem, or the word itself."""
    for suffix, replacement in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= MIN_STEM_LENGTH:
            return word[: -len(suffix)] + replacement
    return word


def concept_of(word: str) -> str | None:
    """The concept head for ``word``, or ``None``."""
    return _CONCEPT_OF.get(word)


def tokenize(text: str, *, is_query: bool = False) -> tuple[str, ...]:
    """Expand ``text`` into the tokens the index is scored over."""
    tokens: list[str] = []

    for word in split_words(text):
        if not MIN_TOKEN_LENGTH <= len(word) <= MAX_TOKEN_LENGTH:
            continue
        if word in STOP_WORDS:
            continue
        if is_query and word in DISCOURSE_WORDS and word not in RESCUED_WORDS:
            continue

        emitted = {word}
        tokens.append(word)

        stemmed = stem(word)
        if stemmed not in emitted:
            emitted.add(stemmed)
            tokens.append(stemmed)

        concept = concept_of(word)
        if concept is not None:
            tokens.append(f"concept:{concept}")

    return tuple(tokens)


def counts(tokens: Iterable[str]) -> dict[str, int]:
    """Term frequencies. A plain helper, kept here so callers share one shape."""
    frequencies: dict[str, int] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def is_code_shaped(word: str) -> bool:
    """Whether a query word looks like an identifier rather than English."""
    if len(word) < 3:
        return False
    if "_" in word or "." in word or "/" in word:
        return True
    return bool(_CAMEL_BOUNDARY.search(word) or _ACRONYM_BOUNDARY.search(word))
