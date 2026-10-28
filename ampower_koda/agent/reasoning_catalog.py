"""Which models reason, read from OpenRouter's public model list and OpenAI's model pages.

Model names change too often for a list in the code, so both sources are fetched and kept in
Redis without an expiry: a failed fetch keeps serving the last good copy. The model pickers
refetch the OpenRouter list when it is an hour old, and building a model only when it is a day
old, so an agent run rarely waits on it. Each OpenAI page entry records when it was checked,
because frappe.local.cache pins whatever a job read first and a Redis expiry alone would never
be retried inside a long agent job.
"""

import re
import time
from urllib.parse import quote

import frappe
import requests

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENAI_MODEL_PAGE_URL = "https://developers.openai.com/api/docs/models/{}.md"
OPENROUTER_CACHE_KEY = "koda:openrouter_model_reasoning"
OPENAI_PAGE_CACHE_KEY_PREFIX = "koda:openai_model_page:"
OPENROUTER_REFRESH_LOCK_KEY = "koda:openrouter_model_reasoning_lock"
FETCH_TIMEOUT_SECONDS = 10
REFRESH_SECONDS = 24 * 3600
# A failed fetch is retried after this.
RETRY_SECONDS = 5 * 60
OPENAI_EFFORT_WORDS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
DATED_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def model_reasoning(provider: str, model: str) -> dict | None:
    """How ``model`` reasons on ``provider``, or None when no source lists it.

    The dict always has ``reasons``; a reasoning model's also has ``supported_efforts``,
    empty when its source names no levels.
    """
    name = model.strip().lower()
    if provider == "OpenRouter":
        return _openrouter_entry(name)
    if provider != "OpenAI":
        return None
    # OpenRouter lists a few dated snapshots; OpenAI's pages and most entries name the model without one.
    undated_model = DATED_SNAPSHOT_SUFFIX.sub("", name)
    entry = _openrouter_entry("openai/" + name)
    if entry is None:
        entry = _openrouter_entry("openai/" + undated_model)
    if entry is None:
        # OpenRouter lacks some OpenAI models and aliases (gpt-5-codex, *-chat-latest).
        entry = _openai_page_entry(undated_model)
    return entry


def _without_variant(model_id: str) -> str:
    # Variant suffixes such as ":nitro" route the same model.
    return model_id.split(":", 1)[0]


def refresh_openrouter_models() -> None:
    """Refetch OpenRouter's model list. A failed fetch keeps the last good copy."""
    fetched_models = _fetch_openrouter_models()
    if fetched_models:
        frappe.cache().set_value(OPENROUTER_CACHE_KEY, {"models": fetched_models, "fetched_at": time.time()})


def openrouter_models(max_age_seconds: int = REFRESH_SECONDS) -> dict:
    """Model id -> its entry from OpenRouter's list, refetched when older than ``max_age_seconds``.

    {} until the list has been read once.
    """
    cache = frappe.cache()
    # expires=True keeps the read out of frappe.local.cache, which would otherwise pin an empty
    # first read for the whole job.
    cached = cache.get_value(OPENROUTER_CACHE_KEY, expires=True)
    stale = cached is None or time.time() - cached["fetched_at"] > max_age_seconds
    if stale and _take_refresh_lock():
        refresh_openrouter_models()
        cached = cache.get_value(OPENROUTER_CACHE_KEY, expires=True)
    if cached is None:
        return {}
    return cached["models"]


def _take_refresh_lock() -> bool:
    # Held past the fetch for RETRY_SECONDS: workers starting at the same moment do not all
    # fetch, and a site that cannot reach OpenRouter does not wait on it for every model it builds.
    cache = frappe.cache()
    lock_key = cache.make_key(OPENROUTER_REFRESH_LOCK_KEY)
    try:
        return bool(cache.set(lock_key, 1, ex=RETRY_SECONDS, nx=True))
    except Exception:  # noqa: BLE001 - without Redis there is nowhere to keep the list anyway
        return False


def _openrouter_entry(model_id: str) -> dict | None:
    models = openrouter_models()
    return models.get(model_id) or models.get(_without_variant(model_id))


def _fetch_openrouter_models() -> dict:
    """Model id -> its entry; {} when the list cannot be read or comes back empty.

    Every entry has ``reasons``, ``tools`` (whether it takes tool calls, for the model picker)
    and ``created`` (unix seconds, 0 when missing); a reasoning model's also has
    ``supported_efforts``.
    """
    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=FETCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        rows = response.json().get("data")
    except Exception:  # noqa: BLE001 - the last good copy keeps serving
        frappe.logger("koda").warning("OpenRouter model list could not be read", exc_info=True)
        return {}
    if not isinstance(rows, list):
        return {}
    models = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        model_id = row["id"].lower()
        created = row.get("created")
        if not isinstance(created, int):
            created = 0
        supported_parameters = row.get("supported_parameters") or []
        takes_tools = "tools" in supported_parameters
        if "reasoning" not in supported_parameters:
            models[model_id] = {"reasons": False, "tools": takes_tools, "created": created}
            continue
        block = row.get("reasoning")
        efforts = block.get("supported_efforts") if isinstance(block, dict) else None
        if not isinstance(efforts, list):
            efforts = []
        models[model_id] = {"reasons": True, "supported_efforts": efforts, "tools": takes_tools,
                            "created": created}
    return models


def _openai_page_entry(name: str) -> dict | None:
    cache = frappe.cache()
    cache_key = OPENAI_PAGE_CACHE_KEY_PREFIX + name
    cached = cache.get_value(cache_key)
    now = time.time()
    if cached and now - cached["checked_at"] < cached["max_age"]:
        return cached["entry"]
    entry, definitive = _fetch_openai_page(name)
    if not definitive and cached:
        entry = cached["entry"]
    max_age = REFRESH_SECONDS if definitive else RETRY_SECONDS
    cache.set_value(cache_key, {"entry": entry, "checked_at": now, "max_age": max_age})
    return entry


def _fetch_openai_page(name: str) -> tuple[dict | None, bool]:
    """The page's entry for ``name`` and whether the answer is definitive (not a transient failure).

    Trusted only when it says the model reasons: after a rewording, a page must not mark a
    reasoning model as one that takes temperature=0.
    """
    try:
        response = requests.get(OPENAI_MODEL_PAGE_URL.format(quote(name, safe="")),
                                timeout=FETCH_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001 - the last good answer keeps serving
        return None, False
    if response.status_code == 404:
        return None, True
    page = response.text
    if response.status_code != 200 or f"Model ID: `{name}`" not in page:
        return None, False
    # Every reasoning model's page has this bullet. The effort sentence is prose and not on every
    # page; without it the configured effort is sent unchanged.
    effort_sentence = re.search(r"reasoning\.effort`?\s+supports:?([^\n]*)", page, re.I)
    has_reasoning_bullet = re.search(r"^- Reasoning token support", page, re.M) is not None
    if effort_sentence is None and not has_reasoning_bullet:
        return None, True
    efforts = []
    if effort_sentence is not None:
        first_clause = effort_sentence.group(1).split(". ")[0].lower()
        efforts = [word for word in re.findall(r"[a-z]+", first_clause) if word in OPENAI_EFFORT_WORDS]
    return {"reasons": True, "supported_efforts": efforts}, True
