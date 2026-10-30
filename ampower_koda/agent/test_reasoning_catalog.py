"""The reasoning catalog, the effort each model is sent, and the live model pickers.

Network and Redis are faked, so no site or internet is needed.
Run with: bench --site <site> run-tests --module ampower_koda.agent.test_reasoning_catalog
"""

import os
import time
import unittest
from unittest import mock

from ampower_koda.agent import api, graph, reasoning_catalog


class FakeCache:
    """The parts of frappe.cache() the catalog uses, kept in a dict."""

    def __init__(self):
        self.values = {}
        self.redis_keys = set()

    def get_value(self, key, expires=False):
        return self.values.get(key)

    def set_value(self, key, value, expires_in_sec=None):
        self.values[key] = value

    def make_key(self, key):
        return "site|" + key

    def set(self, key, value, ex=None, nx=False):
        if nx and key in self.redis_keys:
            return None
        self.redis_keys.add(key)
        return True


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self.json_body = json_body
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.json_body


OPENROUTER_ROWS = [
    {"id": "openai/gpt-5", "created": 300, "supported_parameters": ["reasoning", "tools"],
     "reasoning": {"mandatory": True, "default_enabled": True,
                   "supported_efforts": ["high", "medium", "low", "minimal"]}},
    {"id": "openai/gpt-5:batch", "created": 300, "supported_parameters": ["reasoning", "tools"]},
    {"id": "openai/gpt-4o-mini", "created": 100, "supported_parameters": ["tools", "temperature"]},
    {"id": "openai/gpt-oss-120b", "created": 250, "supported_parameters": ["reasoning", "tools"]},
    {"id": "openai/o4-mini-high", "created": 260, "supported_parameters": ["reasoning", "tools"]},
    {"id": "openai/gpt-5-image", "created": 310, "supported_parameters": ["reasoning", "tools"]},
    {"id": "z-ai/glm-5", "created": 400, "supported_parameters": ["reasoning", "tools"], "reasoning": None},
    {"id": "black-forest-labs/flux", "created": 500, "supported_parameters": []},
    {"id": "deepseek/deepseek-chat:free", "created": 50, "supported_parameters": ["tools"]},
    "not a row",
    {"id": 42},
]


class CatalogTestCase(unittest.TestCase):
    def setUp(self):
        self.cache = FakeCache()
        patcher = mock.patch.object(reasoning_catalog.frappe, "cache", return_value=self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)
        logger_patcher = mock.patch.object(reasoning_catalog.frappe, "logger")
        logger_patcher.start()
        self.addCleanup(logger_patcher.stop)

    def fetch_returns(self, *responses):
        patcher = mock.patch.object(reasoning_catalog.requests, "get", side_effect=list(responses))
        fake_get = patcher.start()
        self.addCleanup(patcher.stop)
        return fake_get


class TestModelEffort(unittest.TestCase):
    def test_listed_level_is_sent_unchanged(self):
        entry = {"reasons": True, "supported_efforts": ["low", "medium", "high", "xhigh"]}
        self.assertEqual(graph._model_effort("xhigh", entry), "xhigh")

    def test_model_taking_only_higher_levels_gets_its_nearest(self):
        entry = {"reasons": True, "supported_efforts": ["high"]}
        self.assertEqual(graph._model_effort("low", entry), "high")

    def test_tie_goes_to_the_cheaper_level(self):
        entry = {"reasons": True, "supported_efforts": ["low", "high"]}
        self.assertEqual(graph._model_effort("medium", entry), "low")

    def test_model_listing_no_levels_is_capped_at_high(self):
        self.assertEqual(graph._model_effort("max", {"reasons": True, "supported_efforts": []}), "high")
        self.assertEqual(graph._model_effort("medium", {"reasons": True}), "medium")

    def test_levels_settings_never_offer_are_ignored(self):
        entry = {"reasons": True, "supported_efforts": ["none", "minimal", "medium"]}
        self.assertEqual(graph._model_effort("low", entry), "medium")

    def test_caller_effort_only_lowers(self):
        self.assertEqual(graph._lower_effort("high", "low"), "low")
        self.assertIsNone(graph._lower_effort("medium", "high"))
        self.assertIsNone(graph._lower_effort("high", "Same as Reasoning Effort"))


class TestOpenRouterList(CatalogTestCase):
    def test_rows_become_entries(self):
        self.fetch_returns(FakeResponse(json_body={"data": OPENROUTER_ROWS}))
        models = reasoning_catalog._fetch_openrouter_models()
        self.assertEqual(models["openai/gpt-5"], {
            "reasons": True, "supported_efforts": ["high", "medium", "low", "minimal"],
            "tools": True, "created": 300})
        self.assertEqual(models["openai/gpt-4o-mini"], {"reasons": False, "tools": True, "created": 100})
        self.assertEqual(models["z-ai/glm-5"]["supported_efforts"], [])
        self.assertNotIn(42, models)

    def test_unreadable_list_is_empty(self):
        self.fetch_returns(RuntimeError("offline"))
        self.assertEqual(reasoning_catalog._fetch_openrouter_models(), {})

    def test_first_use_fetches_once_then_reads_redis(self):
        fake_get = self.fetch_returns(FakeResponse(json_body={"data": OPENROUTER_ROWS}))
        self.assertTrue(reasoning_catalog.openrouter_models()["openai/gpt-5"]["reasons"])
        self.assertTrue(reasoning_catalog.openrouter_models()["openai/gpt-5"]["reasons"])
        self.assertEqual(fake_get.call_count, 1)

    def test_worker_without_the_lock_does_not_fetch(self):
        fake_get = self.fetch_returns()
        self.cache.set(self.cache.make_key(reasoning_catalog.OPENROUTER_REFRESH_LOCK_KEY), 1, nx=True)
        self.assertEqual(reasoning_catalog.openrouter_models(), {})
        fake_get.assert_not_called()

    def test_failed_refresh_keeps_the_last_good_copy(self):
        self.fetch_returns(FakeResponse(json_body={"data": OPENROUTER_ROWS}), RuntimeError("offline"))
        reasoning_catalog.refresh_openrouter_models()
        reasoning_catalog.refresh_openrouter_models()
        self.assertIn("openai/gpt-5", reasoning_catalog.openrouter_models())

    def test_copy_older_than_the_max_age_is_refetched(self):
        fake_get = self.fetch_returns(FakeResponse(json_body={"data": OPENROUTER_ROWS}))
        self.cache.values[reasoning_catalog.OPENROUTER_CACHE_KEY] = {"models": {}, "fetched_at": 0}
        self.assertIn("openai/gpt-5", reasoning_catalog.openrouter_models(max_age_seconds=3600))
        self.assertEqual(fake_get.call_count, 1)

    def test_fresh_copy_is_not_refetched(self):
        fake_get = self.fetch_returns()
        fresh_copy = {"models": {"openai/gpt-5": {"reasons": True}}, "fetched_at": time.time()}
        self.cache.values[reasoning_catalog.OPENROUTER_CACHE_KEY] = fresh_copy
        self.assertIn("openai/gpt-5", reasoning_catalog.openrouter_models(max_age_seconds=3600))
        fake_get.assert_not_called()

    def test_failed_fetch_is_not_retried_by_the_next_call(self):
        fake_get = self.fetch_returns(RuntimeError("offline"))
        self.assertEqual(reasoning_catalog.openrouter_models(), {})
        self.assertEqual(reasoning_catalog.openrouter_models(), {})
        self.assertEqual(fake_get.call_count, 1)

    def test_unlisted_model_does_not_refetch(self):
        fake_get = self.fetch_returns(FakeResponse(json_body={"data": OPENROUTER_ROWS}))
        reasoning_catalog.openrouter_models()
        self.assertIsNone(reasoning_catalog.model_reasoning("OpenRouter", "nobody/typo-model"))
        self.assertEqual(fake_get.call_count, 1)


class TestModelReasoning(CatalogTestCase):
    def setUp(self):
        super().setUp()
        self.cache.values[reasoning_catalog.OPENROUTER_CACHE_KEY] = {
            "models": self.parsed_rows(),
            "fetched_at": time.time(),
        }

    def parsed_rows(self):
        with mock.patch.object(reasoning_catalog.requests, "get",
                               return_value=FakeResponse(json_body={"data": OPENROUTER_ROWS})):
            return reasoning_catalog._fetch_openrouter_models()

    def test_variant_suffix_routes_the_listed_model(self):
        entry = reasoning_catalog.model_reasoning("OpenRouter", "z-ai/glm-5:nitro")
        self.assertTrue(entry["reasons"])

    def test_model_ids_are_case_insensitive(self):
        self.assertTrue(reasoning_catalog.model_reasoning("OpenRouter", "OpenAI/GPT-5")["reasons"])

    def test_direct_openai_dated_snapshot_reads_its_alias(self):
        entry = reasoning_catalog.model_reasoning("OpenAI", "gpt-5-2025-08-07")
        self.assertEqual(entry["supported_efforts"], ["high", "medium", "low", "minimal"])

    def test_direct_openai_model_missing_from_openrouter_reads_its_page(self):
        page = ("# GPT-5 Codex\n\nModel ID: `gpt-5-codex`\n\n- Reasoning token support\n\n"
                "The `reasoning.effort` supports: low, medium, and high. Default is medium.\n")
        self.fetch_returns(FakeResponse(text=page))
        entry = reasoning_catalog.model_reasoning("OpenAI", "gpt-5-codex")
        self.assertEqual(entry, {"reasons": True, "supported_efforts": ["low", "medium", "high"]})

    def test_other_providers_are_not_looked_up(self):
        self.assertIsNone(reasoning_catalog.model_reasoning("Gemini", "gemini-2.5-pro"))


class TestOpenAIPage(CatalogTestCase):
    def test_reasoning_bullet_without_effort_sentence(self):
        self.fetch_returns(FakeResponse(text="Model ID: `o3`\n- Reasoning token support\n"))
        self.assertEqual(reasoning_catalog._fetch_openai_page("o3"),
                         ({"reasons": True, "supported_efforts": []}, True))

    def test_page_without_reasoning_is_definitive_none(self):
        self.fetch_returns(FakeResponse(text="Model ID: `gpt-4.1`\n- Streaming\n"))
        self.assertEqual(reasoning_catalog._fetch_openai_page("gpt-4.1"), (None, True))

    def test_missing_page_is_definitive_none(self):
        self.fetch_returns(FakeResponse(status_code=404))
        self.assertEqual(reasoning_catalog._fetch_openai_page("gpt-nope"), (None, True))

    def test_page_for_another_model_is_not_trusted(self):
        self.fetch_returns(FakeResponse(text="Model ID: `gpt-5`\n- Reasoning token support\n"))
        self.assertEqual(reasoning_catalog._fetch_openai_page("gpt-5-codex"), (None, False))

    def test_transient_failure_keeps_the_last_answer(self):
        cached_entry = {"reasons": True, "supported_efforts": ["high"]}
        self.cache.values[reasoning_catalog.OPENAI_PAGE_CACHE_KEY_PREFIX + "o3"] = {
            "entry": cached_entry, "checked_at": 0, "max_age": 1}
        self.fetch_returns(RuntimeError("offline"))
        self.assertEqual(reasoning_catalog._openai_page_entry("o3"), cached_entry)


class TestOpenRouterReasoningField(unittest.TestCase):
    """What _get_llm puts in OpenRouter's reasoning field."""

    def reasoning_field(self, model, catalog_entry):
        with mock.patch.object(graph.reasoning_catalog, "model_reasoning", return_value=catalog_entry), \
                mock.patch.object(graph, "_configured_reasoning_effort", return_value="high"), \
                mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            llm = graph._get_llm(provider="OpenRouter", model=model)
        return llm.extra_body.get("reasoning")

    def test_listed_reasoning_model_gets_the_effort(self):
        entry = {"reasons": True, "supported_efforts": ["low", "medium", "high"]}
        self.assertEqual(self.reasoning_field("deepseek/deepseek-v4", entry), {"effort": "high"})

    def test_unlisted_always_reasoning_model_is_capped(self):
        self.assertEqual(self.reasoning_field("z-ai/glm-5", None),
                         {"max_tokens": graph.MODEL_REASONING_BUDGET_TOKENS})

    def test_unlisted_hybrid_model_gets_no_field(self):
        self.assertIsNone(self.reasoning_field("anthropic/claude-sonnet-4", None))

    def test_listed_model_that_does_not_reason_gets_no_field(self):
        self.assertIsNone(self.reasoning_field("z-ai/glm-5", {"reasons": False}))


class TestModelPickers(unittest.TestCase):
    def setUp(self):
        with mock.patch.object(reasoning_catalog.requests, "get",
                               return_value=FakeResponse(json_body={"data": OPENROUTER_ROWS})):
            models = reasoning_catalog._fetch_openrouter_models()
        patcher = mock.patch.object(api.reasoning_catalog, "openrouter_models", return_value=models)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_openrouter_picker_lists_tool_models_newest_first_without_variants(self):
        self.assertEqual(api._openrouter_tool_models(0), [
            "z-ai/glm-5", "openai/gpt-5-image", "openai/gpt-5", "openai/o4-mini-high",
            "openai/gpt-oss-120b", "openai/gpt-4o-mini"])

    def test_openai_fallback_keeps_only_models_openai_serves(self):
        self.assertEqual(api._openai_models_on_openrouter(0), ["gpt-5", "gpt-4o-mini"])


if __name__ == "__main__":
    unittest.main()
