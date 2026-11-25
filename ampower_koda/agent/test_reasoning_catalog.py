"""The reasoning catalog, the effort each model is sent, and the live model pickers.

Network and Redis are faked, so no site or internet is needed.
Run with: bench --site <site> run-tests --module ampower_koda.agent.test_reasoning_catalog
"""

import inspect
import os
import signal
import time
import unittest
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage

from ampower_koda.agent import api, cache_usage, graph, reasoning_catalog, run_control


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
     "context_length": 400000, "top_provider": {"max_completion_tokens": 128000},
     "pricing": {"prompt": "0.00000125", "completion": "0.00001", "input_cache_read": "0.000000125"},
     "reasoning": {"mandatory": True, "default_enabled": True,
                   "supported_efforts": ["high", "medium", "low", "minimal"]}},
    {"id": "openai/gpt-5:batch", "created": 300, "supported_parameters": ["reasoning", "tools"]},
    {"id": "openai/gpt-4o-mini", "created": 100, "supported_parameters": ["tools", "temperature"],
     "context_length": 128000, "top_provider": {"max_completion_tokens": 16384}},
    {"id": "openai/gpt-oss-120b", "created": 250, "supported_parameters": ["reasoning", "tools"]},
    {"id": "openai/o4-mini-high", "created": 260, "supported_parameters": ["reasoning", "tools"]},
    {"id": "openai/gpt-5-image", "created": 310, "supported_parameters": ["reasoning", "tools"]},
    {"id": "z-ai/glm-5", "created": 400, "supported_parameters": ["reasoning", "tools"], "reasoning": None},
    {"id": "anthropic/claude-opus-5.5", "created": 90, "supported_parameters": ["reasoning", "tools"],
     "context_length": 1000000, "top_provider": {"max_completion_tokens": 128000},
     "reasoning": {"supported_efforts": ["max", "xhigh", "high", "medium", "low"]}},
    {"id": "anthropic/claude-haiku-4.5", "created": 80, "supported_parameters": ["reasoning", "tools"],
     "context_length": 200000, "top_provider": {"max_completion_tokens": 64000}, "reasoning": {}},
    {"id": "google/gemini-2.5-pro", "created": 70, "supported_parameters": ["reasoning"],
     "pricing": {"prompt": "0.00000125", "completion": "0.00001", "input_cache_read": "0.000000125",
                 "overrides": [{"min_prompt_tokens": 200000, "prompt": "0.0000025", "completion": "0.000015",
                                "input_cache_read": "0.00000025"}]}},
    {"id": "anthropic/claude-opus-4.6", "created": 60, "supported_parameters": ["reasoning"],
     "pricing": {"prompt": "0.000005", "completion": "0.000025", "input_cache_read": "0.0000005",
                 "input_cache_write": "0.00000625"}},
    {"id": "openai/gpt-5.2-codex", "created": 40, "supported_parameters": ["reasoning"],
     "pricing": {"prompt": "0.00000175", "completion": "0.000014", "input_cache_read": "0.000000175"}},
    {"id": "openai/gpt-routed", "created": 30, "supported_parameters": [],
     "pricing": {"prompt": "-1", "completion": "-1"}},
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
            "tools": True, "created": 300, "context": 400000, "max_output": 128000,
            "price_tiers": [(0, (1.25, 0.125, 10.0, 1.25))]})
        self.assertEqual(models["openai/gpt-4o-mini"], {
            "reasons": False, "tools": True, "created": 100, "context": 128000, "max_output": 16384,
            "price_tiers": []})
        self.assertEqual(models["z-ai/glm-5"]["supported_efforts"], [])
        self.assertEqual(models["z-ai/glm-5"]["max_output"], 0)
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


class TestDirectModels(CatalogTestCase):
    """What _get_llm sends a direct Claude or OpenAI model."""

    def setUp(self):
        super().setUp()
        self.fetch_returns(FakeResponse(json_body={"data": OPENROUTER_ROWS}))
        reasoning_catalog.refresh_openrouter_models()
        keys = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "OPENAI_API_KEY": "test-key"})
        keys.start()
        self.addCleanup(keys.stop)
        effort = mock.patch.object(graph, "_configured_reasoning_effort", return_value="xhigh")
        effort.start()
        self.addCleanup(effort.stop)

    def payload(self, provider, model, **options):
        llm = graph._get_llm(provider=provider, model=model, **options)
        return llm._get_request_payload("hi", max_tokens=500000)

    def test_claude_ids_match_their_openrouter_entry(self):
        self.assertEqual(reasoning_catalog.model_limits("Claude", "claude-opus-5-5"), (1000000, 128000))
        for pinned in ("claude-haiku-4-5-20251001", "claude-haiku-4-5@20251001", "claude-haiku-4-5-latest"):
            self.assertEqual(reasoning_catalog.model_limits("Claude", pinned), (200000, 64000))
        self.assertEqual(reasoning_catalog.model_limits("Claude", "claude-unknown-9"), (0, 0))
        self.assertEqual(reasoning_catalog.model_limits("OpenAI", "gpt-5-2025-08-07"), (400000, 128000))

    def test_claude_with_effort_levels_thinks_adaptively(self):
        payload = self.payload("Claude", "claude-opus-5-5")
        self.assertEqual(payload["thinking"], graph.CLAUDE_ADAPTIVE_THINKING)
        self.assertEqual(payload["output_config"], {"effort": "xhigh"})
        self.assertEqual(payload["betas"], [graph.CLAUDE_THINKING_BINDING_BETA])
        self.assertEqual(payload["max_tokens"], 128000)
        self.assertNotIn("temperature", payload)
        lowered = self.payload("Claude", "claude-opus-5-5", reasoning_effort="low")
        self.assertEqual(lowered["output_config"], {"effort": "low"})

    def test_claude_without_effort_levels_does_not_think(self):
        payload = self.payload("Claude", "claude-haiku-4-5")
        self.assertFalse({"thinking", "output_config", "betas", "temperature"} & payload.keys())
        self.assertEqual(payload["max_tokens"], 64000)

    def test_claude_without_the_list_is_told_by_its_id(self):
        self.cache.values.clear()
        self.assertEqual(self.payload("Claude", "claude-sonnet-5-5")["output_config"], {"effort": "high"})
        for older in ("claude-sonnet-4-5", "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"):
            self.assertNotIn("thinking", self.payload("Claude", older))

    def test_openai_without_the_list_is_told_by_its_id(self):
        with mock.patch.object(graph.reasoning_catalog, "model_reasoning", return_value=None):
            for reasoning_model in ("gpt-5-mini", "gpt-5.2-codex", "o3-2025-04-16"):
                self.assertEqual(self.payload("OpenAI", reasoning_model)["reasoning"]["effort"], "high")
            lowered = self.payload("OpenAI", "gpt-5-mini", reasoning_effort="medium")
            self.assertEqual(lowered["reasoning"]["effort"], "medium")
            for other in ("gpt-4o-mini", "gpt-5-chat-latest", "gpt-5-pro", "o1-mini"):
                self.assertNotIn("reasoning", self.payload("OpenAI", other))

    def test_openai_cap_is_clamped_and_no_temperature_is_sent(self):
        payload = self.payload("OpenAI", "gpt-4o-mini")
        self.assertEqual(payload["max_output_tokens"], 16384)
        self.assertNotIn("temperature", payload)


def stream_chunk(stop_reason=None):
    return mock.Mock(message=mock.Mock(response_metadata={"stop_reason": stop_reason} if stop_reason else {}))


class TestClaudeStream(unittest.TestCase):
    """A Claude reply is taken whole or not at all."""

    def setUp(self):
        for target, name in ((graph.time, "sleep"), (graph, "check_active")):
            patcher = mock.patch.object(target, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def stream(self, *attempts):
        """What _stream hands on when the SDK's attempts are these chunks or errors."""
        llm = graph._ClaudeChat(model="claude-opus-5-5", max_tokens=1000, max_retries=2, api_key="test-key")
        with mock.patch.object(graph.ChatAnthropic, "_stream", side_effect=attempts) as self.sdk:
            return list(llm._stream([]))

    def test_incomplete_stream_is_retried_until_it_is_whole(self):
        whole = [stream_chunk(), stream_chunk("end_turn")]
        self.assertEqual(self.stream([stream_chunk()], whole), whole)

    def test_stream_that_never_completes_is_refused_after_the_retries(self):
        with self.assertRaises(ConnectionError):
            self.stream([stream_chunk()], [stream_chunk()], [stream_chunk()], [stream_chunk("end_turn")])
        self.assertEqual(self.sdk.call_count, 3)

    @unittest.skipUnless(hasattr(signal, "SIGALRM"), "needs SIGALRM")
    def test_job_timeout_becomes_the_run_deadline(self):
        from rq.timeouts import JobTimeoutException

        def rq_alarm(signum, frame):
            raise JobTimeoutException()

        def sdk_stream(*args, **kwargs):
            try:
                signal.raise_signal(signal.SIGALRM)
            except Exception as error:  # what the SDK does while it waits for the response headers
                raise graph.httpx.ConnectError("Connection error.") from error
            yield

        previous_handler = signal.signal(signal.SIGALRM, rq_alarm)
        self.addCleanup(signal.signal, signal.SIGALRM, previous_handler)
        llm = graph._ClaudeChat(model="claude-opus-5-5", max_tokens=1000, max_retries=2, api_key="test-key")
        with mock.patch.object(graph.ChatAnthropic, "_stream", side_effect=sdk_stream) as sdk:
            with self.assertRaises(run_control.RunDeadline):
                list(llm._stream([]))
        self.assertEqual(sdk.call_count, 1)
        self.assertIs(signal.getsignal(signal.SIGALRM), rq_alarm)


class TestOutputCap(unittest.TestCase):
    def rejection(self, text):
        request = graph.httpx.Request("POST", "http://test")
        return graph.anthropic.BadRequestError(text, response=graph.httpx.Response(400, request=request), body=None)

    def test_stated_limit_is_always_below_the_cap_sent(self):
        for text in ("max_tokens: 64000 > 32000, which is the maximum", "max_tokens must be at most 32000"):
            self.assertEqual(graph.recovery.stated_cap_limit(RuntimeError(text), 64000), 32000)
            self.assertEqual(graph.recovery.stated_cap_limit(RuntimeError(text), 32000), 0)

    def test_rejected_cap_is_sent_again_at_the_stated_limit(self):
        reply = graph.AIMessage(content="done")
        call = mock.Mock(side_effect=[self.rejection("max_tokens: 32000 > 8192, which is the maximum"), reply])
        outcome = graph.recovery.invoke_growing(call, graph.recovery.OutputCap(32000, 64000))
        self.assertIs(outcome.reply, reply)
        self.assertEqual(call.call_args_list, [mock.call(32000), mock.call(8192)])

    def test_cap_that_does_not_fit_beside_the_prompt_is_lowered(self):
        openai = ("This model's maximum context length is 128000 tokens. However, you requested 130000 tokens "
                  "(98000 in the messages, 32000 in the completion).")
        openrouter = ("This endpoint's maximum context length is 128000 tokens. However, you requested about "
                      "130000 tokens (98000 of text input, 32000 in the output).")
        for text in (openai, openrouter):
            reply = graph.AIMessage(content="done")
            call = mock.Mock(side_effect=[self.rejection(text), reply])
            outcome = graph.recovery.invoke_growing(call, graph.recovery.OutputCap(32000, 64000))
            self.assertIs(outcome.reply, reply)
            self.assertEqual(call.call_args_list, [mock.call(32000), mock.call(30000)])

    def test_growth_stops_at_the_models_output_limit(self):
        llm = graph._OpenAIChat(model="gpt-4o-mini", max_output_limit=16384, api_key="test-key")
        self.assertEqual(graph._within_model_output(64000, llm), 16384)
        self.assertEqual(graph._within_model_output(64000, graph.ChatOpenAI(model="m", api_key="test-key")), 64000)

    def test_other_rejections_are_raised(self):
        for text in ("the prompt must be at most 8192 tokens", "max_tokens is invalid"):
            call = mock.Mock(side_effect=self.rejection(text))
            with self.assertRaises(graph.anthropic.BadRequestError):
                graph.recovery.invoke_growing(call, graph.recovery.OutputCap(32000, 64000))
            call.assert_called_once()


class TestCutOffReply(unittest.TestCase):
    """Half of a reply's last tool call can parse, so a reply that was cut off runs none."""

    def run_loop(self, *replies):
        tool = mock.Mock()
        tool.name = "read_file"
        tool.invoke.return_value = "text"
        llm = mock.Mock(model_name="m")
        with mock.patch.object(graph, "_publish_agent_log"), mock.patch.object(graph, "_persist_token_usage"), \
                mock.patch.object(graph, "convert_to_openai_tool", return_value={}), \
                mock.patch.object(graph, "_request_spend_pressure", return_value=""), \
                mock.patch.object(graph.koda_core, "request_limits", return_value=(200000, 150000)), \
                mock.patch.object(graph, "_within_model_output", return_value=graph.MODEL_ROUND_OUTPUT_TOKENS), \
                mock.patch.object(graph, "_invoke_limited", side_effect=replies) as self.model:
            outcome = graph._run_tool_calling_loop(llm, [tool], "system", "task", provider="Claude", max_rounds=20)
        return tool, outcome

    def reply(self, stop_reason, output_tokens=10):
        usage = {"input_tokens": 1, "output_tokens": output_tokens, "total_tokens": 1 + output_tokens}
        return graph.AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "1"}],
                               response_metadata={"stop_reason": stop_reason}, usage_metadata=usage)

    def test_its_tool_calls_are_not_run(self):
        done = graph.AIMessage(content="done")
        at_the_cap = self.reply("max_tokens", graph.MODEL_ROUND_OUTPUT_TOKENS)
        for cut_off in (self.reply("refusal"), self.reply("model_context_window_exceeded"), at_the_cap):
            tool, _ = self.run_loop(cut_off, done)
            tool.invoke.assert_not_called()
        tool, _ = self.run_loop(self.reply("tool_use"), done)
        tool.invoke.assert_called_once()

    def test_the_pass_ends_when_every_reply_is_cut_off(self):
        tool, outcome = self.run_loop(*[self.reply("refusal")] * 10)
        self.assertEqual(self.model.call_count, graph.MAX_CUT_OFF_REPLIES + 2)
        self.assertEqual(outcome[3:], (graph.MAX_CUT_OFF_REPLIES + 2, True))
        tool.invoke.assert_not_called()


class TestCostEstimate(CatalogTestCase):
    """A direct provider reports no cost, so it is worked out from the response's tokens."""

    def setUp(self):
        super().setUp()
        self.fetch_returns(FakeResponse(json_body={"data": OPENROUTER_ROWS}))
        reasoning_catalog.refresh_openrouter_models()

    def cost(self, model, input_tokens=10000, **details):
        reply = AIMessage(content="ok", response_metadata={"model_name": model}, usage_metadata={
            "input_tokens": input_tokens, "output_tokens": 500, "total_tokens": input_tokens + 500,
            "input_token_details": details})
        return cache_usage.provider_cost(reply)

    def test_model_missing_from_the_table_is_priced_from_the_list(self):
        self.assertEqual(self.cost("gpt-5.2-codex", cache_read=4000), 0.0182)
        self.assertEqual(self.cost("claude-opus-4-6", cache_read=4000, cache_creation=3000), 0.04825)

    def test_dated_and_numbered_releases_are_priced_as_their_model(self):
        self.assertEqual(self.cost("claude-opus-4-6-20260205"), self.cost("claude-opus-4-6"))
        self.assertEqual(self.cost("gemini-2.5-pro-001"), self.cost("gemini-2.5-pro"))

    def test_gemini_is_priced_with_its_long_prompt_tier(self):
        self.assertEqual(self.cost("gemini-2.5-pro", cache_read=4000), 0.013)
        self.assertEqual(self.cost("gemini-2.5-pro", input_tokens=300000), 0.7575)

    def test_claude_writes_split_by_lifetime_are_counted(self):
        details = {"cache_creation": 0, "ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000}
        self.assertEqual(cache_usage.cache_write_tokens(details), 3000)
        self.assertEqual(self.cost("claude-opus-5-5", cache_read=4000, **details), 0.0438)

    def test_unknown_prices_cost_nothing(self):
        self.assertEqual(self.cost("gpt-unknown"), 0.0)
        self.assertEqual(self.cost("gpt-routed"), 0.0)
        self.assertEqual(self.cost(""), 0.0)

    def test_unreadable_list_does_not_fail_the_run(self):
        with mock.patch.object(reasoning_catalog, "openrouter_models", side_effect=RuntimeError("no Redis")), \
                mock.patch.object(cache_usage, "log_agent_error"), \
                mock.patch.object(cache_usage.frappe, "get_traceback", return_value=""):
            self.assertEqual(self.cost("gpt-5.2-codex"), 0.0)


class TestGeminiCap(unittest.TestCase):
    def test_per_call_cap_is_sent_under_googles_name(self):
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            llm = graph._get_llm(provider="Gemini", model="gemini-2.5-pro")
        with mock.patch.object(type(llm), "_prepare_request", side_effect=InterruptedError) as prepare:
            with self.assertRaises(InterruptedError):
                graph._invoke_limited(llm, [("human", "hi")], 1234)
        self.assertEqual(prepare.call_args.kwargs["max_output_tokens"], 1234)
        self.assertNotIn("max_tokens", prepare.call_args.kwargs)

    def test_no_temperature_is_sent(self):
        for model in ("gemini-2.5-pro", "gemini-3.8-flash", "gemini-4-carbon"):
            with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
                llm = graph._get_llm(provider="Gemini", model=model)
            request = llm._prepare_request([HumanMessage(content="hi")])
            self.assertIsNone(request["config"].temperature)


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
            "openai/gpt-oss-120b", "anthropic/claude-opus-5.5"])

    def test_openai_fallback_keeps_only_models_openai_serves(self):
        self.assertEqual(api._openai_models_on_openrouter(0), ["gpt-5"])

    def test_openai_live_list_drops_older_models_and_keeps_unlisted_new_ones(self):
        names = ["gpt-9-new", "gpt-5", "gpt-4o-mini", "gpt-4-turbo", "gpt-5-chat-latest"]
        self.assertEqual(api._current_openai_models(names), ["gpt-9-new", "gpt-5"])

    def test_settings_default_that_is_older_is_not_offered(self):
        for model, offered in (("gpt-4o-mini", ""), ("gpt-5", "gpt-5")):
            settings = mock.Mock(default_ai_provider="OpenAI", default_ai_model=model)
            with mock.patch.object(api.frappe, "get_single", return_value=settings):
                self.assertEqual(inspect.unwrap(api.get_model_defaults)()["model"], offered)


class TestOlderModels(CatalogTestCase):
    """Older models cannot be selected, whether picked or typed."""

    OLDER = (("OpenAI", "gpt-4o-mini"), ("OpenAI", "gpt-4.1-2025-04-14"), ("OpenAI", "gpt-5-chat-latest"),
             ("OpenAI", "o1-mini"), ("Gemini", "gemini-2.0-flash"), ("Gemini", "gemini-2.5-pro-001"),
             ("Claude", "claude-3-5-haiku-20241022"), ("Claude", "claude-sonnet-4-20250514"),
             ("Claude", "claude-haiku-4-5"), ("Claude", "claude-opus-4-1-20250805"),
             ("OpenRouter", "anthropic/claude-sonnet-4"), ("OpenRouter", "anthropic/claude-haiku-4.5"),
             ("OpenRouter", "openai/gpt-4o-mini:nitro"), ("OpenRouter", "google/gemini-2.5-pro"))
    CURRENT = (("OpenAI", "gpt-6-luna"), ("OpenAI", "gpt-6.2-sol"), ("OpenAI", "o3"),
               ("Gemini", "gemini-3.8-flash"), ("Gemini", "gemini-4-carbon"), ("Claude", "claude-sonnet-4-6"),
               ("Claude", "claude-opus-5-5"), ("Claude", "claude-sonnet-5-20260630"),
               ("OpenRouter", "anthropic/claude-sonnet-5.5"), ("OpenRouter", "z-ai/glm-5:nitro"),
               ("OpenRouter", "deepseek/deepseek-v4-flash-0731"))

    def check(self):
        for provider, model in self.OLDER:
            self.assertTrue(reasoning_catalog.is_older_model(provider, model), model)
        for provider, model in self.CURRENT:
            self.assertFalse(reasoning_catalog.is_older_model(provider, model), model)

    def test_with_the_list(self):
        self.fetch_returns(FakeResponse(json_body={"data": OPENROUTER_ROWS}))
        reasoning_catalog.refresh_openrouter_models()
        self.check()
        # Only the list tells that these do not reason.
        self.assertTrue(reasoning_catalog.is_older_model("OpenRouter", "deepseek/deepseek-chat:free"))
        self.assertTrue(reasoning_catalog.is_older_model("OpenAI", "gpt-routed"))

    def test_without_the_list(self):
        with mock.patch.object(reasoning_catalog, "openrouter_models", return_value={}):
            self.check()

    def test_only_an_older_model_stops_the_save(self):
        refused = []
        with mock.patch.object(reasoning_catalog, "openrouter_models", return_value={}), \
                mock.patch.object(reasoning_catalog, "frappe") as fake_frappe:
            fake_frappe.throw.side_effect = refused.append
            for model in ("gpt-6-luna", "", None):
                reasoning_catalog.refuse_older_model("OpenAI", model)
            self.assertEqual(refused, [])
            reasoning_catalog.refuse_older_model(None, "gpt-4o-mini")
            self.assertEqual(len(refused), 1)


if __name__ == "__main__":
    unittest.main()
