"""Recovery shared by every model call in the graph.

A failed call is one of three things, and each gets a different answer:

- **capacity**: the reply ran out of output room (``finish_reason: length``,
  or the OpenAI SDK's ``LengthFinishReasonError`` when it parses structured
  output itself). Reasoning models hit this whenever hidden thinking is
  counted inside ``max_tokens``. The answer is more room: grow the cap and
  send the *same* request again. The content was never the problem.
- **format**: the reply is there but unusable (prose instead of the JSON
  report, a missing field). The answer is feedback: tell the model exactly
  what was wrong and re-ask on the same conversation, so the retained tool
  rounds and prompt cache are kept.
- **terminal**: the provider refused the request outright, or the cap cannot
  grow any further because the model's own limit is reached. Only this class
  ends a run, and it says why.

Nothing here counts attempts as a policy. Growth stops when the cap cannot
grow (the model or context window is the ceiling, not a constant), and
re-asks stop when the model starts repeating itself. A small safety valve on
re-asks exists only so a model that answers differently every time cannot
loop forever; it is not the normal stop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

MIN_OUTPUT_TOKENS = 1024        # never send a cap below this; nothing useful fits
FORMAT_REASKS = 4               # safety valve, not the normal stop (that is "model repeats itself")


class OutputCap:
    """A per-turn output cap that only ever grows, up to a real ceiling.

    ``ceiling`` is the room the model actually has (context window minus the
    prompt, or a limit the provider told us about), never a made-up constant.
    """

    def __init__(self, initial: int, ceiling: int):
        self.ceiling = max(MIN_OUTPUT_TOKENS, int(ceiling))
        self.value = max(MIN_OUTPUT_TOKENS, min(int(initial), self.ceiling))

    def grow(self, needed: int = 0) -> bool:
        """Double the cap (or reach ``needed``); False once the ceiling is hit."""
        wanted = max(self.value * 2, int(needed))
        grown = min(self.ceiling, wanted)
        if grown <= self.value:
            return False
        self.value = grown
        return True

    def lower_ceiling(self, accepted: int) -> None:
        """The provider rejected a larger cap; ``accepted`` is the last one it took."""
        self.ceiling = max(MIN_OUTPUT_TOKENS, min(self.ceiling, int(accepted)))
        self.value = min(self.value, self.ceiling)


def output_ceiling(window_tokens: int, input_tokens: int) -> int:
    """Output room left in the context window once the prompt is in it."""
    return max(MIN_OUTPUT_TOKENS, int(window_tokens) - int(input_tokens))


def hit_output_limit(reply, max_tokens: int | None = None) -> bool:
    """Whether ``reply`` stopped because it ran out of output budget.

    Chat completions and Anthropic report a finish reason; the Responses API
    sets ``status: incomplete`` with the cause under ``incomplete_details``.
    Neither is reliable for a tool call cut off mid-arguments: OpenRouter
    reported ``finish_reason: tool_calls`` with exactly ``max_tokens`` output
    tokens and an unparseable ``write_file`` call, four runs in a row. So a
    reply whose output tokens reach the cap it was sent with is truncated,
    whatever the finish reason says.
    """
    if max_tokens:
        usage = getattr(reply, "usage_metadata", None) or {}
        if int(usage.get("output_tokens") or 0) >= int(max_tokens):
            return True
    metadata = getattr(reply, "response_metadata", None) or {}
    reason = str(metadata.get("finish_reason") or metadata.get("stop_reason") or "").lower()
    if reason in ("length", "max_tokens", "max_output_tokens", "model_length"):
        return True
    details = metadata.get("incomplete_details")
    if isinstance(details, dict):
        return str(details.get("reason") or "").lower() == "max_output_tokens"
    return str(metadata.get("status") or "").lower() == "incomplete"


def truncated_usage(exc, raw) -> tuple[int, object]:
    """``(reasoning_tokens, chargeable)`` for a call that ran out of output.

    The SDK's length error carries the parsed completion with its usage block;
    a returned-but-truncated message carries LangChain usage metadata. Either
    way the tokens were billed and must be charged to the request.
    """
    completion = getattr(exc, "completion", None)
    usage = getattr(completion, "usage", None)
    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)
        chargeable = SimpleNamespace(usage_metadata={
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }, response_metadata={})
        return reasoning, chargeable
    meta = getattr(raw, "usage_metadata", None) or {}
    details = meta.get("output_token_details") or {}
    return int(details.get("reasoning") or 0), raw


_CAP_WORDS = re.compile(r"max_(?:completion_|output_)?tokens", re.IGNORECASE)


def rejected_cap(exc) -> bool:
    """Whether a provider 400 says the output cap we sent is too large."""
    status = getattr(exc, "status_code", None)
    if status not in (400, 422):
        return False
    return bool(_CAP_WORDS.search(str(getattr(exc, "message", None) or exc)))


@dataclass
class Outcome:
    """What one growing call ended with."""

    reply: object
    truncated: bool
    reasoning_tokens: int = 0
    wasted: list = field(default_factory=list)   # chargeable usage of truncated attempts other than ``reply``


def invoke_growing(call: Callable[[int], object], cap: OutputCap, *,
                   raw_of: Callable[[object], object] | None = None,
                   on_retry: Callable[[int, int, int], None] | None = None,
                   before_retry: Callable[[], None] | None = None) -> Outcome:
    """Run ``call(max_tokens)`` until the reply fits or the cap cannot grow.

    Truncated attempts other than the reply go to ``Outcome.wasted``; the caller charges each once.
    ``raw_of(reply)`` picks the chat message out of a structured
    output envelope; ``on_retry(previous_cap, new_cap, reasoning_tokens)``
    lets the caller log what happened. When the provider rejects a grown cap as too
    large, the ceiling drops to the last accepted value and the call is sent
    once more at that value; whatever comes back is final.
    """
    try:
        from openai import BadRequestError, LengthFinishReasonError
    except ImportError:  # pragma: no cover - provider SDK absent
        BadRequestError = LengthFinishReasonError = ()  # type: ignore[assignment]

    wasted: list = []
    accepted: int | None = None
    reasoning = 0
    while True:
        sent = cap.value
        reply = None
        try:
            reply = call(sent)
            accepted = sent
            raw = raw_of(reply) if raw_of else reply
            if not hit_output_limit(raw, sent):
                return Outcome(reply, False, reasoning, wasted)
            reasoning, chargeable = truncated_usage(None, raw)
        except LengthFinishReasonError as exc:
            accepted = sent
            reasoning, chargeable = truncated_usage(exc, None)
        except BadRequestError as exc:
            # Only a grown cap can be lowered; a rejection at an accepted cap would loop forever.
            if accepted is None or sent <= accepted or not rejected_cap(exc):
                raise
            # The grown cap is above what this model accepts. Go back to the
            # last accepted value, which is now the true ceiling, and finish
            # with one more call there.
            cap.lower_ceiling(accepted)
            if before_retry:
                before_retry()
            continue
        previous = cap.value
        if not cap.grow(needed=reasoning + MIN_OUTPUT_TOKENS):
            if reply is None:  # no reply to hand back: its usage is charged as wasted
                wasted.append(chargeable)
            return Outcome(reply, True, reasoning, wasted)
        wasted.append(chargeable)
        if on_retry:
            on_retry(previous, cap.value, reasoning)
        if before_retry:
            before_retry()


def repair_output(text: str, problem_of: Callable[[str], str],
                  resend: Callable[[str, str], str], *, limit: int = FORMAT_REASKS) -> tuple[str, str, int]:
    """Re-ask while the reply is unusable and the model is still changing it.

    ``problem_of(text)`` returns why the text is unusable, or "" when it is
    fine. ``resend(text, problem)`` sends that reason back and returns the new
    text. Stops when the text is usable, when the model repeats an earlier
    answer (more asking will not help), or at the safety valve.

    Returns ``(text, remaining_problem, reasks)``.
    """
    problem = problem_of(text)
    seen = {text}
    reasks = 0
    while problem and reasks < limit:
        reasks += 1
        text = resend(text, problem)
        problem = problem_of(text)
        if not problem or text in seen:
            break
        seen.add(text)
    return text, problem, reasks
