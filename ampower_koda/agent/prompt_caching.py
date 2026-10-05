"""Provider request boundaries, applied after messages reach their final order."""

import hashlib
import json
import re
from copy import deepcopy


def openai_breakpoints(provider: str, model: str) -> bool:
    """Only send OpenAI's explicit-marker extension to documented model families."""
    if provider not in ("OpenAI", "OpenRouter"):
        return False
    identifier = (model or "").lower()
    if "/" in identifier:
        namespace, identifier = identifier.split("/", 1)
        if namespace != "openai":
            return False
    match = re.match(r"^gpt-(\d+)(?:\.(\d+))?(?:-|$)", identifier)
    return bool(match and (int(match[1]), int(match[2] or 0)) >= (5, 6))


def prefix_digest(messages) -> str:
    """Compare retained content without treating cache annotations as edits."""
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                    if key not in ("cache_control", "prompt_cache_breakpoint")}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    payload = [clean({"type": m.type,
                      "content": ([{"type": "text", "text": m.content}]
                                  if isinstance(m.content, str) else m.content),
                      "tool_calls": getattr(m, "tool_calls", []),
                      "tool_call_id": getattr(m, "tool_call_id", "")}) for m in messages]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()


def mark_message(message, ttl="5m", *, openai=False):
    """Mark a text block on a copy; never mutate retained conversation history."""
    if ttl == "none":
        return message
    content = deepcopy(message.content)
    if isinstance(content, str):
        if not content:
            return message
        content = [{"type": "text", "text": content}]
    for part in reversed(content):
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            if openai:
                part["prompt_cache_breakpoint"] = {"mode": "explicit"}
            else:
                part["cache_control"] = {"type": "ephemeral", "ttl": ttl}
            return message.model_copy(update={"content": content})
    return message


def terminal_model(llm, tools, bound=None, *, provider=""):
    """Keep identical tool schemas while forbidding calls on the final request."""
    if not tools or not hasattr(llm, "bind_tools"):
        return llm
    choice = {"type": "none"} if provider == "Claude" else "none"
    try:
        return llm.bind_tools(tools, tool_choice=choice)
    except TypeError:
        # Older adapters bind request options separately from tool schemas.
        bound = bound if bound is not None else llm.bind_tools(tools)
        return bound.bind(tool_choice=choice) if hasattr(bound, "bind") else bound


def native_cache_messages(messages, provider):
    """Anthropic marks the tool_result wrapper, never its nested text blocks.

    New LangChain versions hoist this themselves; explicit wrappers also work
    with the older versions allowed by our dependency requirements.
    """
    if provider != "Claude":
        return messages
    out = []
    for message in messages:
        if message.type != "tool" or not isinstance(message.content, list):
            out.append(message)
            continue
        content = deepcopy(message.content)
        control = None
        for part in content:
            if isinstance(part, dict) and "cache_control" in part and part.get("type") == "text":
                control = part.pop("cache_control")
        if control:
            message = message.model_copy(update={"content": [{
                "type": "tool_result", "tool_use_id": message.tool_call_id,
                "content": content, "is_error": getattr(message, "status", "success") == "error",
                "cache_control": control,
            }]})
        out.append(message)
    return out


def rolling_messages(messages, history, *, enabled):
    """Keep at most two rolling boundaries, alongside two fixed boundaries.

    A previous boundary survives only while its complete prefix is unchanged.
    Compaction therefore invalidates it instead of attaching it to shifted data.
    """
    if not enabled:
        return messages
    out = list(messages)
    fixed = sum(1 for message in out if isinstance(message.content, list)
                for part in message.content if isinstance(part, dict) and "cache_control" in part)
    candidates = [i for i, m in enumerate(out) if m.type != "system" and m.content]
    if not candidates or fixed >= 4:
        return out
    current = candidates[-1]
    previous = history.get("cache_marker")
    if previous and fixed < 3:
        index = previous["index"]
        if 0 <= index < current and prefix_digest(out[:index + 1]) == previous["digest"]:
            out[index] = mark_message(out[index])
    out[current] = mark_message(out[current])
    history["cache_marker"] = {"index": current, "digest": prefix_digest(out[:current + 1])}
    return out
