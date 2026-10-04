"""Recover a tool call a provider shipped as chat-template markup instead of ``tool_calls``.

An OpenAI-compatible server that refuses to turn the model's call into a ``tool_calls`` payload —
because the call names a tool the request did not offer: a deferred tool behind ``tool_search``,
an MCP tool the session folded out — hands the call through as *content*. The client strips that
markup as scaffolding (``agent_runtime_helpers.strip_think_blocks``), so the turn reads as an empty
reply: the user asked for an action and nothing came back, and the empty-response ladder then
re-prompts a model that already knew what it wanted.

``recover_markup_tool_calls`` rebuilds the call the model meant to make, and only when the markup is
the whole answer — nothing visible survives once the blocks are removed. Markup beside prose is
prose mentioning a call; the text is the answer and is left alone.

* A name the session defers goes through the ``tool_call`` bridge exactly as the model would have
  written it: a wrong argument shape comes back as the bridge's own corrective error result instead
  of a vanished turn.
* Any other *registered* tool is called directly. An unregistered name is not recovered — a
  hallucinated tool keeps today's behaviour rather than gaining a dispatch path.

Ids come from ``deterministic_call_id``: a random id would break the prompt-cache prefix for every
later request of the conversation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.message_sanitization import deterministic_call_id
from agent.transports.types import ToolCall
from tools.registry import registry
from tools import tool_search as _tool_search
from tools.tool_search_catalog import TOOL_CALL_NAME

logger = logging.getLogger("agent.conversation_loop")

# Optional XML namespace prefix: some servers serialize a call as ``<ns:tool_call>``.
_NS_PREFIX = r"(?:[\w.-]+:)?"
_ENVELOPE_RE = re.compile(
    rf"<{_NS_PREFIX}tool_call\b[^>]*>(.*?)</{_NS_PREFIX}tool_call>", re.DOTALL | re.IGNORECASE
)
# Qwen/Unsloth XML form: ``<function=NAME><parameter=KEY>value</parameter></function>``.
_FUNCTION_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>", re.DOTALL | re.IGNORECASE)
_PARAMETER_RE = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL | re.IGNORECASE)
_TOOL_NAME_RE = re.compile(r"[\w.:-]+")

_LOOSE_KEYS = frozenset({"name", "tool", "function", "call", "type"})


def _typed(value: str) -> Any:
    """A parameter value as JSON when it is a JSON scalar, else the text as written.

    The bridge validates the result against the tool's schema, so a value that parses but does not
    fit the schema comes back as an ordinary corrective error rather than a wrong call.
    """

    text = value.strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    if isinstance(parsed, (str, int, float, bool)) or parsed is None:
        return parsed
    return text


def _name_and_arguments(name: Any, payload: Dict[str, Any], explicit: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """``(name, arguments)`` from a JSON payload, loose about where the arguments live."""

    text = str(name or "").strip()
    if not text:
        return None
    if explicit is None:
        explicit = {key: value for key, value in payload.items() if key not in _LOOSE_KEYS}
    if isinstance(explicit, str):
        try:
            explicit = json.loads(explicit) if explicit.strip() else {}
        except (TypeError, ValueError):
            return None
    return (text, explicit) if isinstance(explicit, dict) else None


def _call_from_json(block: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """A ``<tool_call>{"name": …, "arguments": {…}}</tool_call>`` payload, or None."""

    try:
        payload = json.loads(block)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("function"), dict):
        function = payload["function"]
        return _name_and_arguments(
            function.get("name") or function.get("tool"), function,
            function.get("arguments", function.get("args", function.get("parameters"))),
        )
    return _name_and_arguments(
        payload.get("name") or payload.get("tool") or payload.get("function"), payload,
        payload.get("arguments", payload.get("args", payload.get("parameters"))),
    )


def _call_from_function_block(block: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """A ``<function=NAME><parameter=K>v</parameter></function>`` block, or None."""

    match = _FUNCTION_RE.fullmatch(block.strip())
    if match is None:
        return None
    name = match.group(1).strip()
    body = match.group(2)
    if _TOOL_NAME_RE.fullmatch(name) is None or _PARAMETER_RE.sub("", body).strip():
        return None
    return name, {key.strip(): _typed(value) for key, value in _PARAMETER_RE.findall(body)}


def _call_from_block(block: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    text = block.strip()
    if not text:
        return None
    return _call_from_function_block(text) or _call_from_json(text)


def parse_markup_tool_calls(text: str) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
    """``(calls, residue)`` from ``text``.

    Every ``<tool_call>…</tool_call>`` envelope that parses contributes one ``(name, arguments)``
    pair and is removed from the residue; an envelope that does not parse stays in the residue, so
    a caller can see there was markup it did not understand. An unterminated envelope is left alone.
    """

    calls: List[Tuple[str, Dict[str, Any]]] = []
    residue: List[str] = []
    cursor = 0
    for envelope in _ENVELOPE_RE.finditer(text):
        call = _call_from_block(envelope.group(1))
        if call is None:
            continue
        calls.append(call)
        residue.append(text[cursor:envelope.start()])
        cursor = envelope.end()
    if not calls:
        call = _call_from_function_block(text)
        return ([call], "") if call is not None else ([], text)
    residue.append(text[cursor:])
    return calls, "".join(residue)


def _route(name: str) -> Optional[str]:
    """``"bridge"`` for a name this session defers, ``"direct"`` for another registered tool, None
    for an unregistered one."""

    try:
        if _tool_search.is_deferrable_tool_name(name, _tool_search.load_config_readonly().effective_defer_tools):
            return "bridge"
    except Exception:  # noqa: BLE001 - a config read that fails must not block a recovery
        pass
    return "direct" if registry.get_schema(name) else None


def _arguments_json(arguments: Dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _tool_call(name: str, arguments: Dict[str, Any]) -> Optional[ToolCall]:
    """The call to run: the bridge for a deferred name, the tool itself otherwise."""

    route = _route(name)
    if route == "bridge":
        bridge_arguments = _arguments_json({"calls": [{"name": name, "arguments": arguments}]})
        return ToolCall(id=deterministic_call_id(TOOL_CALL_NAME, bridge_arguments),
                        name=TOOL_CALL_NAME, arguments=bridge_arguments)
    if route == "direct":
        direct_arguments = _arguments_json(arguments)
        return ToolCall(id=deterministic_call_id(name, direct_arguments),
                        name=name, arguments=direct_arguments)
    return None


def recover_markup_tool_calls(
    content: Optional[str], *, strip_visible: Callable[[str], str],
) -> Optional[Tuple[List[ToolCall], str]]:
    """``(tool_calls, residue)`` recovered from a markup-only reply, else None.

    None — the reply is left exactly as it was — when the markup does not parse, names no tool this
    session can reach, or sits beside visible text (``strip_visible``, normally the agent's
    ``_strip_think_blocks``, must come back empty for the markup to be the whole answer).
    """

    text = content or ""
    if "<" not in text:
        return None
    parsed, residue = parse_markup_tool_calls(text)
    if not parsed or strip_visible(residue).strip():
        return None
    calls = [call for call in (_tool_call(name, arguments) for name, arguments in parsed) if call is not None]
    if not calls:
        return None
    logger.info(
        "Recovered %d tool call(s) shipped as markup instead of tool_calls: %s",
        len(calls), ", ".join(call.function.name for call in calls),
    )
    return calls, ""


__all__ = ["parse_markup_tool_calls", "recover_markup_tool_calls"]
