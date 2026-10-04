"""Regression tests for recovering a tool call a server shipped as markup.

Observed on a local OpenAI-compatible endpoint (TensorFold): when the model's call named a tool the
request did not offer — a deferred tool behind ``tool_search``, an MCP tool the session folds out —
the server returned the call as *content* (``<tool_call><function=…></function></tool_call>``) with
``finish_reason=stop`` and no ``tool_calls``. The client strips that markup as scaffolding, so the
turn read as an empty response: three retries, then a ``(empty)`` turn with the requested action
never run. These tests pin the recovery: the call runs (or comes back as the bridge's own
corrective error), and ordinary replies are untouched.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.markup_tool_calls import parse_markup_tool_calls, recover_markup_tool_calls
from tools.tool_search_catalog import TOOL_CALL_NAME

FUNCTION_ENVELOPE = (
    "<tool_call>\n"
    "<function=process_manage>\n"
    "<parameter=action>\npoll\n</parameter>\n"
    "<parameter=session_id>\nproc_df140a629824\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def _strip_visible(text: str) -> str:
    """Stand-in for ``agent._strip_think_blocks``: no think tags in these fixtures."""

    return text


def _recover(content: str):
    return recover_markup_tool_calls(content, strip_visible=_strip_visible)


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_run_agent's fixture)."""

    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


@pytest.fixture()
def routing(monkeypatch):
    """Route names deterministically instead of reading the developer's own config:
    ``process_manage`` is deferred (bridge), ``terminal`` is a plain registered tool (direct),
    everything else is unknown (not recovered)."""

    from agent import markup_tool_calls

    monkeypatch.setattr(
        markup_tool_calls._tool_search, "is_deferrable_tool_name",
        lambda name, defer_tools=None: name == "process_manage",
    )
    monkeypatch.setattr(
        markup_tool_calls.registry, "get_schema",
        lambda name: {"name": name} if name == "terminal" else None,
    )
    return {"process_manage"}


# ===================================================================
# Parsing — pure
# ===================================================================


class TestParseMarkupToolCalls:
    def test_function_parameter_envelope(self):
        calls, residue = parse_markup_tool_calls(FUNCTION_ENVELOPE)

        assert calls == [("process_manage", {"action": "poll", "session_id": "proc_df140a629824"})]
        assert residue.strip() == ""

    def test_json_envelope(self):
        calls, residue = parse_markup_tool_calls(
            '<tool_call>{"name": "web_search", "arguments": {"query": "hermes"}}</tool_call>'
        )

        assert calls == [("web_search", {"query": "hermes"})]
        assert residue.strip() == ""

    def test_json_envelope_with_function_container(self):
        calls, _ = parse_markup_tool_calls(
            '<tool_call>{"function": {"name": "read_file", "arguments": {"path": "/tmp/x"}}}</tool_call>'
        )

        assert calls == [("read_file", {"path": "/tmp/x"})]

    def test_namespaced_envelope(self):
        calls, _ = parse_markup_tool_calls(
            "<ns:tool_call><function=terminal><parameter=command>ls</parameter></function></ns:tool_call>"
        )

        assert calls == [("terminal", {"command": "ls"})]

    def test_lone_function_block(self):
        calls, residue = parse_markup_tool_calls(
            "<function=terminal>\n<parameter=command>\nls -la\n</parameter>\n</function>"
        )

        assert calls == [("terminal", {"command": "ls -la"})]
        assert residue == ""

    def test_several_envelopes_in_one_reply(self):
        calls, residue = parse_markup_tool_calls(f"{FUNCTION_ENVELOPE}\n{FUNCTION_ENVELOPE}")

        assert len(calls) == 2
        assert residue.strip() == ""

    def test_prose_beside_the_block_is_kept(self):
        calls, residue = parse_markup_tool_calls(f"Let me check that for you.\n{FUNCTION_ENVELOPE}")

        assert calls == [("process_manage", {"action": "poll", "session_id": "proc_df140a629824"})]
        assert "Let me check that for you." in residue
        assert "<tool_call>" not in residue

    def test_unterminated_envelope_is_not_a_call(self):
        calls, residue = parse_markup_tool_calls("<tool_call><function=terminal><parameter=command>ls")

        assert calls == []
        assert residue == "<tool_call><function=terminal><parameter=command>ls"

    def test_envelope_without_a_parsable_body_stays_in_the_residue(self):
        calls, residue = parse_markup_tool_calls("<tool_call>hmm, not a call</tool_call>")

        assert calls == []
        assert residue == "<tool_call>hmm, not a call</tool_call>"

    def test_plain_text_is_untouched(self):
        calls, residue = parse_markup_tool_calls("Here is your answer, no markup at all.")

        assert calls == []
        assert residue == "Here is your answer, no markup at all."


# ===================================================================
# Routing and the recovery contract
# ===================================================================


class TestRecoverMarkupToolCalls:
    def test_deferred_name_goes_through_the_bridge(self, routing):
        recovered = _recover(FUNCTION_ENVELOPE)

        assert recovered is not None
        calls, residue = recovered
        assert [call.function.name for call in calls] == [TOOL_CALL_NAME]
        assert json.loads(calls[0].function.arguments) == {
            "calls": [
                {
                    "name": "process_manage",
                    "arguments": {"action": "poll", "session_id": "proc_df140a629824"},
                }
            ]
        }
        assert residue == ""

    def test_registered_ordinary_tool_is_called_directly(self, routing):
        recovered = _recover(
            "<tool_call><function=terminal><parameter=command>ls</parameter></function></tool_call>"
        )

        assert recovered is not None
        calls, _ = recovered
        assert [call.function.name for call in calls] == ["terminal"]
        assert json.loads(calls[0].function.arguments) == {"command": "ls"}

    def test_unknown_name_is_not_recovered(self, routing):
        assert _recover("<tool_call><function=no_such_tool_here></function></tool_call>") is None

    def test_markup_beside_prose_is_not_recovered(self, routing):
        assert _recover(f"Working on it.\n{FUNCTION_ENVELOPE}") is None

    def test_plain_text_is_not_recovered(self, routing):
        assert _recover("All done — nothing to run.") is None

    def test_ids_are_deterministic(self, routing):
        """A random id would break the prompt-cache prefix on every later request."""

        first = _recover(FUNCTION_ENVELOPE)
        second = _recover(FUNCTION_ENVELOPE)

        assert first[0][0].id == second[0][0].id

    def test_the_shipped_defer_list_still_defers_process_manage(self):
        """The routing premise: the name in the incident is deferred by the shipped config."""

        from tools import tool_search

        defer_tools = tool_search.load_config_readonly().effective_defer_tools
        assert tool_search.is_deferrable_tool_name("process_manage", defer_tools)


# ===================================================================
# The turn loop — an action request must not end as an empty reply
# ===================================================================


class TestMarkupRecoveryInTheTurnLoop:
    def test_markup_only_reply_runs_the_call(self, loop_agent, routing):
        from tests.agent.test_run_agent import _mock_response

        loop_agent.valid_tool_names = {TOOL_CALL_NAME, "web_search"}
        executed = []
        loop_agent._execute_tool_calls = lambda assistant_message, *a, **k: executed.append(assistant_message)
        loop_agent._flush_messages_to_session_db = lambda *a, **k: True
        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content=FUNCTION_ENVELOPE, finish_reason="stop"),
            _mock_response(content="The job is still running.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("is the background job done?")

        assert len(executed) == 1, "the recovered call must reach the tool round"
        call = executed[0].tool_calls[0]
        assert call.function.name == TOOL_CALL_NAME
        assert json.loads(call.function.arguments)["calls"][0]["name"] == "process_manage"
        assert "(empty)" not in (result["final_response"] or "")
        assert "still running" in (result["final_response"] or "")

    def test_ordinary_reply_is_untouched(self, loop_agent, routing):
        from tests.agent.test_run_agent import _mock_response

        loop_agent.valid_tool_names = {TOOL_CALL_NAME, "web_search"}
        executed = []
        loop_agent._execute_tool_calls = lambda assistant_message, *a, **k: executed.append(assistant_message)
        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="Here is your answer.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hello")

        assert loop_agent.client.chat.completions.create.call_count == 1
        assert executed == []
        assert "Here is your answer." in result["final_response"]
