"""Tests for the streaming tool-call scrubber.

Observed: an OpenAI-compatible server hands a model's call through as *content* when the call names a
tool the request did not offer (a deferred tool behind tool_search, an MCP tool the session folds
out). The final-response stripper removes that markup, but the live stream reaches the UI delta by
delta first — and ``StreamingThinkScrubber`` hides reasoning tags only, so the user watched the raw
call scroll past. These tests pin the stream side: markup never reaches the UI, a block split across
deltas cannot leak half of itself, and prose that *mentions* a tag still does.
"""

from __future__ import annotations

from agent.agent_runtime_helpers import strip_think_blocks
from agent.tool_call_scrubber import StreamingToolCallScrubber

ENVELOPE = (
    "<tool_call>\n<function=process_manage>\n<parameter=action>\nwait\n</parameter>\n"
    "<parameter=session_id>\nproc_d755c36c5ade\n</parameter>\n</function>\n</tool_call>"
)
BARE_FUNCTION = (
    "<function=process_manage><parameter=action>wait</parameter>"
    "<parameter=session_id>proc_d755c36c5ade</parameter></function>"
)


def _stream(deltas):
    scrubber = StreamingToolCallScrubber()
    out = "".join(scrubber.feed(delta) for delta in deltas)
    return out + scrubber.flush()


class TestStreamingToolCallScrubber:
    def test_envelope_in_one_delta_is_hidden(self):
        assert _stream([ENVELOPE]) == ""

    def test_envelope_split_across_deltas_is_hidden(self):
        deltas = ["<tool_call>\n<function=process_manage>\n", "<parameter=action>\nwait\n</parameter>\n",
                  "<parameter=session_id>\nproc_d755c36c5ade\n</parameter>\n", "</function>\n</tool_call>"]

        assert _stream(deltas) == ""

    def test_markup_split_mid_tag_is_hidden(self):
        """A tag boundary inside a delta must not leak either half."""
        deltas = ["<tool_ca", "ll><function=te", "rminal><parameter=command>ls</parameter>",
                  "</function></tool_call>"]

        assert _stream(deltas) == ""

    def test_bare_function_block_is_hidden(self):
        assert _stream([BARE_FUNCTION]) == ""

    def test_prose_before_and_after_the_block_survives(self):
        out = _stream(["Let me check.\n", ENVELOPE, "\nDone."])

        assert "Let me check." in out
        assert "Done." in out
        assert "<tool_call" not in out and "function=" not in out

    def test_attribute_form_at_a_boundary_is_hidden(self):
        out = _stream(["Checking.\n", '<function name="terminal">', "ls", "</function>"])

        assert out.strip() == "Checking."

    def test_prose_mentioning_the_attribute_form_survives(self):
        text = "Write <function name=x> in that template to declare it."

        assert _stream([text]) == text

    def test_partial_tag_at_stream_end_is_flushed(self):
        """An innocent '<' that never became a tag must still reach the UI."""
        assert _stream(["5 < 3 and no tag here"]) == "5 < 3 and no tag here"

    def test_unterminated_block_is_dropped_at_flush(self):
        assert _stream(["Checking.\n<tool_call><function=terminal><parameter=command>ls"]) == "Checking.\n"

    def test_orphan_closer_is_dropped(self):
        assert _stream(["Done.</tool_call>"]) == "Done."

    def test_reset_clears_mid_block_state(self):
        scrubber = StreamingToolCallScrubber()
        scrubber.feed("<tool_call><function=terminal>")
        scrubber.reset()

        assert scrubber.feed("plain answer") == "plain answer"

    def test_stream_output_is_already_clean_for_the_final_stripper(self):
        """Contract between the layers: whatever the stream delivered, the final-response stripper
        has nothing left to remove — so the delivered text and the stored text agree."""
        text = f"Let me check that for you.\n\n{ENVELOPE}\n"
        visible = _stream([text])

        assert strip_think_blocks(None, visible) == visible
        assert "tool_call" not in visible
