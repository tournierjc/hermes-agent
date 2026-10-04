"""Stateful scrubber for tool-call markup in streamed assistant text.

A server that hands a model's call through as *content* instead of a ``tool_calls`` payload ships
chat-template markup as ordinary text — ``<tool_call><function=NAME><parameter=K>v</parameter>
</function></tool_call>``, or a ``<function=NAME>`` block, or a ``<tool_calls>``/``<tool_result>``
envelope. The final-response stripper (``agent_runtime_helpers.strip_think_blocks``) removes those
blocks, but the live stream reaches the UI delta by delta first, and ``StreamingThinkScrubber``
hides *reasoning* tags only: the user watches the raw call scroll past and then sees the turn end
without the action.

This scrubber hides the same markup in the live stream. Like the think scrubber it holds partial
tags at delta boundaries, so a block split across deltas cannot leak half of itself; ``flush()``
releases a benign tail (an innocent ``<`` that never became a tag) and drops an unterminated block
(leaking a truncated call is worse than a truncated answer). The ``<function name=…>`` attribute
form is boundary-gated exactly like the stripper's pattern, so prose that *mentions* a call
survives; the ``<function=…>`` and ``<tool_call…>`` forms are unambiguous and hidden wherever they
appear.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

__all__ = ["StreamingToolCallScrubber", "TOOL_CALL_TAG_NAMES"]

# The one list of tool-call envelope names, mirroring ``_TOOL_CALL_TAG_NAMES`` in
# ``agent_runtime_helpers`` so the stream hides what the final-response stripper removes.
TOOL_CALL_TAG_NAMES: Tuple[str, ...] = (
    "tool_call", "tool_calls", "tool_result", "function_call", "function_calls",
)
_NS_PREFIX = r"(?:[\w.-]+:)?"
_ENVELOPE_ALTS = "|".join(TOOL_CALL_TAG_NAMES)
_OPEN_RE = re.compile(rf"<({_NS_PREFIX}(?:{_ENVELOPE_ALTS}))\b[^>]*>", re.IGNORECASE)
_FUNCTION_EQ_RE = re.compile(r"<function\s*=\s*[A-Za-z_][\w.:-]*\s*>", re.IGNORECASE)
_FUNCTION_ATTR_RE = re.compile(r"<function\b[^>]*\bname\s*=[^>]*>", re.IGNORECASE)
_CLOSE_RE = re.compile(rf"</({_NS_PREFIX}(?:{_ENVELOPE_ALTS})|function)\s*>", re.IGNORECASE)
# Block-boundary predecessor for the attribute form (same gate as the stripper's pattern).
_BOUNDARY_PRECEDERS = "\n\r.!?:"


class StreamingToolCallScrubber:
    """Stateful scrubber for streaming tool-call markup.

    State: ``_block`` (lowercased envelope name whose close tag is being awaited, or None),
    ``_buf`` (held-back partial-tag tail), ``_last_emitted_ended_newline`` (True iff nothing was
    emitted yet or the last emission ended with a newline — decides whether an open tag at buffer
    position 0 sits at a block boundary for the gated ``<function name=…>`` form).
    """

    def __init__(self) -> None:
        self._closers: Dict[str, re.Pattern[str]] = {}
        self.reset()

    def reset(self) -> None:
        """Reset all state.  Call at the top of every new turn."""
        self._block: Optional[str] = None
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True
        # Markup the most recent feed() stripped (call scaffolding: the caller drops it).
        self.last_hidden: str = ""

    def feed(self, text: str) -> str:
        """Feed one delta; return the visible portion ("" when it is all markup or held back)."""
        self.last_hidden = ""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: List[str] = []
        hidden: List[str] = []

        while buf:
            if self._block is not None:
                close = self._closer(self._block).search(buf)
                if close is None:
                    # No close yet: hold a partial close-tag prefix; the rest is markup.
                    hidden.append(self._hold_partial(buf))
                    break
                hidden.append(buf[:close.start()])
                buf = buf[close.end():]
                self._block = None
                continue

            opener = self._find_open(buf, out)
            orphan = _CLOSE_RE.search(buf)
            if opener is not None and (orphan is None or opener[0] <= orphan.start()):
                start, end, name = opener
                self._emit(out, buf[:start])
                hidden.append(buf[start:end])
                self._block = name
                buf = buf[end:]
                continue
            if orphan is not None:
                self._emit(out, buf[:orphan.start()])
                hidden.append(orphan.group(0))
                buf = buf[orphan.end():]
                continue

            self._emit(out, self._hold_partial(buf))
            break

        self.last_hidden = "".join(hidden)
        return "".join(out)

    def flush(self) -> str:
        """End-of-stream flush: inside an unterminated block the held-back markup is discarded,
        otherwise a benign partial-tag tail is emitted verbatim."""
        tail = "" if self._block is not None else self._buf
        self._buf = ""
        self._block = None
        self._last_emitted_ended_newline = True
        return tail

    # ── internal helpers ───────────────────────────────────────────────

    def _closer(self, name: str) -> re.Pattern[str]:
        pattern = self._closers.get(name)
        if pattern is None:
            pattern = re.compile(rf"</{_NS_PREFIX}{re.escape(name)}\s*>", re.IGNORECASE)
            self._closers[name] = pattern
        return pattern

    def _find_open(self, buf: str, emitted: List[str]) -> Optional[Tuple[int, int, str]]:
        """(start, end, close_name) of the earliest open tag, or None."""
        hits: List[Tuple[int, int, str]] = []
        envelope = _OPEN_RE.search(buf)
        if envelope is not None:
            hits.append((envelope.start(), envelope.end(), envelope.group(1).rsplit(":", 1)[-1].lower()))
        bare = _FUNCTION_EQ_RE.search(buf)
        if bare is not None:
            hits.append((bare.start(), bare.end(), "function"))
        attribute = self._find_attribute_open(buf, emitted)
        if attribute is not None:
            hits.append(attribute)
        return min(hits) if hits else None

    def _find_attribute_open(self, buf: str, emitted: List[str]) -> Optional[Tuple[int, int, str]]:
        """The boundary-gated ``<function name=…>`` form, or None."""
        match = _FUNCTION_ATTR_RE.search(buf)
        while match is not None and not self._at_boundary(buf, match.start(), emitted):
            match = _FUNCTION_ATTR_RE.search(buf, match.start() + 1)
        return (match.start(), match.end(), "function") if match is not None else None

    def _at_boundary(self, buf: str, idx: int, emitted: List[str]) -> bool:
        """True iff *idx* is a block boundary: start of the stream/turn, or after a newline or
        sentence punctuation with only blanks between."""
        head = buf[:idx]
        if not head.strip():
            return emitted[-1].endswith("\n") if emitted else self._last_emitted_ended_newline
        return head.rstrip(" \t")[-1] in _BOUNDARY_PRECEDERS

    def _emit(self, out: List[str], text: str) -> None:
        if text:
            out.append(text)
            self._last_emitted_ended_newline = text.endswith("\n")

    def _hold_partial(self, buf: str) -> str:
        """Move a trailing partial-tag prefix of *buf* into ``_buf``; return the remainder."""
        idx = buf.rfind("<")
        if idx == -1:
            return buf
        tail = buf[idx:]
        if ">" in tail or not re.fullmatch(r"</?[A-Za-z_/]?[\w.:=-]*", tail):
            return buf
        self._buf = tail
        return buf[:idx]
