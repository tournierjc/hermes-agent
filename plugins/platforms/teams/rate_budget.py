"""Per-conversation outbound budget for the Teams adapter.

Bot Framework meters a bot per conversation, and a Teams channel is ONE conversation for
all of its threads (``19:…@thread.tacv2;messageid=<root>`` shares the channel's quota).
Every outbound activity counts: messages, typing indicators and activity updates (edits).
Published "per agent per thread" limits for send-to-conversation (Microsoft Learn, "Rate
limiting for agents", learn.microsoft.com/microsoftteams/platform/bots/how-to/rate-limit):
7 / 1 s, 8 / 2 s, 60 / 30 s, 1800 / 3600 s.

One sliding-window log per base conversation id backs three priorities:

* ``typing`` — the first thing dropped when the conversation gets busy.
* ``edit`` — intermediate streaming / progress-bubble edits; dropped next (the following
  edit carries the newer text anyway).
* essential — message sends (every chunk), card / media sends and ``finalize=True`` edits.
  Never dropped: :meth:`ConversationRateBudget.acquire` waits for room under the
  :data:`HARD_LIMITS` headroom, bounded by ``max_wait``, then lets the call through.

After an HTTP 429 the conversation is paused (``Retry-After`` + jitter): non-essential
calls are refused and essential calls wait out the pause (still bounded).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import OrderedDict, deque
from typing import Awaitable, Callable, Deque, Dict, Optional, Tuple

Limits = Tuple[Tuple[float, int], ...]

# Microsoft's per-bot, per-conversation limits: (window seconds, max activities).
TEAMS_CONVERSATION_LIMITS: Limits = ((1.0, 7), (2.0, 8), (30.0, 60), (3600.0, 1800))
# Essential calls wait for room under these (~15% headroom for other senders on the same
# conversation, e.g. cron's out-of-process standalone send, and for clock skew).
HARD_LIMITS: Limits = ((1.0, 6), (2.0, 7), (30.0, 50), (3600.0, 1500))
# Non-essential tiers stop earlier so there is always room left for the answer itself.
TIER_LIMITS: Dict[str, Limits] = {
    "typing": ((1.0, 4), (2.0, 5), (30.0, 30), (3600.0, 1000)),
    "edit": ((1.0, 5), (2.0, 6), (30.0, 40), (3600.0, 1350)),
}
_LOG_WINDOW_SECS = max(window for window, _ in HARD_LIMITS)
_LOG_MAX = max(limit for _, limit in HARD_LIMITS) + 100
_DEFAULT_THROTTLE_PAUSE_SECS = 2.0
_THROTTLE_JITTER_SECS = 0.5


def conversation_key(chat_id: str) -> str:
    """Budget key: the flat conversation id (``;messageid=`` thread suffix stripped)."""
    raw = str(chat_id or "").strip()
    idx = raw.lower().find(";messageid=")
    return raw[:idx] if idx != -1 else raw


class ConversationRateBudget:
    """Sliding-window activity log per Teams conversation (see module docstring)."""

    def __init__(
        self,
        *,
        max_wait: float = 20.0,
        max_conversations: int = 500,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.max_wait = max_wait
        self._max_conversations = max_conversations
        self._clock = clock
        self._sleep = sleep
        self._log: "OrderedDict[str, Deque[float]]" = OrderedDict()
        self._paused_until: Dict[str, float] = {}

    def _wait_for(self, key: str, limits: Limits, now: float) -> float:
        """Seconds until every ``(window, limit)`` has room (0.0 = go now)."""
        wait = max(0.0, self._paused_until.get(key, 0.0) - now)
        log = self._log.get(key)
        if log:
            while log and now - log[0] >= _LOG_WINDOW_SECS:
                log.popleft()
            for window, limit in limits:
                if len(log) >= limit and now - log[-limit] < window:
                    wait = max(wait, log[-limit] + window - now)
        return wait

    def allows(self, chat_id: str, tier: str) -> bool:
        """True when a non-essential call of ``tier`` (``typing`` / ``edit``) may fire now."""
        return self._wait_for(conversation_key(chat_id), TIER_LIMITS[tier], self._clock()) <= 0.0

    def record(self, chat_id: str) -> None:
        """Count one outbound activity against the conversation."""
        key = conversation_key(chat_id)
        log = self._log.get(key)
        if log is None:
            log = self._log[key] = deque(maxlen=_LOG_MAX)
            while len(self._log) > self._max_conversations:
                evicted, _ = self._log.popitem(last=False)
                self._paused_until.pop(evicted, None)
        else:
            self._log.move_to_end(key)
        log.append(self._clock())

    async def acquire(self, chat_id: str) -> float:
        """Essential call: wait for room under :data:`HARD_LIMITS` (at most ``max_wait``),
        then record it. Never refuses. Returns the seconds waited."""
        key = conversation_key(chat_id)
        start = self._clock()
        deadline = start + self.max_wait
        while True:
            now = self._clock()
            wait = self._wait_for(key, HARD_LIMITS, now)
            if wait <= 0.0 or now >= deadline:
                break
            await self._sleep(min(wait, deadline - now) + 0.01)
        self.record(chat_id)
        return self._clock() - start

    def note_throttled(self, chat_id: str, retry_after: Optional[float]) -> float:
        """The service answered 429: pause this conversation. Returns the pause length."""
        key = conversation_key(chat_id)
        pause = _DEFAULT_THROTTLE_PAUSE_SECS if retry_after is None else max(0.0, float(retry_after))
        pause += random.uniform(0.0, _THROTTLE_JITTER_SECS)
        self._paused_until[key] = max(self._paused_until.get(key, 0.0), self._clock() + pause)
        return pause
