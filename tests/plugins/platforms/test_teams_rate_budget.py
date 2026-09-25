"""Per-conversation outbound budget for the Teams adapter (plugins/platforms/teams/rate_budget.py)."""

from __future__ import annotations

import random

import pytest

from plugins.platforms.teams.rate_budget import (
    HARD_LIMITS,
    TEAMS_CONVERSATION_LIMITS,
    TIER_LIMITS,
    ConversationRateBudget,
    conversation_key,
)


class FakeClock:
    """Monotonic clock whose ``sleep`` advances time instantly."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        self.now += delay


def _budget(clock: FakeClock, **kwargs) -> ConversationRateBudget:
    return ConversationRateBudget(clock=clock, sleep=clock.sleep, **kwargs)


def _max_in_window(stamps: list[float], window: float) -> int:
    stamps = sorted(stamps)
    best, lo = 0, 0
    for hi, t in enumerate(stamps):
        while t - stamps[lo] >= window:
            lo += 1
        best = max(best, hi - lo + 1)
    return best


def test_tiers_nest_under_the_published_teams_limits():
    """typing < edit < essential headroom < Microsoft's limit, for every window."""
    windows = [w for w, _ in TEAMS_CONVERSATION_LIMITS]
    for limits in (HARD_LIMITS, TIER_LIMITS["edit"], TIER_LIMITS["typing"]):
        assert [w for w, _ in limits] == windows
    for (_, teams), (_, hard), (_, edit), (_, typing) in zip(
            TEAMS_CONVERSATION_LIMITS, HARD_LIMITS, TIER_LIMITS["edit"], TIER_LIMITS["typing"]):
        assert typing < edit < hard < teams


def test_thread_ids_share_the_channel_budget():
    assert conversation_key("19:c@thread.tacv2;messageid=1") == "19:c@thread.tacv2"
    assert conversation_key("19:c@thread.tacv2;messageId=2") == "19:c@thread.tacv2"
    clock = FakeClock()
    budget = _budget(clock)
    typing_cap = TIER_LIMITS["typing"][0][1]
    for _ in range(typing_cap):
        budget.record("19:c@thread.tacv2;messageid=1")
    assert not budget.allows("19:c@thread.tacv2;messageid=2", "typing")
    assert not budget.allows("19:c@thread.tacv2", "typing")
    assert budget.allows("19:other@thread.tacv2", "typing")


def test_typing_is_dropped_before_intermediate_edits():
    clock = FakeClock()
    budget = _budget(clock)
    chat = "a:1"
    while budget.allows(chat, "typing"):
        budget.record(chat)
    assert budget.allows(chat, "edit")
    while budget.allows(chat, "edit"):
        budget.record(chat)
    assert not budget.allows(chat, "typing")
    clock.now += 1.0  # the 1s window frees up again
    assert budget.allows(chat, "edit")


@pytest.mark.anyio
async def test_essential_waits_for_room_then_records():
    clock = FakeClock()
    budget = _budget(clock)
    per_second = HARD_LIMITS[0][1]
    for _ in range(per_second):
        budget.record("a:1")
    waited = await budget.acquire("a:1")
    assert 0.9 < waited < 1.1
    assert _max_in_window(list(budget._log["a:1"]), 1.0) <= per_second


@pytest.mark.anyio
async def test_essential_wait_is_bounded_and_never_refuses():
    clock = FakeClock()
    budget = _budget(clock, max_wait=5.0)
    budget.note_throttled("a:1", retry_after=120.0)
    assert not budget.allows("a:1", "edit") and not budget.allows("a:1", "typing")
    waited = await budget.acquire("a:1")
    assert 5.0 <= waited < 5.1
    assert len(budget._log["a:1"]) == 1  # it went anyway


@pytest.mark.anyio
async def test_throttle_pause_honours_retry_after(monkeypatch):
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
    clock = FakeClock()
    budget = _budget(clock)
    assert budget.note_throttled("a:1", 3.0) == 3.0
    assert not budget.allows("a:1", "edit")
    clock.now += 3.0
    assert budget.allows("a:1", "edit")
    assert budget.note_throttled("a:1", 0.0) == 0.0  # "retry now" is not a pause


@pytest.mark.anyio
async def test_busy_conversation_stays_under_teams_limits_and_keeps_every_essential():
    """Random mix of typing / interim edits / sends over an hour, far above what Teams allows:
    no essential call is dropped, the short windows stay under the headroom limits, and the
    hour stays under Microsoft's limit (essentials may dip into the headroom once
    non-essential traffic has used up its share of the hour)."""
    rng = random.Random(0)
    clock = FakeClock()
    budget = _budget(clock)
    recorded: list[float] = []
    real_record = budget.record

    def _record(chat_id):
        recorded.append(clock.now)
        real_record(chat_id)

    budget.record = _record
    chat = "19:c@thread.tacv2;messageid=7"
    essential = sent_essential = 0
    start = clock.now
    while clock.now - start < 3600.0:
        clock.now += rng.expovariate(2.0)  # ~2 attempted activities/s: well over 1800/h
        kind = rng.choices(("typing", "edit", "essential"), weights=(4, 4, 1))[0]
        if kind == "essential":
            essential += 1
            await budget.acquire(chat)
            sent_essential += 1
        elif budget.allows(chat, kind):
            budget.record(chat)
    assert sent_essential == essential
    for (window, hard), (_, teams) in zip(HARD_LIMITS, TEAMS_CONVERSATION_LIMITS):
        assert _max_in_window(recorded, window) <= (hard if window < 3600.0 else teams)


def test_conversation_logs_are_bounded():
    clock = FakeClock()
    budget = _budget(clock, max_conversations=3)
    for idx in range(10):
        budget.record(f"a:{idx}")
    assert len(budget._log) == 3
