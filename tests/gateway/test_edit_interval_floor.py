"""Adapter-declared edit-pacing floors (``MIN_PROGRESS_EDIT_INTERVAL`` / ``MIN_STREAM_EDIT_INTERVAL``).

Contract: an adapter may only SLOW the gateway's tool-progress and streaming edits; adapters
that declare nothing (every platform but Teams today), duck-typed fakes and MagicMocks keep
the gateway defaults byte-for-byte.
"""

import asyncio
import queue
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, edit_interval_floor
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


class _EditingAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent, self.edits = [], []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False) -> SendResult:
        self.edits.append(content)
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class _SlowEditingAdapter(_EditingAdapter):
    MIN_PROGRESS_EDIT_INTERVAL = 4.0
    MIN_STREAM_EDIT_INTERVAL = 2.5


class TestEditIntervalFloor:
    def test_only_positive_numbers_count(self):
        assert edit_interval_floor(_EditingAdapter(), "MIN_PROGRESS_EDIT_INTERVAL") == 0.0
        assert edit_interval_floor(_SlowEditingAdapter(), "MIN_PROGRESS_EDIT_INTERVAL") == 4.0
        assert edit_interval_floor(MagicMock(), "MIN_STREAM_EDIT_INTERVAL") == 0.0
        assert edit_interval_floor(object(), "MIN_STREAM_EDIT_INTERVAL") == 0.0
        for bogus in (True, -3, float("nan"), "5"):
            assert edit_interval_floor(SimpleNamespace(MIN_STREAM_EDIT_INTERVAL=bogus),
                                       "MIN_STREAM_EDIT_INTERVAL") == 0.0


def _build_stream_cfg(adapter, scfg):
    from gateway.run_turn import GatewayTurnMixin
    source = SessionSource(platform=adapter.platform, chat_id="chat-1", chat_type="group")
    cfg, _ = GatewayTurnMixin._build_stream_consumer_config(
        SimpleNamespace(), source, scfg, adapter, on_missing_cursor="fallback")
    return cfg


class TestStreamConsumerConfigFloor:
    def test_adapter_without_floor_keeps_configured_pacing(self):
        scfg = StreamingConfig(edit_interval=0.8, buffer_threshold=24)
        cfg = _build_stream_cfg(_EditingAdapter(), scfg)
        assert (cfg.edit_interval, cfg.buffer_threshold) == (scfg.edit_interval, scfg.buffer_threshold)

    def test_floor_slows_edits_and_disables_size_trigger(self):
        adapter = _SlowEditingAdapter()
        scfg = StreamingConfig(edit_interval=0.8, buffer_threshold=24)
        cfg = _build_stream_cfg(adapter, scfg)
        assert cfg.edit_interval == adapter.MIN_STREAM_EDIT_INTERVAL
        # Interval-only pacing: no realistic preview reaches the size trigger.
        assert cfg.buffer_threshold >= sys.maxsize

    def test_slower_user_interval_wins_over_floor(self):
        adapter = _SlowEditingAdapter()
        scfg = StreamingConfig(edit_interval=adapter.MIN_STREAM_EDIT_INTERVAL + 3, buffer_threshold=24)
        assert _build_stream_cfg(adapter, scfg).edit_interval == scfg.edit_interval


async def _progress_throttle_sleeps(monkeypatch, adapter) -> list:
    """Run TurnRunner.send_progress_messages over two queued tool lines and return the
    throttle sleep it requested before editing in the second line (the run ends there)."""
    import gateway.run_turn_runner as run_turn_runner
    from gateway.run_turn_runner import TurnRunner

    requested = []

    async def _fake_sleep(delay, *args, **kwargs):
        if delay not in (0, 0.3):  # skip the idle-poll / typing-restore sleeps
            requested.append(delay)

    fake_asyncio = types.SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("__")})
    fake_asyncio.sleep = _fake_sleep
    monkeypatch.setattr(run_turn_runner, "asyncio", fake_asyncio)

    q = queue.Queue()
    q.put("🖥️ terminal: pwd")
    q.put("🌐 web_search: hermes")
    ctx = TurnContext(
        progress_queue=q,
        source=SessionSource(platform=adapter.platform, chat_id="chat-1"),
        _run_still_current=lambda: not requested,
    )
    runner = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda source: adapter), ctx)
    await runner.send_progress_messages()
    assert adapter.sent, "first tool line opens the progress bubble"
    return requested


class TestProgressEditFloor:
    @pytest.mark.anyio
    async def test_default_adapter_keeps_gateway_interval(self, monkeypatch):
        sleeps = await _progress_throttle_sleeps(monkeypatch, _EditingAdapter())
        assert sleeps and max(sleeps) <= 1.5

    @pytest.mark.anyio
    async def test_adapter_floor_spaces_progress_edits(self, monkeypatch):
        adapter = _SlowEditingAdapter()
        sleeps = await _progress_throttle_sleeps(monkeypatch, adapter)
        assert sleeps and max(sleeps) > 1.5
        assert max(sleeps) <= adapter.MIN_PROGRESS_EDIT_INTERVAL
