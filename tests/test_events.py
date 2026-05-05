"""Tests for EventBus pub/sub mechanics."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from visionvoiceasist.events import EventBus, EventType
from visionvoiceasist.types import SpeechEvent


class TestEventBus:
    def test_subscribe_and_publish(self, bus: EventBus) -> None:
        received: list[object] = []
        bus.subscribe(EventType.SPEECH, received.append)
        ev = SpeechEvent(text="test")
        bus.publish(EventType.SPEECH, ev)
        assert received == [ev]

    def test_multiple_subscribers(self, bus: EventBus) -> None:
        calls: list[int] = []
        bus.subscribe(EventType.SHUTDOWN, lambda _: calls.append(1))
        bus.subscribe(EventType.SHUTDOWN, lambda _: calls.append(2))
        bus.publish(EventType.SHUTDOWN, None)
        assert sorted(calls) == [1, 2]

    def test_subscribe_idempotent(self, bus: EventBus) -> None:
        cb = MagicMock()
        bus.subscribe(EventType.SPEECH, cb)
        bus.subscribe(EventType.SPEECH, cb)
        bus.publish(EventType.SPEECH, SpeechEvent("hi"))
        assert cb.call_count == 1

    def test_unsubscribe(self, bus: EventBus) -> None:
        received: list[object] = []
        bus.subscribe(EventType.SPEECH, received.append)
        bus.unsubscribe(EventType.SPEECH, received.append)
        bus.publish(EventType.SPEECH, SpeechEvent("hi"))
        assert received == []

    def test_unsubscribe_nonexistent_is_safe(self, bus: EventBus) -> None:
        bus.unsubscribe(EventType.SPEECH, lambda x: None)

    def test_subscriber_exception_does_not_prevent_others(
        self, bus: EventBus
    ) -> None:
        good_calls: list[str] = []

        def bad_subscriber(_: object) -> None:
            raise RuntimeError("boom")

        bus.subscribe(EventType.OCR_TEXT, bad_subscriber)
        bus.subscribe(EventType.OCR_TEXT, lambda _: good_calls.append("ok"))
        bus.publish(EventType.OCR_TEXT, "some text")
        assert good_calls == ["ok"]

    def test_publish_no_subscribers_is_safe(self, bus: EventBus) -> None:
        bus.publish(EventType.DETECTIONS, [])

    def test_clear(self, bus: EventBus) -> None:
        received: list[object] = []
        bus.subscribe(EventType.SPEECH, received.append)
        bus.clear()
        bus.publish(EventType.SPEECH, SpeechEvent("hi"))
        assert received == []

    def test_thread_safety(self, bus: EventBus) -> None:
        counter: list[int] = []
        bus.subscribe(EventType.SPEECH, lambda _: counter.append(1))

        def publisher() -> None:
            for _ in range(50):
                bus.publish(EventType.SPEECH, SpeechEvent("ping"))

        threads = [threading.Thread(target=publisher) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(counter) == 200

    def test_publish_different_event_types_isolated(self, bus: EventBus) -> None:
        speech_calls: list[object] = []
        health_calls: list[object] = []
        bus.subscribe(EventType.SPEECH, speech_calls.append)
        bus.subscribe(EventType.HEALTH, health_calls.append)
        bus.publish(EventType.SPEECH, SpeechEvent("s"))
        assert len(speech_calls) == 1
        assert len(health_calls) == 0
