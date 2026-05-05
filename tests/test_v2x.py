"""Tests for V2X MQTT client (message decode + dispatch — no real broker)."""

from __future__ import annotations

import pytest

from visionvoiceasist.events import EventBus, EventType
from visionvoiceasist.iot.v2x_client import V2xAlert, V2xClient, _decode_message
from visionvoiceasist.types import Priority


# ── Message decoder ───────────────────────────────────────────────────────────

class TestDecodeMessage:
    def test_emergency_with_all_fields(self) -> None:
        msg = {"type": "emergency", "vehicle": "ambulance", "distance_m": 100, "direction": "sağdan"}
        alert = _decode_message(msg)
        assert alert is not None
        assert alert.alert_type == "emergency"
        assert alert.priority == Priority.CRITICAL
        assert "ambulance" in alert.message_az
        assert "100" in alert.message_az

    def test_emergency_no_distance(self) -> None:
        alert = _decode_message({"type": "emergency", "vehicle": "fire truck"})
        assert alert is not None
        assert "fire truck" in alert.message_az

    def test_emergency_no_vehicle_uses_default(self) -> None:
        alert = _decode_message({"type": "emergency"})
        assert alert is not None
        assert "təcili nəqliyyat" in alert.message_az

    def test_red_traffic_signal_with_countdown(self) -> None:
        alert = _decode_message({"type": "traffic_signal", "phase": "red", "time_to_green_s": 15})
        assert alert is not None
        assert alert.priority == Priority.HIGH
        assert "15" in alert.message_az
        assert "Qırmızı" in alert.message_az

    def test_red_traffic_signal_no_countdown(self) -> None:
        alert = _decode_message({"type": "traffic_signal", "phase": "red"})
        assert alert is not None
        assert alert.priority == Priority.HIGH

    def test_green_traffic_signal(self) -> None:
        alert = _decode_message({"type": "traffic_signal", "phase": "green"})
        assert alert is not None
        assert alert.priority == Priority.NORMAL
        assert "Yaşıl" in alert.message_az

    def test_yellow_signal_returns_none(self) -> None:
        # Unknown phase — no alert defined
        assert _decode_message({"type": "traffic_signal", "phase": "yellow"}) is None

    def test_construction_custom_message(self) -> None:
        alert = _decode_message({"type": "construction", "message": "Körpü bağlıdır."})
        assert alert is not None
        assert alert.priority == Priority.HIGH
        assert "Körpü bağlıdır." in alert.message_az

    def test_construction_default_message(self) -> None:
        alert = _decode_message({"type": "construction"})
        assert alert is not None
        assert "Yol işləri" in alert.message_az

    def test_unknown_type_returns_none(self) -> None:
        assert _decode_message({"type": "weather_forecast"}) is None

    def test_empty_payload_returns_none(self) -> None:
        assert _decode_message({}) is None

    def test_raw_field_preserved(self) -> None:
        raw = {"type": "emergency", "vehicle": "ambulance"}
        alert = _decode_message(raw)
        assert alert is not None
        assert alert.raw is raw


# ── V2xAlert type ─────────────────────────────────────────────────────────────

class TestV2xAlert:
    def test_frozen(self) -> None:
        alert = V2xAlert("emergency", "test", Priority.CRITICAL, {})
        with pytest.raises(Exception):
            alert.priority = Priority.LOW  # type: ignore[misc]


# ── Client dispatch ───────────────────────────────────────────────────────────

class TestV2xClientDispatch:
    @pytest.fixture
    def bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def client(self, bus: EventBus) -> V2xClient:
        return V2xClient("localhost", 1883, bus)

    def test_dispatch_publishes_speech_and_v2x_events(
        self, bus: EventBus, client: V2xClient
    ) -> None:
        speech_events: list = []
        v2x_events: list = []
        bus.subscribe(EventType.SPEECH, speech_events.append)
        bus.subscribe(EventType.V2X_ALERT, v2x_events.append)

        client._dispatch(
            {"type": "emergency", "vehicle": "ambulance", "distance_m": 50, "direction": "önden"}
        )

        assert len(speech_events) == 1
        assert len(v2x_events) == 1
        assert speech_events[0].priority == Priority.CRITICAL

    def test_dispatch_unknown_payload_no_events(
        self, bus: EventBus, client: V2xClient
    ) -> None:
        received: list = []
        bus.subscribe(EventType.SPEECH, received.append)
        bus.subscribe(EventType.V2X_ALERT, received.append)

        client._dispatch({"type": "garbage_data"})

        assert len(received) == 0

    def test_dispatch_traffic_signal_high_priority(
        self, bus: EventBus, client: V2xClient
    ) -> None:
        received: list = []
        bus.subscribe(EventType.SPEECH, received.append)
        client._dispatch({"type": "traffic_signal", "phase": "red"})
        assert received[0].priority == Priority.HIGH

    def test_dispatch_construction_high_priority(
        self, bus: EventBus, client: V2xClient
    ) -> None:
        received: list = []
        bus.subscribe(EventType.SPEECH, received.append)
        client._dispatch({"type": "construction", "message": "Yol bağlıdır."})
        assert received[0].priority == Priority.HIGH

    def test_speech_event_has_force_flag(
        self, bus: EventBus, client: V2xClient
    ) -> None:
        received: list = []
        bus.subscribe(EventType.SPEECH, received.append)
        client._dispatch({"type": "traffic_signal", "phase": "green"})
        assert received[0].force is True


# ── Client lifecycle ──────────────────────────────────────────────────────────

class TestV2xClientLifecycle:
    def test_start_sets_running_flag(self) -> None:
        bus = EventBus()
        client = V2xClient("localhost", 1883, bus, sim_mode=True, sim_interval_s=9999)
        client.start()
        assert client._running is True
        client.stop()

    def test_start_idempotent(self) -> None:
        bus = EventBus()
        client = V2xClient("localhost", 1883, bus, sim_mode=True, sim_interval_s=9999)
        client.start()
        client.start()  # second call must not raise
        assert client._running is True
        client.stop()

    def test_stop_clears_running_flag(self) -> None:
        bus = EventBus()
        client = V2xClient("localhost", 1883, bus, sim_mode=True, sim_interval_s=9999)
        client.start()
        client.stop()
        assert client._running is False
