"""V2X (Vehicle-to-Everything) MQTT client — Smart City integration.

Connects to an MQTT broker that publishes V2X messages from Smart City
infrastructure (traffic signals, emergency vehicles, construction zones).
When a relevant alert arrives, a SPEECH event is published on the EventBus
so the user is warned of hazards *outside* their camera field of view.

Message format (JSON, topic vva/v2x/#):
    {"type": "emergency",       "vehicle": "ambulance", "distance_m": 120, "direction": "sağdan"}
    {"type": "traffic_signal",  "phase": "red",         "time_to_green_s": 18}
    {"type": "construction",    "message": "Yol işləri. Diqqətli olun."}

In production, replace the simulated broker with a real V2X RSU gateway
speaking SAE J2735 / ETSI ITS-G5 message sets over a secured MQTT channel.

Dependencies:
    paho-mqtt >= 1.6 (optional — a graceful stub operates when absent)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from ..events import EventBus, EventType
from ..types import Priority, SpeechEvent

log = logging.getLogger(__name__)

try:
    import paho.mqtt.client as _mqtt  # type: ignore[import-untyped]

    _PAHO_OK = True
except ImportError:
    _PAHO_OK = False
    _mqtt = None  # type: ignore[assignment]


# ── Alert types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class V2xAlert:
    """Decoded V2X alert ready for speech and dashboard consumption."""

    alert_type: str       # "emergency" | "traffic_signal" | "construction"
    message_az: str       # Azerbaijani speech text
    priority: Priority
    raw: dict[str, Any]


# Synthetic events used in simulation mode (round-robin).
_SIM_EVENTS: list[dict[str, Any]] = [
    {"type": "emergency", "vehicle": "sürətli yardım", "distance_m": 200, "direction": "sağdan"},
    {"type": "traffic_signal", "phase": "red", "time_to_green_s": 20},
    {"type": "construction", "message": "Yol işləri. Diqqətli olun."},
    {"type": "traffic_signal", "phase": "green"},
]


# ── Message decoder ───────────────────────────────────────────────────────────

def _decode_message(payload: dict[str, Any]) -> V2xAlert | None:
    """Map a raw JSON payload to a V2xAlert.  Returns None if unrecognised."""
    alert_type = payload.get("type", "")

    if alert_type == "emergency":
        vehicle = payload.get("vehicle", "təcili nəqliyyat")
        direction = payload.get("direction", "")
        distance = payload.get("distance_m")
        dist_str = f"{distance} metr " if distance else ""
        msg = f"DİQQƏT! {dist_str}{direction} {vehicle} gəlir!"
        return V2xAlert(alert_type, msg.strip(), Priority.CRITICAL, payload)

    if alert_type == "traffic_signal":
        phase = payload.get("phase", "")
        if phase == "red":
            ttg = payload.get("time_to_green_s")
            suffix = f" {ttg} saniyə sonra yaşıl." if ttg else "."
            return V2xAlert(alert_type, f"Qırmızı işıq{suffix}", Priority.HIGH, payload)
        if phase == "green":
            return V2xAlert(alert_type, "Yaşıl işıq. Keçə bilərsiniz.", Priority.NORMAL, payload)

    if alert_type == "construction":
        msg = payload.get("message", "Yol işləri var.")
        return V2xAlert(alert_type, str(msg), Priority.HIGH, payload)

    return None


# ── Client ────────────────────────────────────────────────────────────────────

class V2xClient:
    """Async MQTT subscriber that forwards V2X alerts to the EventBus.

    Args:
        broker_host:    MQTT broker hostname or IP address.
        broker_port:    MQTT broker port (default 1883).
        bus:            Shared EventBus instance.
        sim_mode:       When True, fire synthetic events without a real
                        broker (development / testing).
        sim_interval_s: Seconds between simulated alerts (sim_mode only).
    """

    TOPIC = "vva/v2x/#"

    def __init__(
        self,
        broker_host: str,
        broker_port: int,
        bus: EventBus,
        *,
        sim_mode: bool = False,
        sim_interval_s: float = 30.0,
    ) -> None:
        self._host = broker_host
        self._port = broker_port
        self._bus = bus
        self._sim = sim_mode
        self._sim_interval = sim_interval_s
        self._running = False
        self._mqtt_thread: threading.Thread | None = None
        self._sim_thread: threading.Thread | None = None
        self._client: Any = None

    def start(self) -> None:
        """Start the client (idempotent)."""
        if self._running:
            return
        self._running = True

        if self._sim:
            self._sim_thread = threading.Thread(
                target=self._sim_loop, daemon=True, name="v2x-sim"
            )
            self._sim_thread.start()
            log.info(
                "V2xClient: simulation mode — synthetic events every %.0fs",
                self._sim_interval,
            )
        elif _PAHO_OK:
            self._mqtt_thread = threading.Thread(
                target=self._mqtt_loop, daemon=True, name="v2x-mqtt"
            )
            self._mqtt_thread.start()
        else:
            log.warning("V2xClient: paho-mqtt not installed — V2X disabled")

    def stop(self) -> None:
        """Stop the client."""
        self._running = False
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass

    # ── MQTT loop ─────────────────────────────────────────────────────────────

    def _mqtt_loop(self) -> None:
        client = _mqtt.Client()  # type: ignore[union-attr]
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        self._client = client
        try:
            client.connect(self._host, self._port, keepalive=60)
            client.loop_forever()
        except Exception as exc:
            log.warning("V2xClient: MQTT connection error — %s", exc)

    def _on_connect(self, client: Any, _ud: Any, _flags: Any, rc: int) -> None:
        if rc == 0:
            client.subscribe(self.TOPIC)
            log.info(
                "V2xClient: subscribed to %s at %s:%s",
                self.TOPIC, self._host, self._port,
            )
        else:
            log.warning("V2xClient: broker refused connection (rc=%d)", rc)

    def _on_message(self, _client: Any, _ud: Any, msg: Any) -> None:
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.debug("V2xClient: malformed message — %s", exc)
            return
        self._dispatch(payload)

    # ── Simulation loop ───────────────────────────────────────────────────────

    def _sim_loop(self) -> None:
        idx = 0
        while self._running:
            time.sleep(self._sim_interval)
            if not self._running:
                break
            self._dispatch(_SIM_EVENTS[idx % len(_SIM_EVENTS)])
            idx += 1

    # ── Shared dispatch ───────────────────────────────────────────────────────

    def _dispatch(self, payload: dict[str, Any]) -> None:
        alert = _decode_message(payload)
        if alert is None:
            log.debug("V2xClient: unrecognised payload %r", payload)
            return
        log.info("V2X [%s] %s", alert.alert_type, alert.message_az)
        self._bus.publish(EventType.V2X_ALERT, alert)
        self._bus.publish(
            EventType.SPEECH,
            SpeechEvent(text=alert.message_az, priority=alert.priority, force=True),
        )
