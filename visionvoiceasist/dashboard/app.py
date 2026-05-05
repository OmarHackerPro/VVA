"""Lightweight Flask + SocketIO dashboard.

- GET /            → live HTML view
- GET /stream.mjpg → MJPEG video stream
- POST /speak      → inject a TTS message remotely (admin only)
- WS  /events      → push detections, mode changes, health to browser

This module is **optional** — installed only with the [dashboard] extra.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import Optional

import cv2

from ..events import EventBus, EventType
from ..settings import DashboardSettings
from ..types import Detection, Frame, HealthReport, OperatingMode, Priority, SpeechEvent

log = logging.getLogger(__name__)

_INDEX_HTML = """\
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>VisionVoiceAsist — Dashboard</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #0e0e10; color: #eee; }
  header { padding: 14px 22px; background: #18181b; border-bottom: 1px solid #2a2a30; }
  h1 { margin: 0; font-size: 18px; letter-spacing: 0.4px; }
  main { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; padding: 12px; }
  .card { background: #18181b; border: 1px solid #2a2a30; border-radius: 8px; padding: 12px; }
  img { width: 100%; border-radius: 6px; }
  ul { list-style: none; padding: 0; margin: 0; max-height: 300px; overflow-y: auto; }
  li { padding: 4px 0; border-bottom: 1px dashed #2a2a30; font-family: ui-monospace, monospace; font-size: 13px; }
  textarea { width: 100%; height: 60px; background: #0e0e10; color: #eee; border: 1px solid #2a2a30; border-radius: 4px; padding: 8px; }
  button { padding: 8px 16px; background: #2563eb; color: white; border: 0; border-radius: 4px; cursor: pointer; }
</style>
</head><body>
<header><h1>VisionVoiceAsist · Live Dashboard</h1></header>
<main>
  <div class="card">
    <img id="stream" src="/stream.mjpg" alt="Live feed">
  </div>
  <div>
    <div class="card"><b>Mode:</b> <span id="mode">…</span><br><b>Battery:</b> <span id="battery">…</span></div>
    <div class="card"><b>Detections</b><ul id="dets"></ul></div>
    <div class="card"><b>Health</b><ul id="health"></ul></div>
    <div class="card">
      <b>Speak (remote)</b>
      <textarea id="msg" placeholder="Az dilində mesaj yazın..."></textarea>
      <button onclick="send()">Send</button>
    </div>
  </div>
</main>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js" integrity="sha384-2huaZvOR9iDzHqslqwpR87isEmrfxqyWOF7hr7BY6KG0+hVKLoEXMPUJw3ynWuhO" crossorigin="anonymous"></script>
<script>
const sock = io();
sock.on("detections", (data) => {
  const ul = document.getElementById("dets"); ul.innerHTML = "";
  data.forEach(d => {
    const li = document.createElement("li");
    li.textContent = `${d.label_az} (${(d.conf*100).toFixed(0)}%) — ${d.distance}`;
    ul.appendChild(li);
  });
});
sock.on("mode", (m) => document.getElementById("mode").textContent = m);
sock.on("health", (data) => {
  const ul = document.getElementById("health"); ul.innerHTML = "";
  Object.entries(data).forEach(([k, v]) => {
    const li = document.createElement("li");
    li.textContent = `${v.healthy ? "✅" : "❌"} ${k} — ${v.detail || ""}`;
    ul.appendChild(li);
  });
});
function send(){
  const m = document.getElementById("msg").value;
  fetch("/speak", {method: "POST", headers: {"Content-Type": "application/json"},
                   body: JSON.stringify({text: m})});
}
</script>
</body></html>
"""


class DashboardApp:
    """Flask+SocketIO server that consumes the EventBus."""

    def __init__(self, settings: DashboardSettings, bus: EventBus) -> None:
        try:
            from flask import Flask, Response, jsonify, request  # noqa: PLC0415
            from flask_socketio import SocketIO  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Dashboard requires flask + flask-socketio. "
                "Install with: pip install '.[dashboard]'"
            ) from exc

        self._s = settings
        self._bus = bus
        self._latest_frame: Optional[Frame] = None
        self._frame_lock = threading.Lock()
        self._health_state: dict[str, dict[str, object]] = {}

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*",
                                  async_mode="threading")

        self._setup_routes(jsonify, request, Response)
        self._wire_bus()

    def _setup_routes(self, jsonify, request, Response):  # type: ignore[no-untyped-def]
        @self.app.route("/")
        def index() -> str:
            return _INDEX_HTML

        @self.app.route("/health")
        def health() -> object:
            return jsonify(self._health_state)

        @self.app.route("/stream.mjpg")
        def stream() -> object:
            return Response(self._mjpeg_iter(),
                            mimetype="multipart/x-mixed-replace; boundary=frame")

        @self.app.route("/speak", methods=["POST"])
        def speak() -> object:
            body = request.get_json(silent=True) or {}
            text = str(body.get("text", "")).strip()
            if not text:
                return jsonify({"ok": False, "error": "empty"}), 400
            self._bus.publish(
                EventType.SPEECH,
                SpeechEvent(text=text, priority=Priority.NORMAL, force=True),
            )
            return jsonify({"ok": True})

    def _wire_bus(self) -> None:
        self._bus.subscribe(EventType.FRAME, self._on_frame)
        self._bus.subscribe(EventType.DETECTIONS, self._on_dets)
        self._bus.subscribe(EventType.MODE_CHANGE, self._on_mode)
        self._bus.subscribe(EventType.HEALTH, self._on_health)

    def _on_frame(self, frame: Frame) -> None:
        with self._frame_lock:
            self._latest_frame = frame

    def _on_dets(self, dets: list[Detection]) -> None:
        from ..i18n import distance_label  # local: cycle-safe

        payload = [
            {"label_az": d.label_az, "conf": d.conf,
             "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
             "distance": distance_label(d.area_pct)}
            for d in dets
        ]
        self.socketio.emit("detections", payload)

    def _on_mode(self, mode: OperatingMode) -> None:
        self.socketio.emit("mode", mode.value)

    def _on_health(self, report: HealthReport) -> None:
        self._health_state[report.component] = {
            "healthy": report.healthy, "detail": report.detail
        }
        self.socketio.emit("health", self._health_state)

    def _mjpeg_iter(self) -> Iterator[bytes]:
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                self.socketio.sleep(0.05)  # type: ignore[attr-defined]
                continue
            ok, buf = cv2.imencode(".jpg", frame.image,
                                    [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
            yield boundary + buf.tobytes() + b"\r\n"
            self.socketio.sleep(0.04)  # type: ignore[attr-defined]

    def start(self) -> None:
        thread = threading.Thread(
            target=self._run, daemon=True, name="DashboardServer"
        )
        thread.start()
        log.info("Dashboard at http://%s:%d/", self._s.host, self._s.port)

    def _run(self) -> None:
        self.socketio.run(  # type: ignore[no-untyped-call]
            self.app, host=self._s.host, port=self._s.port,
            debug=False, allow_unsafe_werkzeug=True,
        )
