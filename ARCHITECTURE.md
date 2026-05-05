# Architecture — VisionVoiceAsist v5

## Event flow

```
CameraSource ──► [FRAME] ──► DashboardApp (MJPEG stream)
                         └─► Runtime._stage_* (processing)

Runtime._stage_yolo ──► YoloDetector
                     └─► ObjectTracker  (adds stable track_id)
                     └─► StatefulNarrator (suppresses repeats)
                     └─► [DETECTIONS] ──► DashboardApp (WS)
                     └─► [APPROACH_ALERT] ──► SpeechEngine + HapticMotor

Runtime._stage_pit ──► PitDetector
                    └─► [PIT_WARNING] ──► SpeechEngine + HapticMotor

Runtime._stage_ai ──► PrivacyFilter.redact() (PII blur)
                   └─► AiRouter.describe() (thread)
                   └─► [AI_DESCRIPTION] ──► SpeechEngine

Runtime._stage_ocr ──► OcrModule
                    └─► [OCR_TEXT] ──► SpeechEngine

[SPEECH] ──► SpeechEngine.enqueue()
          └─► TTS provider chain ──► AudioPlayer (HrtfCue applied)

[HAPTIC] ──► HapticMotor (GPIO PWM / no-op stub)

[V2X_ALERT] ──► V2xClient (MQTT subscriber, daemon thread)
             └─► [SPEECH] ──► SpeechEngine

[HEALTH] ──► DashboardApp (WS) + HealthMonitor state
[MODE_CHANGE] ──► DashboardApp (WS) + SpeechEngine
```

## Module contracts

### EventBus
- Thread-safe pub/sub. Subscribers run on the publisher's thread.
- Subscriber exceptions are isolated — one bad subscriber can't stop others.
- `clear()` removes all subscriptions (used in tests).

### Settings (dependency injection)
- All configuration is a frozen `@dataclass` loaded once via `Settings.from_env()`.
- Sub-settings (`CameraSettings`, `AiSettings`, etc.) are also frozen.
- No module reads `os.environ` directly — only `Settings` does.
- `with_overrides(**kwargs)` returns a new instance (tests, CLI).

### SpeechEngine (priority-preemptive queue)
- `PriorityQueue` ordered by `Priority.value` (CRITICAL=1 → LOW=4).
- `CRITICAL` events call `_drain_pending()` before enqueue — user always hears the most urgent alert.
- `ThreadPoolExecutor(max_workers=tts.pool_workers)` runs TTS synthesis concurrently.
- `health()` probe: healthy if last successful synthesis within 30s.

### AiRouter (circuit breaker + offline-first)
- `offline_mode=auto`: TCP probe to `8.8.8.8:53` every 10s.
- `offline_mode=always`: always uses `LocalVlm`.
- `offline_mode=never`: always uses `GeminiVlm` (falls back on error).
- Circuit breaker: 3 consecutive Gemini failures → 60s lockout.
- Mode changes publish `MODE_CHANGE` event + SPEECH announcement.

### ObjectTracker (IoU greedy)
- Greedy assignment: each detection matches the best-IoU active track of the same class.
- `max_age=15` frames: tracks not seen for 15 frames are dropped.
- Returns new `Detection` instances with stable `track_id` — original frozen objects unchanged.

### StatefulNarrator
- Announces a track at most once per `cooldown_s=6.0` seconds.
- Re-announces when position bucket (left/center/right) OR distance bucket changes.
- GC: entries older than `5 × cooldown_s` are deleted from `_announced`.

### PitDetector (three-stage classical CV)
1. **Edge density**: Canny on lower 48% of frame. Density > 12% → stair candidate.
2. **Dark mask**: pixels below `pit_dark=36` threshold → shadow/depth discontinuity.
3. **Frame delta**: mean intensity jump > 22 between frames → floor transition alert.
Morphological close+open removes noise. `cv2.findContours` finds connected regions.

### BeepFallbackProvider (silence prevention)
Generates a 880Hz square-wave WAV using only `numpy` + `wave` stdlib.
No external TTS library required. Ensures the user always receives an audible signal.

### PrivacyFilter (GDPR Article 25)
Runs before every cloud API call. Detects faces (Haar cascade, OpenCV built-in) and
probable licence plates (contour + aspect-ratio heuristic) and Gaussian-blurs them.
Returns `(redacted_frame, RedactionStats)`. Never mutates the original frame.
Falls back to a no-op when cv2 is unavailable.

### HrtfCalculator (spatial audio)
Computes per-channel gains and ITD/ILD from a normalised pan value using:
- ITD: Woodworth & Schlosberg (1954) formula `Δt = (a/c)·(sin θ + θ)`
- ILD: Feddersen et al. (1957) approximation `ΔL = L_max·sin θ`
- Equal-power panning law for base gains; ILD applied multiplicatively.
Replaces simple linear panning with a physics-grounded stereo model.

### V2xClient (Smart City integration)
Asynchronous MQTT subscriber on topic `vva/v2x/#`. Decodes JSON payloads covering
emergency vehicle approach, traffic signal phases, and construction warnings.
Fires `V2X_ALERT` + `SPEECH` events on the shared EventBus. Operates in simulation
mode (synthetic events, no real broker) or against a live Mosquitto / RSU gateway.

## How to add a new detector

1. Create `visionvoiceasist/vision/my_detector.py`.
2. Add the new `EventType` value to `events.py` if needed.
3. Instantiate the detector in `Runtime.__init__`.
4. Add a `_stage_my_detector(frame, now)` method to `Runtime` following the timer pattern.
5. Call it in `_main_loop`.
6. Subscribe to its events in `_wire_bus` or other relevant modules.
7. Write unit tests in `tests/test_my_detector.py`.

## How to add a new TTS provider

1. Subclass `TtsProvider` in `visionvoiceasist/audio/providers.py`.
2. Implement `synthesize(text) -> Path` — write a WAV file, return its path.
3. Add it to the `_build_chain()` list in `SpeechEngine` at the appropriate priority.
4. Handle `ImportError` gracefully (provider deps are optional).

## How to add a new language

1. Copy `i18n.py`, rename, translate `AZ_LABELS` and all `Messages.*` strings.
2. Pass the new module to `Runtime` via settings (e.g. `Settings.language="en"`).
3. Update `position_label()` and `distance_label()` with appropriate phrasing.
