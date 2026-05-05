[README_PRO.md](https://github.com/user-attachments/files/27416020/README_PRO.md)
# VisionVoiceAsist v5

**AI-powered wearable smart-glasses for visually impaired users — edge-first, offline-capable, Azerbaijani-native.**

> Developed by Əliəsgər Fatullayev for international Science Fair / Olympiad competition.

---

## What it does

VisionVoiceAsist turns an ordinary camera + Raspberry Pi 5 into a real-time environmental assistant:

| Capability | How |
|---|---|
| Object detection + distance | YOLOv8n on-device, 80 COCO classes, AZ translations |
| Stereo spatial audio | Pan voice alerts left/right to indicate object direction |
| Pit / stair detection | Classical CV — Canny edges + dark-mask, no ML needed |
| OCR | pytesseract — reads signs, labels, menus |
| AI scene description | Gemini Flash (online) or Moondream2 (offline), auto-switched |
| Voice queries | Whisper STT — ask "what is in front of me?" |
| Haptic feedback | RPi GPIO vibration motor — CRITICAL bursts |
| Battery & health monitor | psutil + periodic probes, audible degraded-mode alerts |
| Live dashboard | Flask+SocketIO — MJPEG stream, detection feed, remote TTS |

All speech output is in **Azerbaijani** (AZ). Fallback to English labels when no translation exists.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     EventBus (pub/sub)                  │
└──────────┬──────────────────────┬───────────────────────┘
           │                      │
    ┌──────▼──────┐        ┌──────▼──────┐
    │  Vision     │        │   Output    │
    │  Pipeline   │        │  Layer      │
    │  ─────────  │        │  ─────────  │
    │  CameraSource        │  SpeechEngine (PriorityQ)
    │  YoloDetector        │  HapticMotor (GPIO)
    │  ObjectTracker       │  Overlay (OpenCV HUD)
    │  StatefulNarrator    │  Dashboard (Flask/WS)
    │  PitDetector   │     └─────────────┘
    │  OcrModule     │
    │  SpatialAnalyzer│
    └──────┬─────────┘
           │ DETECTIONS / PIT / OCR events
    ┌──────▼──────────────────┐
    │  AI Router              │
    │  ─────────────────────  │
    │  GeminiVlm  (cloud)     │
    │  LocalVlm   (Moondream) │
    │  VoiceQueryService      │
    │  Circuit breaker, retry │
    └─────────────────────────┘
```

### Key design decisions

**Event-driven pub/sub** — `EventBus` decouples every module. Vision publishes `DETECTIONS`; speech subscribes. No direct method calls between subsystems.

**Offline-first** — `AiRouter` probes `8.8.8.8:53`. On network loss it switches to `LocalVlm` (Moondream2/SmolVLM) and announces the switch in Azerbaijani.

**Priority-preemptive TTS** — `SpeechEngine` uses a `PriorityQueue`. CRITICAL events (car approaching, stairs) drain all pending speech before enqueue — the user always hears the most important alert first.

**Never-silent fallback** — `BeepFallbackProvider` synthesizes a 880Hz square-wave WAV from first principles. Even if ElevenLabs, pyttsx3, and eSpeak all fail, the user receives an audible signal.

**Stateful narration** — `StatefulNarrator` suppresses repeated "car, car, car". It re-announces only when a tracked object changes position bucket (left/center/right) or distance bucket.

**Command injection prevention** — All TTS shell invocations use `safe_subprocess.run(argv, shell=False)`. Shell metacharacters in OCR text or AI responses cannot become commands.

---

## Hardware requirements

| Component | Spec |
|---|---|
| SBC | Raspberry Pi 5 (4 GB+ RAM recommended) |
| Camera | CSI or USB, ≥ 640×480 |
| Speaker / bone-conduction headphones | 3.5mm or Bluetooth |
| Vibration motor | 5V DC, connected to GPIO 18 |
| Optional IMU | MPU-6050 via I²C (smbus2) |

Runs on any Linux/macOS/Windows machine with a webcam for development.

---

## Installation

### Raspberry Pi 5 (production)

```bash
curl -fsSL https://raw.githubusercontent.com/aliasgar/visionvoiceasist/main/install.sh | bash
```

Or clone and run:

```bash
git clone https://github.com/aliasgar/visionvoiceasist
cd visionvoiceasist
chmod +x install.sh && ./install.sh
```

### Development (any OS)

```bash
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[dashboard]"    # optional: Flask dashboard
pip install -e ".[local-vlm]"    # optional: Moondream2 offline AI
pip install -e ".[voice-query]"  # optional: Whisper STT
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `GEMINI_KEY` | — | Google Gemini API key |
| `ELEVENLABS_KEY` | — | ElevenLabs TTS key (optional) |
| `VVA_CAM_INDEX` | `0` | OpenCV camera index |
| `VVA_YOLO_CONF` | `0.35` | YOLO confidence threshold |
| `VVA_OFFLINE_MODE` | `auto` | `auto` / `always` / `never` |
| `VVA_DASHBOARD_ENABLED` | `false` | Enable Flask dashboard |
| `VVA_DASHBOARD_PORT` | `8080` | Dashboard port |
| `VVA_SHOW_GUI` | `true` | Show OpenCV window |
| `VVA_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Usage

```bash
vva                        # run with defaults
vva --no-gui               # headless (Raspberry Pi)
vva --offline-mode always  # force offline AI
vva --dashboard            # enable live dashboard at :8080
vva --cam 1 --conf 0.40    # different camera, higher YOLO threshold
```

### Keyboard shortcuts (GUI mode)

| Key | Action |
|---|---|
| `Q` | Quit |
| `R` | Force AI analysis right now |
| `S` | Status report (battery, FPS, mode) |
| `H` | Force health check |

---

## Running tests

```bash
pytest                          # all tests + coverage report
pytest -m "not slow"           # skip slow tests
pytest tests/test_events.py -v # single module
```

Coverage is measured only on hardware-independent core logic. All 193 tests should pass at ≥ 97% coverage.

---

## TTS provider chain

The system tries providers in order until one succeeds:

1. **ElevenLabs** — cloud, highest quality (requires `ELEVENLABS_KEY`)
2. **pyttsx3** — local, cross-platform, no internet required
3. **eSpeak-NG** — subprocess, available on most Linux systems
4. **gTTS** — Google TTS, requires internet
5. **BeepFallback** — synthesizes a 880Hz square-wave WAV — always works

---

## Project structure

```
visionvoiceasist/
├── types.py          # Core data types (BBox, Detection, SpeechEvent…)
├── events.py         # EventBus pub/sub
├── settings.py       # Frozen config loaded from env vars
├── i18n.py           # Azerbaijani translations + message templates
├── health.py         # HealthMonitor + liveness probes
├── monitoring.py     # FpsCounter, BatteryWatcher
├── runtime.py        # Top-level orchestrator (main loop)
├── cli.py            # argparse entrypoint
├── vision/           # Camera, YOLO, tracking, spatial, color, pit, OCR
├── ai/               # Gemini VLM, local VLM, AI router, voice query
├── audio/            # TTS providers, speech engine, spatial panner
├── hardware/         # GPIO haptics, IMU sensor
├── ui/               # OpenCV HUD overlay
├── dashboard/        # Flask+SocketIO live dashboard
└── utils/            # retry, safe_subprocess, device detection, logging
```

---

## Security

- API keys stored in `.env` only — never committed (`.gitignore` enforced)
- All subprocess calls use `shell=False` — no command injection
- `detect-secrets` pre-commit hook scans for leaked credentials
- `bandit` static analysis in CI

---

## License

Proprietary — all rights reserved. Contact the author for licensing inquiries.
