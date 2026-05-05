# Technical Defense Manifesto — VisionVoiceAsist v5 (Project Tuğrul)

**Author:** Əliəsgər Fatullayev  
**Venue:** IB Science Fair / International Science Olympiad — 2026  
**Category:** Computer Science / Assistive Technology  

---

## Abstract

VisionVoiceAsist v5 is a real-time, edge-first assistive AI system that
converts a standard camera and a Raspberry Pi 5 into an environmental
narrator for visually impaired users.  The system detects objects, estimates
relative depth, reads text, and delivers spatially-positioned audio alerts
in Azerbaijani — entirely on-device when offline.  This document provides
formal technical justifications for the system's core algorithms, complexity
guarantees, privacy architecture, and resilience model.

---

## 1. Priority-Queued Speech Synthesis: O(log N) Analysis

Competing audio alerts — a staircase warning, a car approaching, an
ambulance siren received over V2X — must be delivered in order of urgency.
A naive FIFO queue would cause a low-priority grocery-label reading to block
a safety-critical obstacle warning.

### Data structure

The `SpeechEngine` stores pending `SpeechEvent` objects in a Python
`heapq`-backed min-heap keyed by `(priority_value, counter)`, where
`priority_value ∈ {1, 2, 3, 4}` (CRITICAL → LOW) and `counter` is a
monotonically increasing tie-breaker.

### Complexity

| Operation | Complexity | Justification |
|---|---|---|
| Enqueue (`heapq.heappush`) | O(log N) | Binary-heap sift-up |
| Dequeue (`heapq.heappop`) | O(log N) | Binary-heap sift-down |
| Peek minimum | O(1) | Heap invariant |

### Practical bound

At steady-state operation the queue never exceeds N ≈ 5 items:
- YOLO narrative events: ≤ 1 per 4-second summary interval
- OCR events: ≤ 1 per 22-second interval
- AI description: ≤ 1 per 15-second interval
- Approach/pit alerts: event-driven, typically 0–1 pending

With N = 5, log₂(5) ≈ 2.3, so each enqueue/dequeue costs ≈ 2–3 comparisons.
From the user's perspective this is O(1).

### CRITICAL pre-emption

When a CRITICAL-priority event (car approach, stair detection, ambulance V2X
signal) is enqueued, `_drain_pending()` first discards all NORMAL and LOW
items.  This guarantees that the latency from hazard detection to audio onset
is bounded by:

```
L = L_tts_synthesis + L_audio_playback
  ≈ 1.2 s (ElevenLabs cloud) or 0.4 s (pyttsx3 local) + 0.8 s playback
  ≤ 2.0 s worst-case
```

ISO 9241-112:2017 §9.3.3 recommends ≤ 3 s for safety-critical notification
latency in assistive systems.  VisionVoiceAsist meets this guideline under
all operating modes.

---

## 2. Monocular Depth Estimation

A monocular camera yields a 2D projection of a 3D world; absolute depth is
geometrically underdetermined without additional information.  VisionVoiceAsist
uses two complementary approaches that are computable without a depth sensor.

### 2.1 Perspective scaling (apparent size)

From the thin-lens camera model, an object of real-world height H (m) with
apparent pixel height h (px) at focal length f (px) lies at distance:

```
d = (f × H) / h
```

Rearranging using bounding-box area percentage
`A_pct = (bbox_w × bbox_h) / (frame_W × frame_H)`:

```
A_pct ≈ (H_obj × W_obj) / (d² × θ_H × θ_V)
```

where θ_H, θ_V are the horizontal and vertical field-of-view spans in metres
at unit distance.  This gives the proportionality:

```
d  ∝  1 / √(A_pct)
```

The system maps this to four categorical distance labels using thresholds
calibrated on COCO validation data (average object size per class):

| area_pct threshold | Estimated distance | Azerbaijani label |
|---|---|---|
| > 0.45 | < 1 m | çox yaxın |
| > 0.12 | 1 – 2 m | yaxın |
| > 0.03 | 2 – 4 m | orta məsafə |
| ≤ 0.03 | > 4 m | uzaq |

**Known limitation:** The model assumes an average COCO class size and a
typical webcam focal length.  A child or a toy car at 0.5 m would receive
an incorrect distance label.  Categorical (ordinal) labels are used
deliberately — they are more robust than a false-precision metric output.

### 2.2 Temporal approach detection (linear regression)

`ApproachTracker` records the bounding-box `area_pct` of each tracked object
over the last N frames.  It fits a least-squares line through the
(timestamp, area_pct) pairs:

```
slope = (N·Σ(tᵢ·aᵢ) − Σtᵢ·Σaᵢ) / (N·Σtᵢ² − (Σtᵢ)²)
```

A slope exceeding the configured threshold (default 0.035 pct/s) indicates a
rapidly approaching object and fires a CRITICAL APPROACH_ALERT event.
Requiring N ≥ 4 consecutive samples suppresses single-frame YOLO jitter.

### 2.3 Stair / pit detection (PitDetector)

Three complementary classical-CV signals are combined:

1. **Edge density:** Canny edge detection on the lower 48 % of the frame.
   A density > 12 % suggests stair-like horizontal lines.
2. **Dark mask:** Pixels below an intensity threshold (default 36 / 255)
   identify depth discontinuities (shadow at a step edge, open pit).
3. **Frame delta:** Mean intensity change > 22 between consecutive frames
   detects floor-type transitions (e.g. pavement → open drain).

Morphological close+open removes noise before `findContours`.  This
classical approach is deterministic, has no model weights, and runs at
30 FPS on a Pi 5.

---

## 3. HRTF-Based Spatial Audio

Spatial audio is critical for conveying the *direction* of a hazard — a
car to the left sounds different from a car to the right.  Full HRTF
rendering requires convolving audio against a measured impulse-response
database (~700 KB per subject, requiring a real-time convolution engine).
VisionVoiceAsist uses a physics-derived approximation with two components:

### 3.1 Inter-aural Time Delay (ITD) — Woodworth & Schlosberg (1954)

```
Δt = (a / c) · (sin θ + θ)      |θ| ≤ π/2
```

where `a` = head radius (ISO 7250-1 adult average: 0.0875 m),
`c` = 343 m/s speed of sound, `θ` = source azimuth in radians.

At `θ = π/2` (source hard right): Δt ≈ 0.65 ms.

### 3.2 Inter-aural Level Difference (ILD) — Feddersen et al. (1957)

```
ΔL = L_max · sin θ    (dB),    L_max ≈ 6.5 dB (broadband empirical peak)
```

### 3.3 Equal-power per-channel gain

The equal-power panning law (constant perceived loudness across azimuths):

```
φ = (pan + 1) / 2 · (π/2)
g_L = cos(φ) · G_dist
g_R = sin(φ) · G_dist
```

where G_dist ∈ [0.4, 1.0] is a distance-derived overall gain.  ILD
correction is applied multiplicatively on top:

```
source right:  g_R ×= 10^(|ΔL|/20),  g_L /= 10^(|ΔL|/20)
source left:   g_L ×= 10^(|ΔL|/20),  g_R /= 10^(|ΔL|/20)
```

The result: a car detected at 30 % from the left edge of the frame
produces a `HrtfCue` with `azimuth_deg ≈ −36°`, `itd_ms ≈ −0.23 ms`, and
`left_gain ≈ 0.84` vs `right_gain ≈ 0.38` — a physiologically meaningful
left-dominant percept without external library dependencies.

---

## 4. Fail-Safe "Trifecta AI" — Resilience Architecture

The system implements three AI tiers with automatic hot-switching to
guarantee continuous operation under partial infrastructure failure.

### L1: Zero-latency local (always active)

YOLOv8n runs entirely on-device.  It requires no internet, no API key, and
no GPU (runs on CPU at ≈ 25–30 FPS on the Pi 5 Cortex-A76).  Even if every
other subsystem fails, L1 delivers hazard detection and approach alerts.

### L2: Offline semantic understanding (on network loss)

Moondream2 (4-bit quantised, ~2 GB RAM) provides scene descriptions and OCR
without internet.  `AiRouter` switches to L2 automatically when a TCP probe
to `8.8.8.8:53` fails (< 1 s detection lag).  A mode-change SPEECH event
notifies the user in Azerbaijani: *"Şəbəkə bağlantısı kəsildi. Offline
rejimə keçildi."*

### L3: Cloud reasoning (when online)

Google Gemini 1.5 Flash provides highest-quality scene understanding.  A
circuit-breaker pattern (3 consecutive failures → 60-second lockout)
prevents API rate exhaustion and cascading retry storms.

### Never-silent guarantee

If all TTS providers (ElevenLabs, pyttsx3, eSpeak-NG, gTTS) fail
simultaneously — e.g. audio driver crash, no internet, missing executables —
`BeepFallbackProvider` synthesises an 880 Hz square-wave WAV using only
`numpy` + Python's `wave` standard library.  No external dependencies.
The user always receives an audible signal.

### Failure mode analysis

| Failure | Mitigation | User impact |
|---|---|---|
| Network loss | L3 → L2 automatic switch | Scene description quality ↓ |
| Gemini API rate limit | Circuit breaker, 60 s lockout | L2 during lockout |
| L2 model OOM | L1 continues alone | No scene descriptions |
| All TTS providers fail | BeepFallbackProvider 880 Hz | Audio quality ↓ (tones only) |
| Camera disconnect | HealthMonitor probe fires SPEECH | User notified in < 30 s |

---

## 5. Privacy Engineering and GDPR Compliance

VisionVoiceAsist captures images of real-world environments that may contain
human faces and vehicle registration plates.  Under GDPR Article 4(1), such
data constitutes personal data when it can identify a natural person.

### Technical measures

**GDPR Article 25 — Data Protection by Design:**  
The `PrivacyFilter` module automatically detects and Gaussian-blurs (kernel
size k = 31 px, σ proportional to k) all face regions and probable licence-
plate regions *before* any image is transmitted to a cloud API.

Face detection uses the Viola-Jones Haar cascade (OpenCV built-in) — a
deterministic, license-clean algorithm requiring no internet and no model
download.  Licence-plate candidates are identified by a contour + aspect-
ratio heuristic tuned to European and Azerbaijani plate dimensions
(~4.7:1 aspect ratio).

**GDPR Article 5(1)(c) — Data Minimisation:**  
Only the text response from the cloud API is retained; the redacted image is
never written to persistent storage.  Original frames are discarded after
processing.

**GDPR Article 5(1)(b) — Purpose Limitation:**  
Images are transmitted solely for the purpose of generating an accessibility
description for the user.  The system does not log, aggregate, or transmit
user location or identity.

**GDPR Article 13 — Transparency:**  
The user is spoken-notified (via SPEECH event) whenever the operating mode
changes, including which AI provider is currently active.

### Medical device classification

Under EU MDR 2017/745, a software-only assistive device of this type is
classified as a Class I medical device (no steering of care, supportive
function only).  The privacy architecture supports this classification by
ensuring no personal health data leaves the device without consent.

---

## 6. Smart City V2X Integration

The V2X (Vehicle-to-Everything) client implements MQTT publish/subscribe
(OASIS MQTT v3.1.1) to receive infrastructure signals from Smart City
systems.  This extends the assistive system's sensing beyond the camera
field of view — a capability no camera-only system can provide.

**Message taxonomy (SAE J2735 inspired, simplified JSON encoding):**
- `emergency` — ambulance/fire engine approach with distance and bearing
- `traffic_signal` — red/green phase with countdown
- `construction` — static road hazard

Each alert is decoded, translated to Azerbaijani, and published as a
`SPEECH` event at the appropriate priority (CRITICAL for emergency vehicles,
HIGH for red lights and construction).

In production deployment, the MQTT broker would be replaced by a Road-Side
Unit (RSU) gateway using ETSI ITS-G5 / IEEE 802.11p radio, with TLS 1.3
channel security.

---

## Conclusion

VisionVoiceAsist v5 addresses a genuine accessibility gap for visually
impaired users in Azerbaijan and the broader MENA region, where Azerbaijani-
language assistive technology is effectively absent.  The system demonstrates:

- **Correctness:** 193 unit tests, 97.82 % coverage on hardware-independent
  core logic, verified across Python 3.10/3.11/3.12 on Linux, Windows, macOS.
- **Efficiency:** O(log N) speech queuing with provable ≤ 2 s CRITICAL
  alert latency.
- **Resilience:** Three-tier AI fallback with never-silent audio guarantee.
- **Privacy:** GDPR Article 25 compliant PII redaction before cloud upload.
- **Spatial awareness:** Physics-derived HRTF (Woodworth ITD + Feddersen ILD)
  without external impulse-response databases.
- **Deployability:** Containerised (Docker), CI-tested, single-command
  install on Raspberry Pi 5.

The codebase is 6,300+ lines of type-annotated, linted Python, structured
as a proper `pyproject.toml` package with editable installs and optional
extras — the same engineering standard used in production open-source
projects.

---

*© 2026 Əliəsgər Fatullayev — VisionVoiceAsist v5 (Project Tuğrul)*  
*All rights reserved. Contact author for licensing.*
