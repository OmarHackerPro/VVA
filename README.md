# VisionVoiceAsist v3.0 — Olympiad Edition
### *Ağıllı Görmə Yardımçısı — Görmə Məhdudiyyətli Şəxslər üçün*

**Təsisçi və Baş Tərtibatçı:** Əliəsgər  
**Versiya:** 3.0 "Olympiad Edition"  
**Kateqoriya:** Süni İntellekt · Geyilə Bilən Texnologiya · Robototexnika

---

## 📋 Mündəricat
1. [Layihə Haqqında](#layihə-haqqında)
2. [Texniki Arxitektura](#texniki-arxitektura)
3. [Tələb Olunan Avadanlıq](#tələb-olunan-avadanlıq)
4. [Addım-addım Quraşdırma](#addım-addım-quraşdırma)
5. [API Açarlarının Alınması](#api-açarlarının-alınması)
6. [Proqramın İşlədilməsi](#proqramın-işlədilməsi)
7. [Klaviatura Qısayolları](#klaviatura-qısayolları)
8. [İnkişaf Planı (Roadmap)](#inkişaf-planı)

---

## Layihə Haqqında

**VisionVoiceAsist** — görmə məhdudiyyətli şəxslərin ətraf mühitlə müstəqil şəkildə əlaqə qurmasını təmin edən, süni intellekt əsaslı "ağıllı eynək" sistemidir.

Kamera real vaxt rejimində mühiti skan edir → AI alqoritmləri analiz edir → nəticəni Azərbaycan dilinde səsli komandalar şəklinde birbaşa istifadəçiyə ötürür.

### İnnovasiyalar

| Xüsusiyyət | Açıqlama |
|---|---|
| 🤖 **Gemini Vision AI** | Hər 15s-də tam mühit analizi — "Masada noutbuk var, sağda qapı açıqdır" |
| 📐 **Scene Graph** | Əşyalar arası məkan əlaqəsi — "Masanın üzərində kitab var" |
| 🎯 **Approach Tracker** | Xətti reqressiya ilə yaxınlaşan cisim aşkarlanması |
| 🕳️ **Pit Detector** | 4 mərhələli çuxur/pilləkən/astana aşkarlaması |
| 📳 **GPIO Vibrasiya** | Raspberry Pi vibrasiya motoru — toxunma xəbərdarlığı |
| 🔊 **Priority Queue TTS** | CRITICAL > HIGH > NORMAL — vacib mesajlar növbəni kəsir |
| 📝 **OCR** | Azərbaycan + İngilis mətn tanıma |
| 🔋 **Sistem Monitor** | Batareya, FPS, gecikmə izləmə |

---

## Texniki Arxitektura

```
Kamera (30 FPS)
    │
    ├── [Thread 1] YOLOv8n Detector (hər 4s xülasə)
    │       └── ApproachTracker (bbox velocity regression)
    │       └── SpatialAnalyzer (Scene Graph builder)
    │
    ├── [Thread 2] PitDetector (hər 0.4s)
    │       └── Canny edges + dark mask + mean velocity
    │
    ├── [Thread 3] Gemini Vision AI (hər 15s)
    │       └── -gemini-1.5-flash + Scene Memory
    │
    └── [Thread 4] OCR Module (hər 22s)
            └── Tesseract (aze+eng) + 4-stage preprocessing
                             │
                    Priority Speech Queue
                    CRITICAL(1) > HIGH(2) > NORMAL(3)
                             │
                    ElevenLabs multilingual_v2
                    └── eSpeak (offline fallback)
                    GPIO Vibration Motor
```

---

## Tələb Olunan Avadanlıq

| Komponent | Model | Niyə? |
|---|---|---|
| **Prosessor** | Raspberry Pi 5 (8GB) | AI üçün güclü NPU |
| **Kamera** | Raspberry Pi Camera V2/V3 | CSI interfeysi, düşük gecikmə |
| **Batareya** | Li-ion 10000mAh (PD 3.0) | 6+ saat iş vaxtı |
| **Vibrasiya Motoru** | 5V DC Coin Motor | GPIO 18 pinə qoşulur |
| **Qulaqlıq** | 3.5mm jack və ya Bluetooth | Əllər azad qalır |
| **Yaddaş** | 64GB microSD (A2 sinif) | Sürətli I/O |

---

## Addım-addım Quraşdırma

### ✅ Metod 1: Avtomatik (Tövsiyə olunur)

```bash
# 1. Faylları Pi-yə kopyalayın
scp main.py install.sh pi@raspberrypi.local:~/

# 2. Pi-də terminalı açın
ssh pi@raspberrypi.local

# 3. Bir əmrlə quraşdırın
chmod +x install.sh
sudo ./install.sh
```

Skript avtomatik olaraq bütün asılılıqları quraşdırır, API açarını soruşur, boot-da avtomatik başlatma ayarlar.

---

### 🔧 Metod 2: Əl ilə (Addım-addım)

#### Addım 1 — Raspberry Pi OS Yeniləyin

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    python3-pip python3-dev python3-venv \
    tesseract-ocr tesseract-ocr-aze \
    mpg123 espeak \
    libatlas-base-dev \
    v4l-utils
```

#### Addım 2 — Kameranı aktivləşdirin

```bash
sudo raspi-config
# → 3 Interface Options
# → I1 Camera
# → Yes (Bəli)
# → Finish → Reboot
```

Və ya birbaşa:
```bash
sudo raspi-config nonint do_camera 0
sudo reboot
```

Kameranı yoxlamaq üçün:
```bash
libcamera-hello           # Kamera görüntüsü görünürsə — qoşulub
v4l2-ctl --list-devices   # USB kamera üçün
```

#### Addım 3 — Python Virtual Mühiti Yaradın

```bash
python3 -m venv ~/vva_env --system-site-packages
source ~/vva_env/bin/activate
pip install --upgrade pip
```

#### Addım 4 — Python Asılılıqlarını Quraşdırın

```bash
pip install \
    ultralytics \
    anthropic \
    requests \
    psutil \
    numpy \
    pytesseract \
    opencv-python-headless \
    RPi.GPIO
```

> ⏱ Bu addım 5-10 dəqiqə çəkə bilər — PyTorch yüklənir.

#### Addım 5 — YOLOv8 Modelini Yükləyin

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Model avtomatik yüklənəcək (~6MB)
```

#### Addım 6 — Vibrasiya Motorunu Qoşun (GPIO)

```
Raspberry Pi 5 Pinout:
┌─────────────────────┐
│  Pi 5 GPIO Header   │
│  Pin 12 (GPIO 18) ──┼──► Motor (+) ucu
│  Pin 14 (GND)    ──┼──► Motor (-) ucu
└─────────────────────┘
Not: 5V motor üçün araya BC547 tranzistor + 1kΩ rezistor qoyun
```

#### Addım 7 — API Açarlarını Təyin Edin

`main.py` faylını açın və bu sətirləri dəyişdirin:

```python
# Config sinifi içində:
ANTHROPIC_KEY = "sk-ant-xxxxxxxxxxxxxxxxxx"   # ← buraya
ELEVENLABS_KEY = "sk_xxxxxxxxxxxxxxxxxxxxxxx" # ← artıq yazılıb
```

---

## API Açarlarının Alınması

### 🔑 Anthropic (gemini AI) — Pulsuz sınaq var

1. [console.anthropic.com](https://console.anthropic.com) saytına daxil olun
2. Qeydiyyatdan keçin (Gmail ilə mümkündür)
3. Sol panel: **API Keys** → **Create Key**
4. Açarı kopyalayın: `sk-ant-api03-xxxxxxxx...`
5. `main.py`-də `ANTHROPIC_KEY = "..."` sətrinə yapışdırın

> 💡 Yeni hesablara $5 pulsuz kredit verilir — günlük onlarla analiz üçün kifayətdir.

### 🔑 ElevenLabs (TTS Nitq) — Artıq konfiqurasiya edilib

ElevenLabs açarı kodda artıq yazılıdır. Dəyişdirmək lazım deyil.  
Yeni açar almaq istəsəniz: [elevenlabs.io](https://elevenlabs.io) → Profile → API Key

---

## Proqramın İşlədilməsi

### Əl ilə başlatma:

```bash
cd ~/VisionVoiceAsist
source /opt/visionvoiceasist/bin/activate
python3 main.py
```

### Parametrlərlə başlatma:

```bash
# Fərqli kamera indeksi ilə (USB kamera)
python3 main.py --cam 1

# Daha həssas aşkarlama üçün
python3 main.py --conf 0.40

# AI olmadan (offline rejim)
python3 main.py --noai
```

### Boot-da avtomatik başlatma:

```bash
sudo systemctl enable visionvoiceasist
sudo systemctl start visionvoiceasist

# Status yoxlamaq üçün
sudo systemctl status visionvoiceasist

# Jurnalları görmək üçün
journalctl -u visionvoiceasist -f
```

---

## Klaviatura Qısayolları

| Düymə | Funksiya |
|---|---|
| `Q` | Proqramdan çıxış |
| `R` | Dərhal gemini AI analizi başlat |
| `S` | Sistem statusu (batareya, FPS, statistika) |

---

## Səs Çıxışı Nümunələri

Sistem aşağıdakı kimi mesajlar söyləyir:

```
"Salam! VisionVoiceAsist aktiv oldu. Batareya 87 faizdir. Süni intellekt aktiv."

"Qarşınızda masa var, üzərində noutbuk, çay fincanı və 2 kitab var. Siz ofis mühitindəsiniz."

"KRİTİK TƏHLÜKƏ! insan çox yaxındır, dərhal dayanın!"

"DİQQƏT! Qarşınızda pilləkən var! Dərhal dayanın!"

"Xəbərdarlıq: Solunuzda maşın yaxınlaşır."

"Yazı oxuyuram: Çıxış. Exit."

"Xəbərdarlıq: Batareya 15 faizdir. Şarj etməyiniz tövsiyə olunur."
```

---

## Sorun Giderme

### Kamera görünmür:
```bash
ls /dev/video*              # USB kamera olmalıdır
libcamera-hello --list-cameras  # Pi kamerası üçün
sudo modprobe bcm2835-v4l2      # Pi kamera driver
```

### Səs çıxmır:
```bash
aplay -l                    # Səs cihazlarını listele
amixer set Master unmute    # Səsi aç
mpg123 -o alsa test.mp3     # mpg123 test
```

### YOLO modeli tapılmır:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Tesseract xətası:
```bash
tesseract --list-langs       # aze görünməldir
sudo apt install tesseract-ocr-aze
```

---

## İnkişaf Planı

| Mərhələ | Tarix | Hədəf |
|---|---|---|
| **v1.0** | 2024 Q1 | Prototipin yaradılması, YOLO + TTS |
| **v2.0** | 2024 Q3 | gemini Vision + Məkan analizi |
| **v3.0** | 2025 Q1 | Olimpiya Edisiyası — tam sistem |
| **v4.0** | 2025 Q3 | Xüsusi PCB, avadanlığın kiçildilməsi |
| **v5.0** | 2026 Q1 | Kütləvi istehsal, qlobal bazar |

---

*VisionVoiceAsist — Hər bir görmə məhdudiyyətli şəxs üçün müstəqillik.*

**© 2025 Əliəsgər. Bütün hüquqlar qorunur.**
