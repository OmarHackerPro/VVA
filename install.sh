#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  VisionVoiceAsist v3.0 — Avtomatik Quraşdırma Skripti
#  Raspberry Pi OS (Bookworm 64-bit) üçün optimallaşdırılmışdır
#  İstifadə: chmod +x install.sh && sudo ./install.sh
# ════════════════════════════════════════════════════════════════════════════

set -e  # Hər hansı xətada dayandır

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

banner() {
cat << 'EOF'

 ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗
 ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║
 ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║
 ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║
  ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║
   ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝

 VoiceAsist v3.0 — Quraşdırma Skripti
 Founder: Əliəsgər | Olympiad Edition

EOF
}

step()  { echo -e "${CYAN}${BOLD}[►]${NC} $1"; }
ok()    { echo -e "${GREEN}${BOLD}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}${BOLD}[!]${NC} $1"; }
fail()  { echo -e "${RED}${BOLD}[✗]${NC} $1"; exit 1; }

banner

# ── Kök icazəsi yoxlaması ────────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    fail "Bu skript root icazəsi tələb edir. sudo ./install.sh ilə işlədin."
fi

# ── Sistem yeniləmə ─────────────────────────────────────────────────────────
step "Sistem paketləri yenilənir..."
apt-get update -qq && apt-get upgrade -y -qq
ok "Sistem yeniləndi."

# ── Sistem asılılıqları ─────────────────────────────────────────────────────
step "Sistem asılılıqları quraşdırılır..."
apt-get install -y -qq \
    python3-pip python3-dev python3-venv \
    libopencv-dev python3-opencv \
    tesseract-ocr tesseract-ocr-aze tesseract-ocr-eng \
    mpg123 espeak espeak-data \
    libatlas-base-dev libhdf5-dev \
    libcamera-apps rpicam-apps \
    git curl wget unzip \
    libgpiod2 python3-gpiod \
    v4l-utils \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    2>/dev/null || warn "Bəzi paketlər quraşdırılmadı (normal ola bilər)"
ok "Sistem asılılıqları hazırdır."

# ── Kamera aktivləşdirməsi (Pi 5) ───────────────────────────────────────────
step "Kamera interfeysi yoxlanılır..."
if command -v raspi-config &>/dev/null; then
    raspi-config nonint do_camera 0 2>/dev/null || true
    ok "Kamera interfeysi aktiv edildi."
else
    warn "raspi-config tapılmadı (Pi olmayan sistem)."
fi

# ── Python virtual mühiti ───────────────────────────────────────────────────
VENV_DIR="/opt/visionvoiceasist"
step "Python virtual mühiti yaradılır: $VENV_DIR"
python3 -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"
ok "Virtual mühit aktiv."

# ── pip yeniləmə ────────────────────────────────────────────────────────────
step "pip yenilənir..."
pip install --upgrade pip setuptools wheel -q
ok "pip hazırdır."

# ── Python kitabxanaları ─────────────────────────────────────────────────────
step "Python asılılıqları quraşdırılır (bu bir neçə dəqiqə çəkə bilər)..."

pip install -q \
    "ultralytics>=8.2.0" \
    "google-generativeai>=0.30.0" \
    "requests>=2.32.0" \
    "psutil>=5.9.0" \
    "numpy>=1.26.0" \
    "pytesseract>=0.3.10" \
    "opencv-python-headless>=4.9.0" \
    "RPi.GPIO>=0.7.0" \
    2>/dev/null || warn "Bəzi paketlər quraşdırılmadı."

ok "Python kitabxanaları hazırdır."

# ── YOLOv8 modelini əvvəlcədən yüklə ───────────────────────────────────────
step "YOLOv8 nano modeli yüklənir..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null
ok "YOLOv8n modeli hazırdır."

# ── Layihə qovluğu ───────────────────────────────────────────────────────────
PROJECT_DIR="/home/$SUDO_USER/VisionVoiceAsist"
step "Layihə qovluğu hazırlanır: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"
cp main.py "$PROJECT_DIR/"
chown -R "$SUDO_USER:$SUDO_USER" "$PROJECT_DIR"
ok "Layihə qovluğu hazırdır."

# ── Başlatma skripti ─────────────────────────────────────────────────────────
step "Başlatma skripti yaradılır..."
cat > "$PROJECT_DIR/start.sh" << STARTSCRIPT
#!/bin/bash
source /opt/visionvoiceasist/bin/activate
cd $PROJECT_DIR
python3 main.py "\$@"
STARTSCRIPT
chmod +x "$PROJECT_DIR/start.sh"
chown "$SUDO_USER:$SUDO_USER" "$PROJECT_DIR/start.sh"
ok "start.sh hazırdır."

# ── systemd Xidməti (avtomatik başlatma) ─────────────────────────────────────
step "systemd xidməti quraşdırılır (boot-da avtomatik başlatma)..."
cat > /etc/systemd/system/visionvoiceasist.service << SERVICE
[Unit]
Description=VisionVoiceAsist - AI Gorma Yardimcisi
After=network.target sound.target
Wants=network.target

[Service]
Type=simple
User=$SUDO_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=/opt/visionvoiceasist/bin/python3 $PROJECT_DIR/main.py
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/$SUDO_USER/.Xauthority

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
ok "systemd xidməti quraşdırıldı."

# ── GPIO İcazəsi ─────────────────────────────────────────────────────────────
step "GPIO icazəsi verilir..."
usermod -aG gpio "$SUDO_USER" 2>/dev/null || true
usermod -aG video "$SUDO_USER" 2>/dev/null || true
usermod -aG audio "$SUDO_USER" 2>/dev/null || true
ok "İcazələr verildi."

# ── API Açarı Konfiqurasiyası ─────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}${BOLD}  API Açarı Konfiqurasiyası${NC}"
echo -e "${YELLOW}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo ""
read -p "  Gemini API açarınızı daxil edin ... " GEMINI_KEY
if [ -n "$GEMINI_KEY" ]; then
    sed -i "s|AIzaSyDyVH7h6JK7Hn0dzVlCMpA8W30NNdCuRks|$GEMINI_KEY|g" "$PROJECT_DIR/main.py"
    echo "Gemini açarı yazıldı."
else
    echo "Açar verilmədi, sonra əl ilə main.py faylına yazın."
fi
# ── Nəticə ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║         QURAŞDIRMA UĞURLA TAMAMLANDI! ✓                 ║${NC}"
echo -e "${GREEN}${BOLD}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}${BOLD}║  Başlatmaq üçün:                                         ║${NC}"
echo -e "${GREEN}${BOLD}║    cd $PROJECT_DIR                                ║${NC}"
echo -e "${GREEN}${BOLD}║    ./start.sh                                            ║${NC}"
echo -e "${GREEN}${BOLD}║                                                          ║${NC}"
echo -e "${GREEN}${BOLD}║  Boot-da avtomatik başlatmaq üçün:                       ║${NC}"
echo -e "${GREEN}${BOLD}║    sudo systemctl enable visionvoiceasist                ║${NC}"
echo -e "${GREEN}${BOLD}║    sudo systemctl start visionvoiceasist                 ║${NC}"
echo -e "${GREEN}${BOLD}║                                                          ║${NC}"
echo -e "${GREEN}${BOLD}║  Jurnalları görmək üçün:                                 ║${NC}"
echo -e "${GREEN}${BOLD}║    ls $PROJECT_DIR/logs/               ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
warn "Dəyişikliklərin tam qüvvəyə minməsi üçün sistemi yenidən başladın: sudo reboot"
