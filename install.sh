#!/usr/bin/env bash
# VisionVoiceAsist v5 — Raspberry Pi 5 installer + systemd service setup
# Usage: bash install.sh [--no-service] [--no-deps]
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Argument parsing ──────────────────────────────────────────────────────────
INSTALL_SERVICE=true
INSTALL_DEPS=true
for arg in "$@"; do
  case "$arg" in
    --no-service) INSTALL_SERVICE=false ;;
    --no-deps)    INSTALL_DEPS=false ;;
    --help|-h)
      echo "Usage: $0 [--no-service] [--no-deps]"
      echo "  --no-service  Skip systemd service installation"
      echo "  --no-deps     Skip apt / pip dependency installation"
      exit 0 ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
SERVICE_NAME="visionvoiceasist"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_DIR="$REPO_DIR/logs"
ENV_FILE="$REPO_DIR/.env"

# ── Sanity checks ─────────────────────────────────────────────────────────────
info "VisionVoiceAsist v5 installer"
info "Repo: $REPO_DIR"

if [[ "$(id -u)" -eq 0 ]]; then
  warn "Running as root. Service will be installed for root."
  SERVICE_USER="root"
else
  SERVICE_USER="$(id -un)"
fi

# Check Python 3.10+
if ! command -v python3 &>/dev/null; then
  error "python3 not found. Install with: sudo apt install python3"
  exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
  error "Python 3.10+ required. Found: $PY_VER"
  exit 1
fi
info "Python $PY_VER OK"

# ── System dependencies ───────────────────────────────────────────────────────
if $INSTALL_DEPS; then
  info "Installing system dependencies..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
      python3-venv python3-pip \
      libatlas-base-dev libopenblas-dev \
      libhdf5-dev libhdf5-serial-dev \
      libjpeg-dev libpng-dev libtiff-dev \
      espeak-ng \
      tesseract-ocr tesseract-ocr-aze \
      libportaudio2 portaudio19-dev \
      i2c-tools \
      git curl
    info "System deps installed"
  else
    warn "apt-get not found. Skipping system package install."
  fi
fi

# ── Python virtual environment ────────────────────────────────────────────────
info "Setting up Python virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ── Python packages ───────────────────────────────────────────────────────────
if $INSTALL_DEPS; then
  info "Installing Python packages (this may take several minutes on Pi 5)..."
  pip install -e "$REPO_DIR" -q

  # Hardware extra (GPIO) — only on Linux
  if [[ "$(uname -s)" == "Linux" ]]; then
    pip install "RPi.GPIO>=0.7" -q 2>/dev/null || warn "RPi.GPIO unavailable (not on RPi?)"
    pip install smbus2 -q || warn "smbus2 unavailable"
  fi

  info "Core packages installed"
fi

# ── .env file ─────────────────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "$REPO_DIR/.env.example" ]]; then
    cp "$REPO_DIR/.env.example" "$ENV_FILE"
    warn ".env created from .env.example — EDIT IT and add your API keys:"
    warn "  nano $ENV_FILE"
  else
    cat > "$ENV_FILE" <<'ENVEOF'
GEMINI_KEY=
ELEVENLABS_KEY=
VVA_CAM_INDEX=0
VVA_OFFLINE_MODE=auto
VVA_SHOW_GUI=false
VVA_LOG_LEVEL=INFO
ENVEOF
    warn ".env created with defaults — add API keys before first run"
  fi
else
  info ".env already exists — skipping"
fi

# ── Log directory ─────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
info "Log directory: $LOG_DIR"

# ── Systemd service ────────────────────────────────────────────────────────────
if $INSTALL_SERVICE && command -v systemctl &>/dev/null; then
  info "Installing systemd service: $SERVICE_NAME"

  sudo tee "$SERVICE_FILE" > /dev/null <<SVCEOF
[Unit]
Description=VisionVoiceAsist v5 — AI Smart Glasses
Documentation=file://${REPO_DIR}/ARCHITECTURE.md
After=network.target sound.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${VENV_DIR}/bin/vva --no-gui
Restart=on-failure
RestartSec=5s
StandardOutput=append:${LOG_DIR}/service.log
StandardError=append:${LOG_DIR}/service.log
KillMode=mixed
TimeoutStopSec=10s

# Resource limits for Raspberry Pi 5
CPUQuota=90%
MemoryMax=3G
Nice=-5

[Install]
WantedBy=multi-user.target
SVCEOF

  sudo systemctl daemon-reload
  sudo systemctl enable "$SERVICE_NAME"
  info "Service enabled. Commands:"
  info "  sudo systemctl start $SERVICE_NAME   # start now"
  info "  sudo systemctl status $SERVICE_NAME  # check status"
  info "  journalctl -u $SERVICE_NAME -f       # follow logs"
else
  if ! $INSTALL_SERVICE; then
    info "Skipping service install (--no-service)"
  else
    warn "systemctl not found — skipping service install"
  fi
fi

# ── I2C / camera / audio permissions ─────────────────────────────────────────
if [[ "$(uname -s)" == "Linux" ]] && id -nG "$SERVICE_USER" | grep -qv "i2c"; then
  info "Adding $SERVICE_USER to i2c, gpio, audio, video groups..."
  for grp in i2c gpio audio video spi; do
    sudo usermod -aG "$grp" "$SERVICE_USER" 2>/dev/null || true
  done
  warn "Group changes take effect after logout/login (or reboot)"
fi

# ── Enable I2C on Raspberry Pi ────────────────────────────────────────────────
if [[ -f /boot/firmware/config.txt ]] && ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt; then
  info "Enabling I2C in /boot/firmware/config.txt..."
  echo "dtparam=i2c_arm=on" | sudo tee -a /boot/firmware/config.txt > /dev/null
  warn "I2C enabled — reboot required for MPU-6050"
fi

# ── Final summary ──────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN} VisionVoiceAsist v5 — Installation Complete${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Edit .env and add your API keys:"
echo "    nano $ENV_FILE"
echo ""
echo "  Run manually:"
echo "    source $VENV_DIR/bin/activate"
echo "    vva --no-gui"
echo ""
if $INSTALL_SERVICE && command -v systemctl &>/dev/null; then
  echo "  Start as service:"
  echo "    sudo systemctl start $SERVICE_NAME"
  echo ""
fi
echo "  View logs:"
echo "    tail -f $LOG_DIR/service.log"
echo ""
