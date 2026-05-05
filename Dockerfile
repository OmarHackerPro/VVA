# ─────────────────────────────────────────────────────────────────────────────
# VisionVoiceAsist v5 — Multi-stage container image
# Target: ARM64 (Raspberry Pi 5)  /  x86-64 (CI / development)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libatlas-base-dev libopenblas-dev \
    libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Dependency manifest first so Docker can cache the pip layer
COPY pyproject.toml ./
COPY visionvoiceasist/__init__.py  ./visionvoiceasist/
COPY visionvoiceasist/py.typed     ./visionvoiceasist/

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install runtime + optional extras into an isolated prefix
COPY . .
RUN pip install --no-cache-dir --prefix=/install -e ".[iot,dashboard]"


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="VisionVoiceAsist" \
      org.opencontainers.image.version="5.0.0" \
      org.opencontainers.image.description="AI-powered wearable smart-glasses for visually impaired users" \
      org.opencontainers.image.authors="Əliəsgər Fatullayev"

WORKDIR /app

# Minimal runtime system packages
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libatlas3-base libopenblas0 \
    libjpeg62-turbo libpng16-16 \
    espeak-ng \
    tesseract-ocr tesseract-ocr-aze \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages + source from builder
COPY --from=builder /install /usr/local
COPY --from=builder /build   /app

# Non-root user for security
RUN useradd -r -s /sbin/nologin -u 999 vva \
    && mkdir -p /app/logs \
    && chown -R vva:vva /app

USER vva

ENV VVA_SHOW_GUI=false \
    VVA_DASHBOARD_ENABLED=true \
    VVA_OFFLINE_MODE=auto \
    VVA_LOG_LEVEL=INFO

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" \
        || exit 1

ENTRYPOINT ["vva"]
CMD ["--no-gui", "--dashboard"]
