# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # OpenCV runtime libs
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # PDF rendering (PyMuPDF / pikepdf)
    libmupdf-dev \
    libfreetype6-dev \
    # PostgreSQL client lib
    libpq-dev \
    # Build tools (needed by some pip packages)
    gcc \
    g++ \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Upload directory ──────────────────────────────────────────────────────────
RUN mkdir -p /app/uploads

# ── Non-root user — matches the server's docuser ─────────────────────────────
# UID is passed at build time so file permissions align with the host user.
# Build with:  docker compose build --build-arg UID=$(id -u)
# Falls back to 1000 if not specified.
ARG UID=1000
RUN useradd -m -u ${UID} -s /bin/bash docuser \
 && chown -R docuser:docuser /app
USER docuser

# ── Default command (overridden per service in docker-compose) ────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
