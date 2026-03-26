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

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

# ── Default command (overridden per service in docker-compose) ─────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
