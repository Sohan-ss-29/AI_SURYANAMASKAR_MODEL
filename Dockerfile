# ============================================================
# Surya Namaskara AI Backend — Docker Image
# ============================================================
# Base: Python 3.11 slim (matches local dev; avoids TF/MP
# compatibility issues that exist on 3.12 for these versions)
# ============================================================

FROM python:3.11-slim

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================
# libgl1-mesa-glx  → required by opencv-python-headless
# libglib2.0-0     → required by mediapipe
# libgomp1         → required by TensorFlow (OpenMP)
# ============================================================

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# WORKING DIRECTORY
# ============================================================

WORKDIR /app

# ============================================================
# PYTHON DEPENDENCIES
# Install before copying source to leverage Docker layer cache
# ============================================================

COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# APPLICATION SOURCE
# Only copy what is needed at runtime — no training data,
# no .keras/.h5 weights (only the SavedModel is used)
# ============================================================

COPY backend/   ./backend/
COPY inference/ ./inference/
COPY models/suryanamaskar_mobilenetv2_savedmodel/ \
     ./models/suryanamaskar_mobilenetv2_savedmodel/
COPY models/class_names.json ./models/class_names.json

# ============================================================
# START SERVER
# Render provides $PORT at runtime (default 10000)
# uvicorn is started as a module so Python resolves
# `backend.server` from WORKDIR /app correctly
# ============================================================

CMD uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-10000}
