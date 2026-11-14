# ============================================================
# Surya Namaskara AI Backend — Hugging Face Spaces Docker Image
# ============================================================
# Base: Python 3.11 slim
# Port: 7860 (required by Hugging Face Spaces)
# User: non-root uid 1000 (required by Hugging Face Spaces)
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
# NON-ROOT USER (Hugging Face Spaces requirement)
# ============================================================

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY --chown=user backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# APPLICATION SOURCE
# ============================================================

COPY --chown=user backend/   ./backend/
COPY --chown=user inference/ ./inference/
COPY --chown=user models/suryanamaskar_mobilenetv2_savedmodel/ \
     ./models/suryanamaskar_mobilenetv2_savedmodel/
COPY --chown=user models/class_names.json ./models/class_names.json

# ============================================================
# ENVIRONMENT & USER SWITCH
# ============================================================

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# ============================================================
# START SERVER
# ============================================================

CMD uvicorn backend.server:app --host 0.0.0.0 --port 7860
