---
title: AI Surya Backend
emoji: 🧘
colorFrom: orange
colorTo: yellow
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# AI Suryanamaskar Backend

FastAPI backend for the AI Suryanamaskar Model.
- TensorFlow MobileNetV2 pose classifier
- MediaPipe pose landmark extraction
- WebSocket real-time inference at `/ws/practice`
- Health check at `/health`
- Session history at `/history`
