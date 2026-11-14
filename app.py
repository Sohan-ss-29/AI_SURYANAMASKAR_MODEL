"""
HuggingFace Spaces entry point.

HuggingFace Gradio SDK detects the `app` variable (FastAPI + Gradio mounted)
and serves it with uvicorn on port 7860.

This file wraps our existing FastAPI backend without changing any of its logic.
"""

import sys
import os

# Ensure the project root is on sys.path so `backend` and `inference`
# packages resolve correctly when HuggingFace runs this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

# ============================================================
# IMPORT OUR EXISTING FASTAPI APP
# This triggers: TensorFlow model load + MediaPipe init
# (happens once at startup, same as before)
# ============================================================

from backend.server import app as fastapi_app

# ============================================================
# MINIMAL GRADIO INTERFACE
# Required so HuggingFace recognises this as a Gradio Space.
# Mounted at /ui so it doesn't conflict with our API routes.
# ============================================================

with gr.Blocks(title="Surya Namaskara AI Backend") as demo:
    gr.Markdown("## 🧘 Surya Namaskara AI Backend")
    gr.Markdown(
        "Backend is running.\n\n"
        "- **Health check:** `/health`\n"
        "- **WebSocket:** `/ws/practice`\n"
        "- **Session history:** `/history`"
    )

# ============================================================
# MOUNT GRADIO ON TOP OF FASTAPI
# All existing FastAPI routes (/ws/practice, /health, /history)
# remain fully intact. Gradio lives at /ui only.
# ============================================================

app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
