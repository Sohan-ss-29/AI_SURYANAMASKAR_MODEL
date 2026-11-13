import base64
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.ai_processor import AIProcessor


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Surya AI Backend",
    description="Real-time Surya Namaskara AI Backend",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AI
# ============================================================

ai = AIProcessor()


# ============================================================
# HTTP ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Surya AI backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ai": "ready"
    }


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws/practice")
async def practice(websocket: WebSocket):

    print("\n========================================")
    print("WebSocket connection attempt received")
    print("========================================")

    await websocket.accept()

    print("Browser connected to AI practice.")

    try:

        await websocket.send_json({
            "type": "connection",
            "success": True,
            "message": "AI WebSocket connected"
        })

        while True:

            data = await websocket.receive_text()

            print(
                f"Received frame: {len(data)} characters"
            )

            try:

                image_data = base64.b64decode(
                    data
                )

                image_array = np.frombuffer(
                    image_data,
                    dtype=np.uint8
                )

                frame = cv2.imdecode(
                    image_array,
                    cv2.IMREAD_COLOR
                )

                if frame is None:

                    await websocket.send_json({
                        "success": False,
                        "message": "Invalid image frame"
                    })

                    continue

                result = ai.process_frame(
                    frame
                )

                await websocket.send_json(
                    result
                )

            except Exception as frame_error:

                print(
                    "Frame processing error:",
                    repr(frame_error)
                )

                await websocket.send_json({
                    "success": False,
                    "message": "Frame processing failed"
                })

    except WebSocketDisconnect:

        print("Browser disconnected.")

    except Exception as e:

        print(
            "WebSocket error:",
            repr(e)
        )

        try:
            await websocket.close()
        except Exception:
            pass