import base64
import cv2
import numpy as np

import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect



from ai_processor import AIProcessor
from inference.sequence_controller import SequenceController

from inference.session_manager import SessionManager


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
        "https://aisuryanamaskarmodel.vercel.app",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AI PROCESSOR
# ============================================================

ai = AIProcessor()


# ============================================================
# HEALTH
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
# HISTORY
# ============================================================

@app.get("/history")
def history():

    manager = SessionManager(BASE_DIR)

    return {
        "sessions": manager.load_history()
    }


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws/practice")
async def practice(websocket: WebSocket):

    print("WebSocket connection attempt received.")

    await websocket.accept()

    print("Browser connected to AI practice.")

    # --------------------------------------------------------
    # Create a separate sequence controller for this session
    # --------------------------------------------------------

    sequence = SequenceController(
        total_poses=12,
        hold_time=2.0,
        wrong_pose_frames=4,
        hold_grace_time=0.8
    )
    session_manager = SessionManager(BASE_DIR)

    await websocket.send_json({
        "success": True,
        "message": "WebSocket connected successfully"
    })

    try:

        while True:

            # =================================================
            # RECEIVE IMAGE
            # =================================================

            data = await websocket.receive_text()

            # =================================================
            # DECODE IMAGE
            # =================================================

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

            # =================================================
            # RUN AI
            # =================================================

            result = ai.process_frame(
                frame,
                expected_pose_id=sequence.get_expected_pose()
            )

            # =================================================
            # AI DID NOT DETECT PERSON
            # =================================================

            if not result.get(
                "pose_detected",
                False
            ):

                sequence_result = sequence.update(
                    raw_pose=None,
                    posture_valid=False 
                )

            # =================================================
            # AI DETECTED PERSON
            # =================================================

            else:

                sequence_result = sequence.update(
                    raw_pose=result["pose_id"],
                    posture_valid=result["posture_valid"]
                )

                if sequence_result["completed_step"] is not None:

                    completed_step = (
                        sequence_result["completed_step"]
                    )

                    session_manager.record_pose(
                        completed_step,
                        result.get("posture_score", 0.0)
                    )

                if sequence_result["session_complete"]:

                    session = session_manager.save_session()

                    result["session_saved"] = True

                    result["session"] = session

            # =================================================
            # ADD SEQUENCE INFORMATION
            # =================================================

            result["expected_step"] = (
                sequence_result["expected_step"]
            )

            result["expected_pose"] = (
                sequence_result["expected_pose"]
            )

            result["sequence_state"] = (
                sequence_result["state"]
            )

            result["sequence_message"] = (
                sequence_result["message"]
            )

            result["hold_elapsed"] = (
                sequence_result["hold_elapsed"]
            )

            result["completed_step"] = (
                sequence_result["completed_step"]
            )

            result["session_complete"] = (
                sequence_result["session_complete"]
            )

            # =================================================
            # SEND RESULT TO FRONTEND
            # =================================================

            await websocket.send_json(
                result
            )

    except WebSocketDisconnect:

        print("")
        print("Browser disconnected.")

    except Exception as e:

        print(
            "WebSocket error:",
            repr(e)
        )

        try:
            await websocket.close()
        except:
            pass