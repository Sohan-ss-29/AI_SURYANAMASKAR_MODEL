import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# EXISTING AI MODULES
# ============================================================

from inference.posture_validator import validate_pose
from inference.realtime_features import (
    calculate_real_time_features
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "suryanamaskar_mobilenetv2_savedmodel"
)


POSE_NAMES = [
    "1_Pranamasana",
    "2_Hasta_Uttanasana",
    "3_Padahastasana",
    "4_AshwaSanchalanasana",
    "5_Parvatasana",
    "6_Ashtanga_Namaskara",
    "7_Bhujangasana",
]


class AIProcessor:

    def __init__(self):

        print("")
        print("=" * 50)
        print("Loading Surya AI...")
        print("=" * 50)

        # ----------------------------------------------------
        # Load SavedModel
        # ----------------------------------------------------

        self.model = tf.saved_model.load(
            MODEL_PATH
        )

        self.infer = self.model.signatures[
            "serving_default"
        ]

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("MobileNetV2 loaded.")
        print("MediaPipe loaded.")
        print("Surya AI ready.")
        print("=" * 50)


    # ========================================================
    # PROCESS ONE FRAME
    # ========================================================

    def process_frame(self, frame):

        if frame is None:

            return {
                "success": False,
                "message": "Invalid frame"
            }


        # ====================================================
        # 1. MEDIA PIPE
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(
            rgb
        )


        # ----------------------------------------------------
        # No person detected
        # ----------------------------------------------------

        if not results.pose_landmarks:

            return {
                "success": True,
                "pose_detected": False,
                "message": "No person detected"
            }


        landmarks = results.pose_landmarks.landmark


        # ====================================================
        # 2. CLASSIFICATION IMAGE
        # ====================================================

        image = cv2.resize(
            rgb,
            (224, 224)
        )

        image = image.astype(
            np.float32
        ) / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )


        # ====================================================
        # 3. MOBILE NET CLASSIFICATION
        # ====================================================

        output = self.infer(
            input_image=tf.constant(
                image,
                dtype=tf.float32
            )
        )

        predictions = list(
            output.values()
        )[0].numpy()[0]


        class_index = int(
            np.argmax(predictions)
        )

        pose_id = class_index + 1

        classification_confidence = float(
            predictions[class_index]
        )

        pose_name = POSE_NAMES[
            class_index
        ]


        # ====================================================
        # 4. REAL-TIME BODY FEATURES
        # ====================================================

        features = calculate_real_time_features(
            landmarks,
            self.mp_pose
        )


        # ====================================================
        # 5. POSTURE VALIDATION
        # ====================================================

        validation = validate_pose(
            features,
            pose_name
        )


        # ====================================================
        # 6. RETURN RESULT
        # ====================================================

        return {

            "success": True,

            "pose_detected": True,

            "pose_id": pose_id,

            "pose_name": pose_name,

            "classification_confidence":
                round(
                    classification_confidence,
                    4
                ),

            "posture_valid":
                validation["valid"],

            "posture_score":
                round(
                    validation["score"],
                    4
                ),

            "passed_rules":
                validation["passed_rules"],

            "available_rules":
                validation["available_rules"],

            "message":
                validation["message"]
        }


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.pose.close()