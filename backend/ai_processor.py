import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# IMPORTS
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


# ============================================================
# AI PROCESSOR
# ============================================================

class AIProcessor:

    def __init__(self):

        print("")
        print("=" * 50)
        print("Loading Surya AI...")
        print("=" * 50)

        # Load TensorFlow SavedModel
        self.model = tf.saved_model.load(
            MODEL_PATH
        )

        self.infer = self.model.signatures[
            "serving_default"
        ]

        print("MobileNetV2 loaded.")

        # MediaPipe
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("MediaPipe loaded.")
        print("Surya AI ready.")
        print("=" * 50)

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, frame, expected_pose_id=None):

        if frame is None:
            return {
                "success": False,
                "message": "Invalid frame"
            }

        # ----------------------------------------------------
        # 1. MEDIA PIPE
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(
            rgb
        )

        if not results.pose_landmarks:

            return {
                "success": True,
                "pose_detected": False,
                "pose_id": None,
                "pose_name": "",
                "classification_confidence": 0.0,
                "posture_valid": False,
                "posture_score": 0.0,
                "message": "No person detected"
            }

        landmarks = (
            results.pose_landmarks.landmark
        )

        # ----------------------------------------------------
        # 2. PREPARE IMAGE FOR MOBILE NET
        # ----------------------------------------------------

        image = cv2.resize(
            frame,
            (224, 224)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
            np.float32
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        # ----------------------------------------------------
        # 3. CLASSIFICATION
        # ----------------------------------------------------

        tensor = tf.constant(
            image,
            dtype=tf.float32
        )

        output = self.infer(
            input_image=tensor
        )

        predictions = list(
            output.values()
        )[0].numpy()[0]

        top_indices = np.argsort(predictions)[::-1][:3]

        print(
            "Top predictions:",
            [
                (
                    int(i + 1),
                    POSE_NAMES[i],
                    round(float(predictions[i]), 4)
                )
                for i in top_indices
            ]
        )

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

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        top_indices = np.argsort(
            predictions
        )[::-1][:3]

        top_predictions = []

        for index in top_indices:

            top_predictions.append({
                "pose_id": int(index + 1),
                "confidence": round(
                    float(predictions[index]),
                    4
                )
            })

        print(
            f"Prediction: Pose {pose_id} "
            f"({classification_confidence * 100:.1f}%) "
            f"| Top: {top_predictions}"
        )

        # ----------------------------------------------------
        # 4. REAL-TIME FEATURES
        # ----------------------------------------------------

        features = calculate_real_time_features(
            landmarks,
            self.mp_pose
        )

        # ----------------------------------------------------
        # 5. POSTURE VALIDATION
        # ----------------------------------------------------

        validation = validate_pose(
            features,
            pose_name
        )

        effective_pose_id = None

        effective_pose_name = ""

        effective_posture_valid = False

        effective_posture_score = 0.0

        effective_message = ""

        if (
            expected_pose_id is not None
            and pose_id == expected_pose_id
        ):

            # Classifier agrees with the expected
            # sequence pose.

            effective_pose_id = pose_id

            effective_pose_name = pose_name

            effective_posture_valid = (
                validation["valid"]
            )

            effective_posture_score = (
                validation["score"]
            )

            effective_message = (
                validation["message"]
            )

        else:

            # Classifier predicted some other pose.
            # Do NOT expose that pose to the sequence UI.

            effective_pose_id = None

            effective_pose_name = ""

            effective_posture_valid = False

            effective_posture_score = 0.0

            if expected_pose_id is not None:

                effective_message = (
                    f"Perform Pose {expected_pose_id}."
                )

            else:

                effective_message = (
                    "Waiting for pose."
                )

        # ----------------------------------------------------
        # 6. RESULT
        # ----------------------------------------------------

        return {

            "success": True,

            "pose_detected":
                effective_pose_id is not None,

            "pose_id":
                effective_pose_id,

            "pose_name":
                effective_pose_name,

            "classification_confidence": round(
                classification_confidence,
                4
            ),

            "posture_valid":
                effective_posture_valid,

            "posture_score": round(
                effective_posture_score,
                4
            ),

            "passed_rules":
                validation["passed_rules"]
                if effective_pose_id is not None
                else 0,

            "available_rules":
                validation["available_rules"]
                if effective_pose_id is not None
                else 0,

            "message":
                effective_message,

            "top_predictions":
                top_predictions
        }

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.pose.close()