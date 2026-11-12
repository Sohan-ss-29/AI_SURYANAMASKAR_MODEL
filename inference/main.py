import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import sys
import os
import time

from posture_validator import validate_pose
from realtime_features import calculate_real_time_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "suryanamaskar_mobilenetv2_best.keras"
)


# ============================================================
# POSE NAMES
# ============================================================

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
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING SURYA NAMASKARA MODEL")
print("=" * 60)

print("Model:", MODEL_PATH)

try:

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
        }
    )

except Exception as e:

    print("\nERROR loading model:")
    print(e)
    sys.exit(1)

print("Model loaded successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    sys.exit(1)


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "Surya Namaskara - Posture Correction"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ============================================================
# SETTINGS
# ============================================================

CLASSIFICATION_THRESHOLD = 0.50

previous_pose = None
pose_change_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read webcam frame.")
        break

    # --------------------------------------------------------
    # Mirror webcam
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)

    display = frame.copy()

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose_detector.process(rgb)

    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    predicted_pose = "Unknown"
    confidence = 0.0

    posture_valid = False
    posture_score = 0.0

    correction = "No pose detected"

    # --------------------------------------------------------
    # If body detected
    # --------------------------------------------------------

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # ====================================================
        # DRAW SKELETON
        # ====================================================

        mp_drawing.draw_landmarks(
            display,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # ====================================================
        # MOBILE NET CLASSIFICATION
        # ====================================================

        img = cv2.resize(
            frame,
            (224, 224)
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = img.astype(np.float32)

        img = np.expand_dims(
            img,
            axis=0
        )

        prediction = model.predict(
            img,
            verbose=0
        )[0]

        class_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[class_index]
        )

        if confidence >= CLASSIFICATION_THRESHOLD:

            predicted_pose = POSE_NAMES[class_index]

        else:

            predicted_pose = "Unknown"

        # ====================================================
        # GEOMETRIC VALIDATION
        # ====================================================

        if predicted_pose != "Unknown":

            features = calculate_real_time_features(
                landmarks,
                mp_pose
            )

            validation = validate_pose(
                features,
                predicted_pose
            )

            posture_valid = validation["valid"]

            posture_score = validation["score"]

            correction = validation["message"]

        # ====================================================
        # POSE CHANGE DETECTION
        # ====================================================

        if predicted_pose != previous_pose:

            previous_pose = predicted_pose
            pose_change_time = time.time()

    else:

        predicted_pose = "Unknown"
        correction = "Step into the camera."


    # ========================================================
    # DISPLAY
    # ========================================================

    h, w = display.shape[:2]

    # --------------------------------------------------------
    # Top title
    # --------------------------------------------------------

    cv2.putText(
        display,
        "SURYA NAMASKARA",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )

    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    pose_text = f"Pose: {predicted_pose}"

    cv2.putText(
        display,
        pose_text,
        (40, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence_text = (
        f"Classification confidence: "
        f"{confidence * 100:.1f}%"
    )

    cv2.putText(
        display,
        confidence_text,
        (40, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Posture score
    # --------------------------------------------------------

    score_text = (
        f"Posture score: "
        f"{posture_score * 100:.1f}%"
    )

    cv2.putText(
        display,
        score_text,
        (40, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Correct / Incorrect
    # --------------------------------------------------------

    if predicted_pose != "Unknown":

        if posture_valid:

            status = "CORRECT POSTURE"

            cv2.putText(
                display,
                status,
                (40, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        else:

            status = "ADJUST POSTURE"

            cv2.putText(
                display,
                status,
                (40, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    cv2.putText(
        display,
        correction,
        (40, 305),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    cv2.putText(
        display,
        "Press Q to quit",
        (40, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Show
    # --------------------------------------------------------

    cv2.imshow(
        WINDOW_NAME,
        display
    )

    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

pose_detector.close()

cv2.destroyAllWindows()

print("\nApplication closed.")