import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import sys
import os
import time

from posture_validator import validate_pose
from realtime_features import calculate_real_time_features
from sequence_controller import SequenceController


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "suryanamaskar_mobilenetv2_savedmodel"
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

    model = tf.saved_model.load(
        MODEL_PATH
    )

    inference_fn = model.signatures[
        "serving_default"
    ]

except Exception as e:

    print("\nERROR loading SavedModel:")
    print(e)
    sys.exit(1)

print("SavedModel loaded successfully.")


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

WINDOW_NAME = (
    "Surya Namaskara - Posture Correction"
)

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

# ------------------------------------------------------------
# Classification threshold
#
# This is intentionally lower than a normal classifier
# confidence threshold because the sequence controller
# and posture validator provide additional validation.
# ------------------------------------------------------------

CLASSIFICATION_THRESHOLD = 0.35


# ------------------------------------------------------------
# Hold settings
# ------------------------------------------------------------

HOLD_TIME = 2.0

HOLD_GRACE_TIME = 0.8

WRONG_POSE_FRAMES = 4


# ============================================================
# FEEDBACK DISPLAY SETTINGS
# ============================================================

# ------------------------------------------------------------
# This controls ONLY how long a non-holding feedback message
# remains visible on the screen.
#
# It does NOT change:
# - classifier
# - posture validation
# - sequence controller
# - hold timer
# - grace period
# - wrong pose detection
#
# The internal system still processes every frame normally.
# ------------------------------------------------------------

FEEDBACK_MIN_DISPLAY_TIME = 1.0


# ============================================================
# SEQUENCE CONTROLLER
# ============================================================

sequence_controller = SequenceController(
    total_poses=12,
    hold_time=HOLD_TIME,
    wrong_pose_frames=WRONG_POSE_FRAMES,
    hold_grace_time=HOLD_GRACE_TIME
)


# ============================================================
# FEEDBACK DISPLAY STATE
# ============================================================

# This stores the message currently shown to the user.

last_feedback = ""

# Time at which the current displayed message started.

last_feedback_time = 0.0


# ============================================================
# SESSION SCORE TRACKING
# ============================================================

# Stores the final score for each completed step.

step_scores = {}

# Temporary values used while holding the current pose.

current_step_score_sum = 0.0

current_step_score_count = 0

# Final overall score.

overall_score = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # READ FRAME
    # ========================================================

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Cannot read webcam frame."
        )

        break


    # ========================================================
    # MIRROR WEBCAM
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )

    display = frame.copy()


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose_detector.process(
        rgb
    )


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    raw_pose = None

    raw_pose_name = "Unknown"

    confidence = 0.0

    posture_valid = False

    posture_score = 0.0

    correction = ""

    sequence_state = "waiting"

    sequence_feedback = ""


    # ========================================================
    # GET CURRENT EXPECTED POSE
    # ========================================================

    expected_pose_id = (
        sequence_controller.get_expected_pose()
    )

    # Safety check in case the session is complete.

    if expected_pose_id is not None:

        expected_pose_name = (
            POSE_NAMES[
                expected_pose_id - 1
            ]
        )

    else:

        expected_pose_name = (
            "Completed"
        )


    # ========================================================
    # BODY DETECTED
    # ========================================================

    if results.pose_landmarks:

        landmarks = (
            results.pose_landmarks.landmark
        )


        # ====================================================
        # DRAW MEDIAPIPE SKELETON
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

        img = img.astype(
            np.float32
        )

        img = np.expand_dims(
            img,
            axis=0
        )


        # ====================================================
        # SAVEDMODEL INFERENCE
        # ====================================================

        prediction = inference_fn(
            input_image=tf.constant(img)
        )["output_0"].numpy()[0]


        # ====================================================
        # HIGHEST PROBABILITY CLASS
        # ====================================================

        class_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[class_index]
        )


        # ====================================================
        # RAW CLASSIFIER RESULT
        # ====================================================

        raw_pose_name = (
            POSE_NAMES[
                class_index
            ]
        )


        # ====================================================
        # CLASSIFICATION THRESHOLD
        # ====================================================

        if (
            confidence >=
            CLASSIFICATION_THRESHOLD
        ):

            raw_pose = (
                class_index + 1
            )

        else:

            raw_pose = None


        # ====================================================
        # POSTURE VALIDATION
        # ====================================================

        # Validate only when the classifier currently
        # agrees with the expected sequence pose.

        if (
            raw_pose ==
            expected_pose_id
        ):

            features = (
                calculate_real_time_features(
                    landmarks,
                    mp_pose
                )
            )

            validation = (
                validate_pose(
                    features,
                    expected_pose_name
                )
            )

            posture_valid = (
                validation["valid"]
            )

            posture_score = (
                validation["score"]
            )

            correction = (
                validation["message"]
            )


        # ====================================================
        # COLLECT SCORE DURING VALID HOLD
        # ====================================================

        if (
            expected_pose_id is not None
            and posture_valid
        ):

            current_step_score_sum += posture_score

            current_step_score_count += 1


        # ====================================================
        # SEQUENCE CONTROLLER
        # ====================================================

        sequence_result = (
            sequence_controller.update(
                raw_pose,
                posture_valid
            )
        )

        sequence_state = (
            sequence_result["state"]
        )

        sequence_feedback = (
            sequence_result["message"]
        )


        # ====================================================
        # SAVE COMPLETED POSE SCORE
        # ====================================================

        completed_step = (
            sequence_result["completed_step"]
        )

        if completed_step is not None:

            if current_step_score_count > 0:

                average_score = (
                    current_step_score_sum /
                    current_step_score_count
                )

            else:

                average_score = 0.0

            step_scores[completed_step] = (
                average_score
            )

            # Reset temporary score tracking
            # for the next pose.

            current_step_score_sum = 0.0

            current_step_score_count = 0

            # Calculate overall score
            # whenever a step is completed.

            if len(step_scores) > 0:

                overall_score = (
                    sum(step_scores.values()) /
                    len(step_scores)
                )


        # ====================================================
        # USER FEEDBACK
        # ====================================================

        # ----------------------------------------------------
        # EXPECTED POSE BUT POSTURE NEEDS ADJUSTMENT
        # ----------------------------------------------------

        if (
            sequence_state ==
            "correct_pose_adjust"
        ):

            if correction:

                sequence_feedback = (
                    correction
                )

            else:

                sequence_feedback = (
                    f"Adjust your posture "
                    f"for Pose "
                    f"{expected_pose_id}."
                )


        # ----------------------------------------------------
        # CORRECT POSTURE / HOLDING
        # ----------------------------------------------------

        elif (
            sequence_state ==
            "holding"
        ):

            hold_elapsed = (
                sequence_result[
                    "hold_elapsed"
                ]
            )

            sequence_feedback = (
                f"Good! Hold Pose "
                f"{expected_pose_id} "
                f"for "
                f"{hold_elapsed:.1f} / "
                f"{HOLD_TIME:.1f} sec"
            )


        # ----------------------------------------------------
        # POSE COMPLETED
        # ----------------------------------------------------

        elif (
            sequence_state ==
            "pose_complete"
        ):

            sequence_feedback = (
                sequence_result[
                    "message"
                ]
            )


        # ----------------------------------------------------
        # WRONG POSE
        # ----------------------------------------------------

        elif (
            sequence_state ==
            "wrong_pose"
        ):

            sequence_feedback = (
                f"WRONG POSE! "
                f"Pose {expected_pose_id} "
                f"should be performed here."
            )


        # ----------------------------------------------------
        # TEMPORARY DETECTION LOSS
        # ----------------------------------------------------

        elif (
            sequence_state ==
            "waiting"
        ):

            sequence_feedback = (
                sequence_result[
                    "message"
                ]
            )


        # ----------------------------------------------------
        # CHECKING
        # ----------------------------------------------------

        elif (
            sequence_state ==
            "checking"
        ):

            sequence_feedback = (
                sequence_result[
                    "message"
                ]
            )


        # ----------------------------------------------------
        # SESSION COMPLETE
        # ----------------------------------------------------

        elif (
            sequence_state ==
            "complete"
        ):

            sequence_feedback = (
                "Excellent! "
                "Surya Namaskara "
                "completed."
            )


    # ========================================================
    # NO BODY DETECTED
    # ========================================================

    else:

        raw_pose = None

        raw_pose_name = (
            "Unknown"
        )

        confidence = 0.0

        posture_valid = False

        posture_score = 0.0

        sequence_result = (
            sequence_controller.update(
                None,
                False
            )
        )

        sequence_state = (
            sequence_result[
                "state"
            ]
        )

        # If there is no person in the camera,
        # keep the instruction simple.

        sequence_feedback = (
            "Step into the camera."
        )


    # ========================================================
    # STABILIZE DISPLAYED FEEDBACK
    # ========================================================

    # IMPORTANT:
    #
    # The sequence controller still changes every frame.
    #
    # We only prevent the TEXT displayed to the user from
    # changing every frame.
    #
    # This means messages such as:
    #
    #   "Perform Step 2."
    #   "Checking..."
    #   "Wrong pose!"
    #
    # will no longer rapidly replace each other.
    #
    # The holding message is allowed to update continuously
    # because its timer needs to increase.
    #
    # ========================================================

    current_time = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # HOLDING
    #
    # Always update holding message because the timer must
    # remain live.
    # --------------------------------------------------------

    if (
        sequence_state ==
        "holding"
    ):

        last_feedback = (
            sequence_feedback
        )

        last_feedback_time = (
            current_time
        )


    # --------------------------------------------------------
    # IMPORTANT EVENTS
    #
    # These should appear immediately.
    # --------------------------------------------------------

    elif sequence_state in (
        "pose_complete",
        "complete",
        "correct_pose_adjust"
    ):

        last_feedback = (
            sequence_feedback
        )

        last_feedback_time = (
            current_time
        )


    # --------------------------------------------------------
    # NORMAL / TRANSIENT MESSAGES
    #
    # Only allow a new message to replace the current displayed
    # message after the minimum display time has passed.
    # --------------------------------------------------------

    elif (
        sequence_feedback !=
        last_feedback
    ):

        if (
            current_time -
            last_feedback_time
        ) >= (
            FEEDBACK_MIN_DISPLAY_TIME
        ):

            last_feedback = (
                sequence_feedback
            )

            last_feedback_time = (
                current_time
            )


    # --------------------------------------------------------
    # USE THE STABILIZED MESSAGE
    # --------------------------------------------------------

    sequence_feedback = (
        last_feedback
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    h, w = display.shape[:2]


    # ========================================================
    # FINAL SESSION REPORT
    # ========================================================

    if sequence_state == "complete":

        # Dark background using a filled rectangle.

        cv2.rectangle(
            display,
            (80, 80),
            (w - 80, h - 80),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            "SURYA NAMASKARA COMPLETE",
            (140, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            display,
            f"Overall Score: {overall_score * 100:.1f}%",
            (140, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # INDIVIDUAL POSE SCORES
        # ----------------------------------------------------

        for i in range(1, 13):

            pose_score = (
                step_scores.get(
                    i,
                    0.0
                )
            )

            # Poses 1-6 in the left column,
            # poses 7-12 in the right column.

            column = 0 if i <= 6 else 1

            row = (i - 1) % 6

            score_x = 140 + column * 350

            score_y = 270 + row * 40

            cv2.putText(
                display,
                f"Pose {i}: {pose_score * 100:.1f}%",
                (score_x, score_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

        cv2.putText(
            display,
            "Excellent performance!",
            (140, 540),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "R = Start Again    Q = Quit",
            (140, h - 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    if sequence_state != "complete":

        # ====================================================
        # TEXT POSITIONS
        # ====================================================

        x = 40

        y_title = 60
        y_step = 110
        y_expected = 150
        y_detected = 190
        y_score = 230
        y_status = 285
        y_feedback = 340


        # ====================================================
        # TITLE
        # ====================================================

        cv2.putText(
            display,
            "SURYA NAMASKARA",
            (x, y_title),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )


        # ====================================================
        # STEP
        # ====================================================

        expected_step = (
            sequence_controller.get_expected_step()
        )

        step_text = (
            f"Step {expected_step} / 12"
        )

        cv2.putText(
            display,
            step_text,
            (x, y_step),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )


        # ====================================================
        # EXPECTED POSE
        # ====================================================

        expected_pose_text = (
            f"Expected: "
            f"{expected_pose_name}"
        )

        cv2.putText(
            display,
            expected_pose_text,
            (x, y_expected),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DETECTED POSE
        # ====================================================

        if raw_pose is not None:

            detected_text = (
                f"Detected: "
                f"{raw_pose_name} "
                f"({confidence * 100:.1f}%)"
            )

        else:

            detected_text = (
                f"Detected: Unknown "
                f"({confidence * 100:.1f}%)"
            )


        cv2.putText(
            display,
            detected_text,
            (x, y_detected),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 200, 0),
            2
        )


        # ====================================================
        # POSTURE SCORE
        # ====================================================

        score_text = (
            f"Posture score: "
            f"{posture_score * 100:.1f}%"
        )

        cv2.putText(
            display,
            score_text,
            (x, y_score),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ====================================================
        # STATUS
        # ====================================================

        if (
            sequence_state ==
            "holding"
        ):

            status = (
                "CORRECT POSTURE"
            )

            status_color = (
                0,
                255,
                0
            )


        elif (
            sequence_state ==
            "correct_pose_adjust"
        ):

            status = (
                "ADJUST POSTURE"
            )

            status_color = (
                0,
                0,
                255
            )


        elif (
            sequence_state ==
            "wrong_pose"
        ):

            status = (
                "WRONG POSE"
            )

            status_color = (
                0,
                0,
                255
            )


        elif (
            sequence_state ==
            "pose_complete"
        ):

            status = (
                "POSE COMPLETE"
            )

            status_color = (
                0,
                255,
                0
            )


        elif (
            sequence_state ==
            "complete"
        ):

            status = (
                "SESSION COMPLETE"
            )

            status_color = (
                0,
                255,
                0
            )


        else:

            status = (
                "READY"
            )

            status_color = (
                255,
                255,
                0
            )


        cv2.putText(
            display,
            status,
            (x, y_status),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            status_color,
            3
        )


        # ====================================================
        # FEEDBACK
        # ====================================================

        # Prevent extremely long text from running across
        # the entire webcam frame.

        feedback = (
            sequence_feedback
        )

        if len(feedback) > 65:

            feedback = (
                feedback[:62] +
                "..."
            )


        cv2.putText(
            display,
            feedback,
            (x, y_feedback),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 0),
            2
        )


        # ====================================================
        # BOTTOM INSTRUCTION
        # ====================================================

        cv2.putText(
            display,
            "R = Reset Sequence    Q = Quit",
            (x, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = (
        cv2.waitKey(1) &
        0xFF
    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if key == ord("r"):

        sequence_controller.reset()

        # Clear the old feedback so the new sequence starts
        # with a clean message.

        last_feedback = ""
        last_feedback_time = 0.0

        # Reset session scores.

        step_scores = {}

        current_step_score_sum = 0.0

        current_step_score_count = 0

        overall_score = 0.0

        print(
            "\nSequence reset to Pose 1."
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

pose_detector.close()

cv2.destroyAllWindows()

print(
    "\nApplication closed."
)