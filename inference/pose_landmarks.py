import cv2
import mediapipe as mp

from posture_utils import (
    calculate_joint_angles,
    format_angle
)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    raise SystemExit


print("✅ Webcam started")
print("Press 'q' to quit.")


# ============================================================
# POSE DETECTION
# ============================================================

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to read webcam frame.")
            break

        # ----------------------------------------------------
        # Mirror webcam
        # ----------------------------------------------------

        frame = cv2.flip(frame, 1)

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_frame.flags.writeable = False

        results = pose.process(rgb_frame)

        rgb_frame.flags.writeable = True

        # ====================================================
        # PROCESS DETECTED POSE
        # ====================================================

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # ------------------------------------------------
            # Calculate both-side joint angles
            # ------------------------------------------------

            angles = calculate_joint_angles(
                landmarks,
                mp_pose
            )

            # ------------------------------------------------
            # Draw skeleton
            # ------------------------------------------------

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # =================================================
            # DISPLAY LEFT SIDE
            # =================================================

            cv2.putText(
                frame,
                f"L Elbow: {format_angle(angles['left_elbow'])}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"L Knee: {format_angle(angles['left_knee'])}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"L Shoulder: {format_angle(angles['left_shoulder'])}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"L Hip: {format_angle(angles['left_hip'])}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # =================================================
            # DISPLAY RIGHT SIDE
            # =================================================

            cv2.putText(
                frame,
                f"R Elbow: {format_angle(angles['right_elbow'])}",
                (300, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"R Knee: {format_angle(angles['right_knee'])}",
                (300, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"R Shoulder: {format_angle(angles['right_shoulder'])}",
                (300, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"R Hip: {format_angle(angles['right_hip'])}",
                (300, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Tracking status
            # ------------------------------------------------

            cv2.putText(
                frame,
                "Pose Detected",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "No Pose Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "MediaPipe Bilateral Joint Analysis",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Webcam stopped.")