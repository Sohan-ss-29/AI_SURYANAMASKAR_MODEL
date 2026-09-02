import mediapipe as mp

print("MediaPipe version:", mp.__version__)

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

print("SUCCESS: MediaPipe Pose initialized!")

pose.close()

print("SUCCESS: MediaPipe Pose closed correctly!")