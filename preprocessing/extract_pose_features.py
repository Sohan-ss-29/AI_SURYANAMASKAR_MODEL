import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(
    os.path.join(PROJECT_ROOT, "inference")
)

from posture_utils import calculate_joint_angles


# ============================================================
# PATHS
# ============================================================

# Usage:
#
# Train:
# python preprocessing\extract_pose_features.py
#
# Validation:
# python preprocessing\extract_pose_features.py dataset_split\val reports\val_pose_features.csv
#
# Test:
# python preprocessing\extract_pose_features.py dataset_split\test reports\test_pose_features.csv


if len(sys.argv) > 1:

    DATASET_DIR = os.path.join(
        PROJECT_ROOT,
        sys.argv[1]
    )

else:

    DATASET_DIR = os.path.join(
        PROJECT_ROOT,
        "dataset_split",
        "train"
    )


if len(sys.argv) > 2:

    OUTPUT_CSV = os.path.join(
        PROJECT_ROOT,
        sys.argv[2]
    )

else:

    OUTPUT_CSV = os.path.join(
        PROJECT_ROOT,
        "reports",
        "train_pose_features.csv"
    )


os.makedirs(
    os.path.join(
        PROJECT_ROOT,
        "reports"
    ),
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

VISIBILITY_THRESHOLD = 0.30

POSES = [
    "1_Pranamasana",
    "2_Hasta_Uttanasana",
    "3_Padahastasana",
    "4_AshwaSanchalanasana",
    "5_Parvatasana",
    "6_Ashtanga_Namaskara",
    "7_Bhujangasana"
]


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_point(landmarks, landmark_id):

    lm = landmarks[landmark_id]

    return np.array(
        [lm.x, lm.y],
        dtype=np.float32
    )


def visible(landmarks, landmark_id):

    return (
        landmarks[landmark_id].visibility
        >= VISIBILITY_THRESHOLD
    )


def normalized_distance(p1, p2, scale):

    if (
        p1 is None
        or p2 is None
        or scale is None
        or scale <= 1e-6
    ):
        return np.nan

    return (
        np.linalg.norm(p1 - p2)
        / scale
    )


def safe_difference(a, b):

    if pd.isna(a) or pd.isna(b):
        return np.nan

    return abs(a - b)


def relative_y(point_y, reference_y):

    if (
        pd.isna(point_y)
        or pd.isna(reference_y)
    ):
        return np.nan

    return point_y - reference_y


def relative_x(point_x, reference_x):

    if (
        pd.isna(point_x)
        or pd.isna(reference_x)
    ):
        return np.nan

    return point_x - reference_x


# ============================================================
# EXTRACT FEATURES FROM ONE IMAGE
# ============================================================

def extract_features(image, pose_detector):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = pose_detector.process(rgb)

    if not results.pose_landmarks:
        return None

    landmarks = (
        results.pose_landmarks.landmark
    )

    # --------------------------------------------------------
    # Required landmarks
    # --------------------------------------------------------

    LS = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RS = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

    LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
    RE = mp_pose.PoseLandmark.RIGHT_ELBOW.value

    LW = mp_pose.PoseLandmark.LEFT_WRIST.value
    RW = mp_pose.PoseLandmark.RIGHT_WRIST.value

    LH = mp_pose.PoseLandmark.LEFT_HIP.value
    RH = mp_pose.PoseLandmark.RIGHT_HIP.value

    LK = mp_pose.PoseLandmark.LEFT_KNEE.value
    RK = mp_pose.PoseLandmark.RIGHT_KNEE.value

    LA = mp_pose.PoseLandmark.LEFT_ANKLE.value
    RA = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    points = {

        "LS": get_point(landmarks, LS),
        "RS": get_point(landmarks, RS),

        "LE": get_point(landmarks, LE),
        "RE": get_point(landmarks, RE),

        "LW": get_point(landmarks, LW),
        "RW": get_point(landmarks, RW),

        "LH": get_point(landmarks, LH),
        "RH": get_point(landmarks, RH),

        "LK": get_point(landmarks, LK),
        "RK": get_point(landmarks, RK),

        "LA": get_point(landmarks, LA),
        "RA": get_point(landmarks, RA),
    }

    # --------------------------------------------------------
    # Body scales
    # --------------------------------------------------------

    shoulder_width = np.linalg.norm(
        points["LS"] - points["RS"]
    )

    hip_width = np.linalg.norm(
        points["LH"] - points["RH"]
    )

    body_height = np.linalg.norm(
        (
            points["LS"]
            + points["RS"]
        ) / 2
        -
        (
            points["LA"]
            + points["RA"]
        ) / 2
    )

    # Use shoulder width as fallback
    if body_height <= 1e-6:

        body_height = shoulder_width

    # --------------------------------------------------------
    # Existing joint angles
    # --------------------------------------------------------

    angles = calculate_joint_angles(
        landmarks,
        mp_pose
    )

    # Rename angle keys to match the feature names
    # used by this analysis pipeline.

    angles = {

        "left_elbow_angle":
            angles["left_elbow"],

        "right_elbow_angle":
            angles["right_elbow"],

        "left_knee_angle":
            angles["left_knee"],

        "right_knee_angle":
            angles["right_knee"],

        "left_shoulder_angle":
            angles["left_shoulder"],

        "right_shoulder_angle":
            angles["right_shoulder"],

        "left_hip_angle":
            angles["left_hip"],

        "right_hip_angle":
            angles["right_hip"],
    }

    # --------------------------------------------------------
    # Existing distances
    # --------------------------------------------------------

    wrist_distance_norm = normalized_distance(
        points["LW"],
        points["RW"],
        shoulder_width
    )

    ankle_distance_norm = normalized_distance(
        points["LA"],
        points["RA"],
        hip_width
    )

    wrist_ankle_distance_norm = normalized_distance(
        (
            points["LW"]
            + points["RW"]
        ) / 2,

        (
            points["LA"]
            + points["RA"]
        ) / 2,

        body_height
    )

    # --------------------------------------------------------
    # Torso angle
    # --------------------------------------------------------

    shoulder_center = (
        points["LS"]
        + points["RS"]
    ) / 2

    hip_center = (
        points["LH"]
        + points["RH"]
    ) / 2

    torso_vector = (
        shoulder_center
        - hip_center
    )

    torso_angle = np.degrees(
        np.arctan2(
            abs(torso_vector[0]),
            -torso_vector[1]
        )
    )

    # --------------------------------------------------------
    # Basic Y coordinates
    # --------------------------------------------------------

    left_wrist_y = points["LW"][1]
    right_wrist_y = points["RW"][1]

    left_shoulder_y = points["LS"][1]
    right_shoulder_y = points["RS"][1]

    left_hip_y = points["LH"][1]
    right_hip_y = points["RH"][1]

    left_knee_y = points["LK"][1]
    right_knee_y = points["RK"][1]

    left_ankle_y = points["LA"][1]
    right_ankle_y = points["RA"][1]

    # ========================================================
    # NEW FEATURES
    # ========================================================

    # --------------------------------------------------------
    # 1. Relative vertical positions
    # --------------------------------------------------------

    left_wrist_to_shoulder_y = (
        left_wrist_y
        - left_shoulder_y
    )

    right_wrist_to_shoulder_y = (
        right_wrist_y
        - right_shoulder_y
    )

    left_wrist_to_hip_y = (
        left_wrist_y
        - left_hip_y
    )

    right_wrist_to_hip_y = (
        right_wrist_y
        - right_hip_y
    )

    left_knee_to_hip_y = (
        left_knee_y
        - left_hip_y
    )

    right_knee_to_hip_y = (
        right_knee_y
        - right_hip_y
    )

    left_ankle_to_hip_y = (
        left_ankle_y
        - left_hip_y
    )

    right_ankle_to_hip_y = (
        right_ankle_y
        - right_hip_y
    )

    # --------------------------------------------------------
    # 2. Normalized relative vertical positions
    # --------------------------------------------------------

    left_wrist_to_shoulder_y_norm = (
        left_wrist_to_shoulder_y
        / body_height
    )

    right_wrist_to_shoulder_y_norm = (
        right_wrist_to_shoulder_y
        / body_height
    )

    left_wrist_to_hip_y_norm = (
        left_wrist_to_hip_y
        / body_height
    )

    right_wrist_to_hip_y_norm = (
        right_wrist_to_hip_y
        / body_height
    )

    left_knee_to_hip_y_norm = (
        left_knee_to_hip_y
        / body_height
    )

    right_knee_to_hip_y_norm = (
        right_knee_to_hip_y
        / body_height
    )

    left_ankle_to_hip_y_norm = (
        left_ankle_to_hip_y
        / body_height
    )

    right_ankle_to_hip_y_norm = (
        right_ankle_to_hip_y
        / body_height
    )

    # --------------------------------------------------------
    # 3. Relative horizontal positions
    # --------------------------------------------------------

    left_wrist_to_shoulder_x = (
        points["LW"][0]
        - points["LS"][0]
    )

    right_wrist_to_shoulder_x = (
        points["RW"][0]
        - points["RS"][0]
    )

    left_knee_to_hip_x = (
        points["LK"][0]
        - points["LH"][0]
    )

    right_knee_to_hip_x = (
        points["RK"][0]
        - points["RH"][0]
    )

    left_ankle_to_hip_x = (
        points["LA"][0]
        - points["LH"][0]
    )

    right_ankle_to_hip_x = (
        points["RA"][0]
        - points["RH"][0]
    )

    # --------------------------------------------------------
    # 4. Normalized horizontal positions
    # --------------------------------------------------------

    left_wrist_to_shoulder_x_norm = (
        left_wrist_to_shoulder_x
        / shoulder_width
    )

    right_wrist_to_shoulder_x_norm = (
        right_wrist_to_shoulder_x
        / shoulder_width
    )

    left_knee_to_hip_x_norm = (
        left_knee_to_hip_x
        / hip_width
    )

    right_knee_to_hip_x_norm = (
        right_knee_to_hip_x
        / hip_width
    )

    left_ankle_to_hip_x_norm = (
        left_ankle_to_hip_x
        / hip_width
    )

    right_ankle_to_hip_x_norm = (
        right_ankle_to_hip_x
        / hip_width
    )

    # --------------------------------------------------------
    # 5. Left/right asymmetry
    # --------------------------------------------------------

    knee_angle_asymmetry = safe_difference(
        angles["left_knee_angle"],
        angles["right_knee_angle"]
    )

    hip_angle_asymmetry = safe_difference(
        angles["left_hip_angle"],
        angles["right_hip_angle"]
    )

    elbow_angle_asymmetry = safe_difference(
        angles["left_elbow_angle"],
        angles["right_elbow_angle"]
    )

    shoulder_angle_asymmetry = safe_difference(
        angles["left_shoulder_angle"],
        angles["right_shoulder_angle"]
    )

    wrist_y_asymmetry = abs(
        left_wrist_y
        - right_wrist_y
    )

    knee_y_asymmetry = abs(
        left_knee_y
        - right_knee_y
    )

    ankle_y_asymmetry = abs(
        left_ankle_y
        - right_ankle_y
    )

    # --------------------------------------------------------
    # 6. Body proportions
    # --------------------------------------------------------

    shoulder_width_norm = (
        shoulder_width
        / body_height
    )

    hip_width_norm = (
        hip_width
        / body_height
    )

    # ========================================================
    # FINAL FEATURE DICTIONARY
    # ========================================================

    features = {

        # ----------------------------------------------
        # Joint angles
        # ----------------------------------------------

        **angles,

        # ----------------------------------------------
        # Distances
        # ----------------------------------------------

        "wrist_distance_norm":
            wrist_distance_norm,

        "ankle_distance_norm":
            ankle_distance_norm,

        "wrist_ankle_distance_norm":
            wrist_ankle_distance_norm,

        # ----------------------------------------------
        # Torso
        # ----------------------------------------------

        "torso_angle":
            torso_angle,

        # ----------------------------------------------
        # Raw Y positions
        # ----------------------------------------------

        "left_wrist_y":
            left_wrist_y,

        "right_wrist_y":
            right_wrist_y,

        "left_shoulder_y":
            left_shoulder_y,

        "right_shoulder_y":
            right_shoulder_y,

        "left_hip_y":
            left_hip_y,

        "right_hip_y":
            right_hip_y,

        "left_knee_y":
            left_knee_y,

        "right_knee_y":
            right_knee_y,

        "left_ankle_y":
            left_ankle_y,

        "right_ankle_y":
            right_ankle_y,

        # ----------------------------------------------
        # Relative Y
        # ----------------------------------------------

        "left_wrist_to_shoulder_y":
            left_wrist_to_shoulder_y,

        "right_wrist_to_shoulder_y":
            right_wrist_to_shoulder_y,

        "left_wrist_to_hip_y":
            left_wrist_to_hip_y,

        "right_wrist_to_hip_y":
            right_wrist_to_hip_y,

        "left_knee_to_hip_y":
            left_knee_to_hip_y,

        "right_knee_to_hip_y":
            right_knee_to_hip_y,

        "left_ankle_to_hip_y":
            left_ankle_to_hip_y,

        "right_ankle_to_hip_y":
            right_ankle_to_hip_y,

        # ----------------------------------------------
        # Normalized relative Y
        # ----------------------------------------------

        "left_wrist_to_shoulder_y_norm":
            left_wrist_to_shoulder_y_norm,

        "right_wrist_to_shoulder_y_norm":
            right_wrist_to_shoulder_y_norm,

        "left_wrist_to_hip_y_norm":
            left_wrist_to_hip_y_norm,

        "right_wrist_to_hip_y_norm":
            right_wrist_to_hip_y_norm,

        "left_knee_to_hip_y_norm":
            left_knee_to_hip_y_norm,

        "right_knee_to_hip_y_norm":
            right_knee_to_hip_y_norm,

        "left_ankle_to_hip_y_norm":
            left_ankle_to_hip_y_norm,

        "right_ankle_to_hip_y_norm":
            right_ankle_to_hip_y_norm,

        # ----------------------------------------------
        # Relative X
        # ----------------------------------------------

        "left_wrist_to_shoulder_x":
            left_wrist_to_shoulder_x,

        "right_wrist_to_shoulder_x":
            right_wrist_to_shoulder_x,

        "left_knee_to_hip_x":
            left_knee_to_hip_x,

        "right_knee_to_hip_x":
            right_knee_to_hip_x,

        "left_ankle_to_hip_x":
            left_ankle_to_hip_x,

        "right_ankle_to_hip_x":
            right_ankle_to_hip_x,

        # ----------------------------------------------
        # Normalized relative X
        # ----------------------------------------------

        "left_wrist_to_shoulder_x_norm":
            left_wrist_to_shoulder_x_norm,

        "right_wrist_to_shoulder_x_norm":
            right_wrist_to_shoulder_x_norm,

        "left_knee_to_hip_x_norm":
            left_knee_to_hip_x_norm,

        "right_knee_to_hip_x_norm":
            right_knee_to_hip_x_norm,

        "left_ankle_to_hip_x_norm":
            left_ankle_to_hip_x_norm,

        "right_ankle_to_hip_x_norm":
            right_ankle_to_hip_x_norm,

        # ----------------------------------------------
        # Asymmetry
        # ----------------------------------------------

        "knee_angle_asymmetry":
            knee_angle_asymmetry,

        "hip_angle_asymmetry":
            hip_angle_asymmetry,

        "elbow_angle_asymmetry":
            elbow_angle_asymmetry,

        "shoulder_angle_asymmetry":
            shoulder_angle_asymmetry,

        "wrist_y_asymmetry":
            wrist_y_asymmetry,

        "knee_y_asymmetry":
            knee_y_asymmetry,

        "ankle_y_asymmetry":
            ankle_y_asymmetry,

        # ----------------------------------------------
        # Body proportions
        # ----------------------------------------------

        "shoulder_width_norm":
            shoulder_width_norm,

        "hip_width_norm":
            hip_width_norm,
    }

    # ========================================================
    # VISIBILITY VALUES
    # ========================================================

    visibility_landmarks = {

        "left_shoulder": LS,
        "right_shoulder": RS,

        "left_elbow": LE,
        "right_elbow": RE,

        "left_wrist": LW,
        "right_wrist": RW,

        "left_hip": LH,
        "right_hip": RH,

        "left_knee": LK,
        "right_knee": RK,

        "left_ankle": LA,
        "right_ankle": RA,
    }

    for name, landmark_id in (
        visibility_landmarks.items()
    ):

        features[
            f"{name}_visibility"
        ] = landmarks[
            landmark_id
        ].visibility

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SURYA NAMASKARA - POSE FEATURE EXTRACTION")
    print("=" * 70)

    print()
    print("Dataset:", DATASET_DIR)
    print("Output :", OUTPUT_CSV)
    print(
        "Visibility threshold:",
        VISIBILITY_THRESHOLD
    )
    print()

    rows = []

    total_images = 0
    successful = 0
    failed = 0

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose_detector:

        for pose_name in POSES:

            pose_dir = os.path.join(
                DATASET_DIR,
                pose_name
            )

            if not os.path.exists(
                pose_dir
            ):

                print(
                    f"WARNING: {pose_dir} not found"
                )

                continue

            image_files = [
                f
                for f in os.listdir(
                    pose_dir
                )
                if f.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp"
                    )
                )
            ]

            print(
                f"Processing: {pose_name}"
            )

            pose_success = 0

            for filename in image_files:

                total_images += 1

                image_path = os.path.join(
                    pose_dir,
                    filename
                )

                image = cv2.imread(
                    image_path
                )

                if image is None:

                    failed += 1
                    continue

                features = extract_features(
                    image,
                    pose_detector
                )

                if features is None:

                    failed += 1
                    continue

                features["pose"] = pose_name
                features["image"] = filename

                rows.append(
                    features
                )

                successful += 1
                pose_success += 1

            print(
                f"  Images: {len(image_files)} | "
                f"Features extracted: "
                f"{pose_success}"
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    # Put pose/image first

    first_columns = [
        "pose",
        "image"
    ]

    remaining_columns = [
        c
        for c in df.columns
        if c not in first_columns
    ]

    df = df[
        first_columns
        + remaining_columns
    ]

    # ========================================================
    # SAVE
    # ========================================================

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Total images found      : "
        f"{total_images}"
    )

    print(
        f"Successful extractions : "
        f"{successful}"
    )

    print(
        f"Failed                  : "
        f"{failed}"
    )

    print(
        f"Feature rows saved      : "
        f"{len(df)}"
    )

    print()
    print("Saved to:")
    print(OUTPUT_CSV)

    print()
    print("Class counts:")

    print(
        df["pose"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Number of features:")

    print(
        len(df.columns) - 2
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()