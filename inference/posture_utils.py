import math


# ============================================================
# ANGLE CALCULATION
# ============================================================

def calculate_angle(a, b, c):
    """
    Calculate angle ABC using three 2D points.

    a = first point
    b = vertex / joint
    c = third point

    Returns:
        Angle in degrees.
    """

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# ============================================================
# VISIBILITY-AWARE ANGLE
# ============================================================

def calculate_angle_if_visible(
    landmarks,
    landmark_a,
    landmark_b,
    landmark_c,
    min_visibility=0.5
):
    """
    Calculate an angle only when all three landmarks
    have sufficient visibility.

    Returns:
        Angle in degrees, or None if visibility is insufficient.
    """

    visibility_a = landmarks[landmark_a].visibility
    visibility_b = landmarks[landmark_b].visibility
    visibility_c = landmarks[landmark_c].visibility

    if (
        visibility_a < min_visibility
        or visibility_b < min_visibility
        or visibility_c < min_visibility
    ):
        return None

    a = (
        landmarks[landmark_a].x,
        landmarks[landmark_a].y
    )

    b = (
        landmarks[landmark_b].x,
        landmarks[landmark_b].y
    )

    c = (
        landmarks[landmark_c].x,
        landmarks[landmark_c].y
    )

    return calculate_angle(a, b, c)


# ============================================================
# LANDMARK POINT
# ============================================================

def get_landmark_point(landmarks, landmark_id):
    """
    Return normalized x and y coordinates.
    """

    landmark = landmarks[landmark_id]

    return (
        landmark.x,
        landmark.y
    )


# ============================================================
# LANDMARK VISIBILITY
# ============================================================

def get_landmark_visibility(landmarks, landmark_id):
    """
    Return MediaPipe landmark visibility.
    """

    return landmarks[landmark_id].visibility


# ============================================================
# BILATERAL JOINT ANGLES
# ============================================================

def calculate_joint_angles(landmarks, mp_pose):
    """
    Calculate important left and right side joint angles.

    Returns a dictionary containing:
        left_elbow
        right_elbow
        left_knee
        right_knee
        left_shoulder
        right_shoulder
        left_hip
        right_hip
    """

    # --------------------------------------------------------
    # Landmark IDs
    # --------------------------------------------------------

    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

    LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
    RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value

    LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
    RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value

    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value

    LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
    RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value

    LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
    RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    # --------------------------------------------------------
    # Elbow angles
    # --------------------------------------------------------

    left_elbow = calculate_angle_if_visible(
        landmarks,
        LEFT_SHOULDER,
        LEFT_ELBOW,
        LEFT_WRIST
    )

    right_elbow = calculate_angle_if_visible(
        landmarks,
        RIGHT_SHOULDER,
        RIGHT_ELBOW,
        RIGHT_WRIST
    )

    # --------------------------------------------------------
    # Knee angles
    # --------------------------------------------------------

    left_knee = calculate_angle_if_visible(
        landmarks,
        LEFT_HIP,
        LEFT_KNEE,
        LEFT_ANKLE
    )

    right_knee = calculate_angle_if_visible(
        landmarks,
        RIGHT_HIP,
        RIGHT_KNEE,
        RIGHT_ANKLE
    )

    # --------------------------------------------------------
    # Shoulder angles
    # --------------------------------------------------------

    left_shoulder = calculate_angle_if_visible(
        landmarks,
        LEFT_ELBOW,
        LEFT_SHOULDER,
        LEFT_HIP
    )

    right_shoulder = calculate_angle_if_visible(
        landmarks,
        RIGHT_ELBOW,
        RIGHT_SHOULDER,
        RIGHT_HIP
    )

    # --------------------------------------------------------
    # Hip angles
    # --------------------------------------------------------

    left_hip = calculate_angle_if_visible(
        landmarks,
        LEFT_SHOULDER,
        LEFT_HIP,
        LEFT_KNEE
    )

    right_hip = calculate_angle_if_visible(
        landmarks,
        RIGHT_SHOULDER,
        RIGHT_HIP,
        RIGHT_KNEE
    )

    return {
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,

        "left_knee": left_knee,
        "right_knee": right_knee,

        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,

        "left_hip": left_hip,
        "right_hip": right_hip,
    }


# ============================================================
# FORMAT ANGLE FOR DISPLAY
# ============================================================

def format_angle(angle):
    """
    Convert angle to a display-friendly string.
    """

    if angle is None:
        return "NA"

    return f"{angle:.1f}"