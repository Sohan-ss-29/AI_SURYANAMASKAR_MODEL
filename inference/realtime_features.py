import math


def normalize(value, scale):
    if scale == 0:
        return 0.0

    return value / scale


def get_point(landmarks, idx):
    return (
        landmarks[idx].x,
        landmarks[idx].y
    )


def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def calculate_real_time_features(landmarks, mp_pose):

    P = mp_pose.PoseLandmark

    left_shoulder = get_point(
        landmarks, P.LEFT_SHOULDER.value
    )

    right_shoulder = get_point(
        landmarks, P.RIGHT_SHOULDER.value
    )

    left_hip = get_point(
        landmarks, P.LEFT_HIP.value
    )

    right_hip = get_point(
        landmarks, P.RIGHT_HIP.value
    )

    left_wrist = get_point(
        landmarks, P.LEFT_WRIST.value
    )

    right_wrist = get_point(
        landmarks, P.RIGHT_WRIST.value
    )

    left_knee = get_point(
        landmarks, P.LEFT_KNEE.value
    )

    right_knee = get_point(
        landmarks, P.RIGHT_KNEE.value
    )

    left_ankle = get_point(
        landmarks, P.LEFT_ANKLE.value
    )

    right_ankle = get_point(
        landmarks, P.RIGHT_ANKLE.value
    )

    # --------------------------------------------------------
    # Body scale
    # --------------------------------------------------------

    shoulder_width = distance(
        left_shoulder,
        right_shoulder
    )

    hip_width = distance(
        left_hip,
        right_hip
    )

    body_scale = max(
        shoulder_width,
        hip_width,
        1e-6
    )

    # --------------------------------------------------------
    # Angles
    # --------------------------------------------------------

    from posture_utils import calculate_angle_if_visible

    left_elbow = calculate_angle_if_visible(
        landmarks,
        P.LEFT_SHOULDER.value,
        P.LEFT_ELBOW.value,
        P.LEFT_WRIST.value
    )

    right_elbow = calculate_angle_if_visible(
        landmarks,
        P.RIGHT_SHOULDER.value,
        P.RIGHT_ELBOW.value,
        P.RIGHT_WRIST.value
    )

    left_knee_angle = calculate_angle_if_visible(
        landmarks,
        P.LEFT_HIP.value,
        P.LEFT_KNEE.value,
        P.LEFT_ANKLE.value
    )

    right_knee_angle = calculate_angle_if_visible(
        landmarks,
        P.RIGHT_HIP.value,
        P.RIGHT_KNEE.value,
        P.RIGHT_ANKLE.value
    )

    left_hip_angle = calculate_angle_if_visible(
        landmarks,
        P.LEFT_SHOULDER.value,
        P.LEFT_HIP.value,
        P.LEFT_KNEE.value
    )

    right_hip_angle = calculate_angle_if_visible(
        landmarks,
        P.RIGHT_SHOULDER.value,
        P.RIGHT_HIP.value,
        P.RIGHT_KNEE.value
    )

    # --------------------------------------------------------
    # Torso angle
    # --------------------------------------------------------

    shoulder_mid = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    )

    hip_mid = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    )

    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]

    torso_angle = abs(
        math.degrees(math.atan2(dx, -dy))
    )

    # --------------------------------------------------------
    # Features used by current rules
    # --------------------------------------------------------

    features = {

        "torso_angle":
            torso_angle,

        "left_elbow_angle":
            left_elbow,

        "right_elbow_angle":
            right_elbow,

        "left_knee_angle":
            left_knee_angle,

        "right_knee_angle":
            right_knee_angle,

        "left_hip_angle":
            left_hip_angle,

        "right_hip_angle":
            right_hip_angle,

        "left_wrist_y":
            left_wrist[1],

        "right_wrist_y":
            right_wrist[1],

        "left_shoulder_y":
            left_shoulder[1],

        "right_shoulder_y":
            right_shoulder[1],

        "right_hip_y":
            right_hip[1],

        "left_wrist_to_shoulder_y":
            left_wrist[1] - left_shoulder[1],

        "right_wrist_to_shoulder_y":
            right_wrist[1] - right_shoulder[1],

        "left_wrist_to_hip_y":
            left_wrist[1] - left_hip[1],

        "right_wrist_to_hip_y":
            right_wrist[1] - right_hip[1],

        "left_wrist_to_hip_y_norm":
            normalize(
                left_wrist[1] - left_hip[1],
                body_scale
            ),

        "right_wrist_to_hip_y_norm":
            normalize(
                right_wrist[1] - right_hip[1],
                body_scale
            ),

        "left_ankle_to_hip_x":
            left_ankle[0] - left_hip[0],

        "right_ankle_to_hip_x":
            right_ankle[0] - right_hip[0],

        "left_ankle_to_hip_y":
            left_ankle[1] - left_hip[1],

        "right_ankle_to_hip_y":
            right_ankle[1] - right_hip[1],

        "left_ankle_to_hip_y_norm":
            normalize(
                left_ankle[1] - left_hip[1],
                body_scale
            ),

        "right_ankle_to_hip_y_norm":
            normalize(
                right_ankle[1] - right_hip[1],
                body_scale
            ),

        "right_knee_to_hip_x":
            right_knee[0] - right_hip[0],

        "knee_y_asymmetry":
            abs(left_knee[1] - right_knee[1]),
    }

    return features