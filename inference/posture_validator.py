from typing import Dict, Tuple


# ============================================================
# RULE CONFIGURATION
# ============================================================

RULES = {

    "1_Pranamasana": [
        ("torso_angle", "less_than", 3.5581, 1.00),
        ("right_elbow_angle", "less_than", 42.5392, 1.00),
        ("right_wrist_y", "less_than", 0.5521, 0.60),
        ("right_wrist_to_hip_y", "less_than", -0.0891, 0.60),
        ("right_shoulder_y", "less_than", 0.4555, 0.60),
    ],

    "2_Hasta_Uttanasana": [
        ("left_wrist_y", "less_than", 0.4206, 0.40),
        ("right_knee_to_hip_x", "greater_than", 0.0327, 0.40),
        ("left_wrist_to_shoulder_y", "less_than", 0.0095, 0.60),
        ("left_wrist_to_hip_y", "less_than", -0.1463, 0.40),
        ("left_ankle_to_hip_x", "greater_than", -0.0003, 0.40),
    ],

    "3_Padahastasana": [
        ("torso_angle", "greater_than", 149.4726, 0.60),
        ("left_wrist_to_hip_y_norm", "greater_than", 1.4507, 0.80),
        ("left_ankle_to_hip_y_norm", "greater_than", 1.1074, 1.00),
        ("right_hip_y", "less_than", 0.4292, 0.60),
    ],

    "4_AshwaSanchalanasana": [
        ("knee_y_asymmetry", "greater_than", 0.1618, 1.00),
        ("right_wrist_y", "less_than", 0.2573, 0.40),
        ("left_wrist_to_hip_y", "less_than", -0.4217, 0.40),
        ("right_hip_angle", "less_than", 98.0287, 0.25),
    ],

    "5_Parvatasana": [
        ("torso_angle", "greater_than", 128.6277, 0.40),
        ("left_wrist_to_hip_y", "greater_than", 0.2864, 0.40),
        ("right_hip_y", "less_than", 0.3700, 0.25),
        ("left_wrist_y", "greater_than", 0.7232, 0.25),
    ],

    "6_Ashtanga_Namaskara": [
        ("right_ankle_to_hip_x", "less_than", -0.2877, 1.00),
        ("left_ankle_to_hip_x", "less_than", -0.2434, 1.00),
        ("right_wrist_y", "greater_than", 0.6317, 0.40),
        ("right_shoulder_y", "greater_than", 0.5483, 0.60),
        ("right_ankle_to_hip_y", "less_than", 0.0531, 0.60),
    ],

    "7_Bhujangasana": [
        ("right_wrist_to_shoulder_y", "greater_than", 0.2856, 0.25),
        ("left_wrist_to_shoulder_y", "greater_than", 0.2830, 0.25),
        ("left_wrist_y", "greater_than", 0.7312, 0.25),
        ("right_ankle_to_hip_y", "less_than", 0.1184, 0.25),
    ],
}


# Acceptance thresholds obtained from validation
ACCEPTANCE_THRESHOLDS = {
    "1_Pranamasana": 0.65,
    "2_Hasta_Uttanasana": 0.55,
    "3_Padahastasana": 0.40,
    "4_AshwaSanchalanasana": 0.40,
    "5_Parvatasana": 0.70,
    "6_Ashtanga_Namaskara": 0.84,
    "7_Bhujangasana": 0.75,
}


# Human-readable correction messages
CORRECTIONS = {
    "1_Pranamasana": "Bring your palms together at your chest and stand straight.",
    "2_Hasta_Uttanasana": "Raise your arms upward and gently arch your upper body.",
    "3_Padahastasana": "Bend forward and bring your hands toward the floor.",
    "4_AshwaSanchalanasana": "Step one leg back and keep your front knee properly bent.",
    "5_Parvatasana": "Lift your hips upward and straighten your legs.",
    "6_Ashtanga_Namaskara": "Lower your knees, chest and chin toward the floor.",
    "7_Bhujangasana": "Keep your lower body down and lift your chest upward.",
}


def check_rule(value, operator, threshold):
    if value is None:
        return None

    if operator == "less_than":
        return value < threshold

    if operator == "greater_than":
        return value > threshold

    return False


def validate_pose(features: Dict[str, float], pose_name: str) -> Dict:
    """
    Validate whether the current body geometry matches
    the predicted pose.

    Returns:
        {
            "valid": True/False,
            "score": float,
            "passed_rules": int,
            "available_rules": int,
            "message": str
        }
    """

    if pose_name not in RULES:
        return {
            "valid": False,
            "score": 0.0,
            "passed_rules": 0,
            "available_rules": 0,
            "message": "Unknown pose"
        }

    rules = RULES[pose_name]

    total_weight = 0.0
    passed_weight = 0.0

    passed_rules = 0
    available_rules = 0

    for feature_name, operator, threshold, weight in rules:

        value = features.get(feature_name)

        result = check_rule(
            value,
            operator,
            threshold
        )

        # Missing landmark / unavailable feature
        if result is None:
            continue

        available_rules += 1
        total_weight += weight

        if result:
            passed_rules += 1
            passed_weight += weight

    if total_weight == 0:
        return {
            "valid": False,
            "score": 0.0,
            "passed_rules": 0,
            "available_rules": 0,
            "message": "Body landmarks not clear"
        }

    score = passed_weight / total_weight

    required_score = ACCEPTANCE_THRESHOLDS[pose_name]

    valid = score >= required_score

    if valid:
        message = "Correct posture"
    else:
        message = CORRECTIONS.get(
            pose_name,
            "Adjust your posture."
        )

    return {
        "valid": valid,
        "score": score,
        "passed_rules": passed_rules,
        "available_rules": available_rules,
        "message": message
    }