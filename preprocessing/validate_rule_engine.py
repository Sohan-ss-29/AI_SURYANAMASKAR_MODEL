import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

TRAIN_FEATURES = "reports/train_pose_features.csv"
VAL_FEATURES = "reports/val_pose_features.csv"

RULE_FEATURES = (
    "reports/pose_analysis/"
    "pose_independent_features.csv"
)

RULE_THRESHOLDS = (
    "reports/pose_analysis/"
    "optimal_rule_thresholds.csv"
)

OUTPUT_DIR = "reports/pose_analysis"

OUTPUT_RULE_RESULTS = (
    f"{OUTPUT_DIR}/validation_rule_results.csv"
)

OUTPUT_POSE_RESULTS = (
    f"{OUTPUT_DIR}/validation_pose_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Minimum number of rules that should be available
# for a sample to receive a meaningful score.
MIN_AVAILABLE_RULES = 2

# Test acceptance thresholds from 0.40 to 0.90
ACCEPTANCE_THRESHOLDS = np.arange(
    0.40,
    0.91,
    0.01
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FEATURES)
val_df = pd.read_csv(VAL_FEATURES)

independent_df = pd.read_csv(RULE_FEATURES)
threshold_df = pd.read_csv(RULE_THRESHOLDS)

print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Selected rules     : {len(independent_df)}")
print(f"Threshold rules    : {len(threshold_df)}")


# ============================================================
# BUILD RULE TABLE
# ============================================================

# Merge the selected independent features
# with their optimized thresholds.

rules = independent_df.merge(
    threshold_df[
        [
            "pose",
            "feature",
            "direction",
            "threshold",
            "f1"
        ]
    ],
    on=["pose", "feature"],
    how="inner"
)

print()
print(f"Usable rules after merge : {len(rules)}")


# ============================================================
# RULE STRENGTH
# ============================================================

def get_rule_weight(f1):
    """
    Convert individual training F1 into a rule weight.

    Stronger individual rules get more influence.
    """

    if f1 >= 0.80:
        return 1.00

    elif f1 >= 0.70:
        return 0.80

    elif f1 >= 0.60:
        return 0.60

    elif f1 >= 0.50:
        return 0.40

    else:
        return 0.25


rules["weight"] = rules["f1"].apply(get_rule_weight)


# ============================================================
# RULE EVALUATION
# ============================================================

def evaluate_rule(row, feature_value):

    if pd.isna(feature_value):
        return np.nan

    threshold = row["threshold"]
    direction = row["direction"]

    if direction == "less_than":
        return float(feature_value < threshold)

    elif direction == "greater_than":
        return float(feature_value > threshold)

    return np.nan


# ============================================================
# CREATE RULE SCORES
# ============================================================

print()
print("=" * 70)
print("CALCULATING VALIDATION RULE SCORES")
print("=" * 70)

all_rule_results = []
pose_results = []

poses = sorted(
    rules["pose"].unique()
)

for pose in poses:

    print()
    print("=" * 70)
    print(f"POSE: {pose}")
    print("=" * 70)

    pose_rules = rules[
        rules["pose"] == pose
    ].copy()

    feature_names = pose_rules[
        "feature"
    ].tolist()

    weights = pose_rules[
        "weight"
    ].values

    thresholds = pose_rules[
        "threshold"
    ].values

    directions = pose_rules[
        "direction"
    ].tolist()

    f1_values = pose_rules[
        "f1"
    ].values

    rule_scores = []
    rule_counts = []

    # --------------------------------------------------------
    # Evaluate every validation sample
    # --------------------------------------------------------

    for idx, sample in val_df.iterrows():

        weighted_sum = 0.0
        weight_sum = 0.0
        available_rules = 0

        individual_results = []

        for feature, weight, threshold, direction in zip(
            feature_names,
            weights,
            thresholds,
            directions
        ):

            value = sample.get(feature, np.nan)

            if pd.isna(value):
                individual_results.append(np.nan)
                continue

            passed = evaluate_rule(
                {
                    "threshold": threshold,
                    "direction": direction
                },
                value
            )

            individual_results.append(passed)

            weighted_sum += passed * weight
            weight_sum += weight

            available_rules += 1

        if (
            available_rules >= MIN_AVAILABLE_RULES
            and weight_sum > 0
        ):
            score = weighted_sum / weight_sum
        else:
            score = np.nan

        rule_scores.append(score)
        rule_counts.append(available_rules)

        all_rule_results.append({
            "sample_index": idx,
            "target_pose": sample["pose"],
            "evaluated_pose": pose,
            "rule_score": score,
            "available_rules": available_rules
        })

    val_scores = np.array(rule_scores)
    val_counts = np.array(rule_counts)

    # --------------------------------------------------------
    # Binary ground truth
    # --------------------------------------------------------

    y_true = (
        val_df["pose"].values == pose
    ).astype(int)

    valid_mask = (
        ~np.isnan(val_scores)
    )

    y_true_valid = y_true[
        valid_mask
    ]

    scores_valid = val_scores[
        valid_mask
    ]

    print(
        f"Validation samples evaluated : "
        f"{len(scores_valid)}"
    )

    print(
        f"Samples with insufficient rules: "
        f"{np.sum(~valid_mask)}"
    )

    # --------------------------------------------------------
    # Find best acceptance threshold
    # --------------------------------------------------------

    best_result = None

    for acceptance_threshold in ACCEPTANCE_THRESHOLDS:

        y_pred = (
            scores_valid >= acceptance_threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true_valid,
            y_pred
        )

        precision = precision_score(
            y_true_valid,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true_valid,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true_valid,
            y_pred,
            zero_division=0
        )

        result = {
            "pose": pose,
            "acceptance_threshold":
                acceptance_threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "evaluated_samples":
                len(scores_valid)
        }

        if (
            best_result is None
            or f1 > best_result["f1"]
        ):
            best_result = result

    pose_results.append(
        best_result
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("BEST VALIDATION RESULT")

    print(
        f"Acceptance threshold : "
        f"{best_result['acceptance_threshold']:.2f}"
    )

    print(
        f"Accuracy             : "
        f"{best_result['accuracy']:.4f}"
    )

    print(
        f"Precision            : "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall               : "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"F1-score             : "
        f"{best_result['f1']:.4f}"
    )

    print()
    print("RULES")

    for _, rule in pose_rules.iterrows():

        print(
            f"  {rule['feature']}"
        )

        print(
            f"      {rule['direction']} "
            f"{rule['threshold']:.4f}"
        )

        print(
            f"      Training F1 = "
            f"{rule['f1']:.4f}"
        )

        print(
            f"      Weight      = "
            f"{rule['weight']:.2f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

rule_results_df = pd.DataFrame(
    all_rule_results
)

pose_results_df = pd.DataFrame(
    pose_results
)

rule_results_df.to_csv(
    OUTPUT_RULE_RESULTS,
    index=False
)

pose_results_df.to_csv(
    OUTPUT_POSE_RESULTS,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print()

for _, row in pose_results_df.iterrows():

    print(
        f"{row['pose']:<28} "
        f"F1 = {row['f1']:.4f}   "
        f"Threshold = "
        f"{row['acceptance_threshold']:.2f}"
    )

print()
print("Saved:")
print(
    os.path.abspath(
        OUTPUT_RULE_RESULTS
    )
)

print(
    os.path.abspath(
        OUTPUT_POSE_RESULTS
    )
)