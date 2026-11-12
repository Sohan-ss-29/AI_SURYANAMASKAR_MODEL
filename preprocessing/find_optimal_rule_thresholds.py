import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FEATURES_CSV = os.path.join(
    PROJECT_ROOT,
    "reports",
    "train_pose_features.csv"
)

SELECTED_FEATURES_CSV = os.path.join(
    PROJECT_ROOT,
    "reports",
    "pose_analysis",
    "pose_independent_features.csv"
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "reports",
    "pose_analysis",
    "optimal_rule_thresholds.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_SAMPLES = 20

# Number of candidate threshold points
N_THRESHOLDS = 100


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURES_CSV)

selected_df = pd.read_csv(
    SELECTED_FEATURES_CSV
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    tp = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    tn = np.sum(
        (y_true == 0) & (y_pred == 0)
    )

    fp = np.sum(
        (y_true == 0) & (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1) & (y_pred == 0)
    )

    accuracy = (
        (tp + tn) /
        len(y_true)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    if (
        precision + recall
    ) > 0:

        f1 = (
            2 *
            precision *
            recall /
            (precision + recall)
        )

    else:
        f1 = 0

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

results = []

poses = sorted(
    selected_df["pose"].unique()
)

for pose in poses:

    print("\n" + "=" * 70)
    print(f"POSE: {pose}")
    print("=" * 70)

    pose_df = df[
        df["pose"] == pose
    ]

    other_df = df[
        df["pose"] != pose
    ]

    features = selected_df[
        selected_df["pose"] == pose
    ]["feature"].tolist()

    for feature in features:

        pose_values = pose_df[
            feature
        ].dropna()

        other_values = other_df[
            feature
        ].dropna()

        if (
            len(pose_values) < MIN_SAMPLES
            or
            len(other_values) < MIN_SAMPLES
        ):
            continue

        # ----------------------------------------------------
        # Combine values
        # ----------------------------------------------------

        all_values = np.concatenate(
            [
                pose_values.values,
                other_values.values
            ]
        )

        min_value = np.percentile(
            all_values,
            1
        )

        max_value = np.percentile(
            all_values,
            99
        )

        thresholds = np.linspace(
            min_value,
            max_value,
            N_THRESHOLDS
        )

        best_result = None

        # ====================================================
        # TEST BOTH DIRECTIONS
        # ====================================================

        for direction in [
            "less_than",
            "greater_than"
        ]:

            for threshold in thresholds:

                if direction == "less_than":

                    pose_predictions = (
                        pose_values < threshold
                    )

                    other_predictions = (
                        other_values < threshold
                    )

                else:

                    pose_predictions = (
                        pose_values > threshold
                    )

                    other_predictions = (
                        other_values > threshold
                    )

                y_true = np.concatenate(
                    [
                        np.ones(
                            len(pose_values)
                        ),
                        np.zeros(
                            len(other_values)
                        )
                    ]
                )

                y_pred = np.concatenate(
                    [
                        pose_predictions,
                        other_predictions
                    ]
                )

                accuracy, precision, recall, f1 = (
                    calculate_metrics(
                        y_true,
                        y_pred
                    )
                )

                if (
                    best_result is None
                    or f1 > best_result["f1"]
                ):

                    best_result = {
                        "pose": pose,
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "pose_median": pose_values.median(),
                        "other_median": other_values.median(),
                        "pose_q10": pose_values.quantile(
                            0.10
                        ),
                        "pose_q90": pose_values.quantile(
                            0.90
                        ),
                        "pose_samples": len(
                            pose_values
                        ),
                        "other_samples": len(
                            other_values
                        )
                    }

        results.append(
            best_result
        )

        print(
            f"\nFeature: {feature}"
        )

        print(
            f"  Direction : "
            f"{best_result['direction']}"
        )

        print(
            f"  Threshold : "
            f"{best_result['threshold']:.4f}"
        )

        print(
            f"  Accuracy  : "
            f"{best_result['accuracy']:.4f}"
        )

        print(
            f"  Precision : "
            f"{best_result['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{best_result['recall']:.4f}"
        )

        print(
            f"  F1-score  : "
            f"{best_result['f1']:.4f}"
        )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    [
        "pose",
        "f1"
    ],
    ascending=[
        True,
        False
    ]
)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("OPTIMAL RULE THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nSaved to:\n{OUTPUT_CSV}"
)

print(
    f"\nTotal threshold candidates: "
    f"{len(results_df)}"
)