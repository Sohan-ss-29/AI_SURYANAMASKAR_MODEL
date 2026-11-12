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

CANDIDATES_CSV = os.path.join(
    PROJECT_ROOT,
    "reports",
    "pose_analysis",
    "pose_rule_candidates.csv"
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "reports",
    "pose_analysis",
    "pose_independent_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Features with absolute correlation above this value
# are considered highly redundant.
CORRELATION_THRESHOLD = 0.85

# Final number of independent features to keep per pose.
TOP_FEATURES_PER_POSE = 5


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURES_CSV)

candidates_df = pd.read_csv(CANDIDATES_CSV)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {
    "pose",
    "feature",
    "effect_size",
    "global_separation_score"
}

missing_columns = required_columns - set(
    candidates_df.columns
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CALCULATE FEATURE CORRELATION MATRIX
# ============================================================

candidate_features = candidates_df[
    "feature"
].unique()

candidate_features = [
    feature
    for feature in candidate_features
    if feature in df.columns
]

correlation_matrix = df[
    candidate_features
].corr(method="spearman")


# ============================================================
# FEATURE SELECTION
# ============================================================

results = []

poses = sorted(
    candidates_df["pose"].unique()
)

for pose in poses:

    print("\n" + "=" * 70)
    print(f"POSE: {pose}")
    print("=" * 70)

    pose_candidates = candidates_df[
        candidates_df["pose"] == pose
    ].copy()

    # Sort strongest features first.
    pose_candidates = pose_candidates.sort_values(
        [
            "effect_size",
            "global_separation_score"
        ],
        ascending=False
    )

    selected_features = []

    for _, row in pose_candidates.iterrows():

        feature = row["feature"]

        # First feature is automatically selected.
        if len(selected_features) == 0:
            selected_features.append(feature)

            results.append(row.to_dict())

            print(
                f"SELECTED: {feature}"
            )

            continue

        # Check correlation against already selected features.
        is_redundant = False

        for selected_feature in selected_features:

            correlation = correlation_matrix.loc[
                feature,
                selected_feature
            ]

            if abs(correlation) >= CORRELATION_THRESHOLD:

                is_redundant = True

                print(
                    f"SKIPPED: {feature}"
                    f"  | correlation with "
                    f"{selected_feature}: "
                    f"{correlation:.3f}"
                )

                break

        # Keep feature if it provides sufficiently
        # independent information.
        if not is_redundant:

            selected_features.append(feature)

            results.append(row.to_dict())

            print(
                f"SELECTED: {feature}"
            )

        # Stop after desired number of features.
        if len(selected_features) >= TOP_FEATURES_PER_POSE:
            break

    print(
        f"\nFinal features for {pose}:"
    )

    for i, feature in enumerate(
        selected_features,
        start=1
    ):
        print(
            f"  {i}. {feature}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

final_df = pd.DataFrame(results)

final_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INDEPENDENT RULE FEATURE SELECTION COMPLETE")
print("=" * 70)

print(
    f"\nSaved to:\n{OUTPUT_CSV}"
)

print(
    f"\nTotal selected features: "
    f"{len(final_df)}"
)

print("\nFeatures per pose:")

print(
    final_df.groupby("pose")[
        "feature"
    ].count().to_string()
)