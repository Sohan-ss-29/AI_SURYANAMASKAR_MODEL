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
    "pose_rule_thresholds.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURES_CSV)

selected_df = pd.read_csv(
    SELECTED_FEATURES_CSV
)


# ============================================================
# ANALYZE EACH POSE / FEATURE
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

    pose_features = selected_df[
        selected_df["pose"] == pose
    ]["feature"].tolist()

    for feature in pose_features:

        values = pose_df[
            feature
        ].dropna()

        if len(values) < 10:
            continue

        # ----------------------------------------------------
        # Calculate distribution statistics
        # ----------------------------------------------------

        minimum = values.min()
        q10 = values.quantile(0.10)
        q25 = values.quantile(0.25)
        median = values.median()
        q75 = values.quantile(0.75)
        q90 = values.quantile(0.90)
        maximum = values.max()
        mean = values.mean()
        std = values.std()

        results.append({
            "pose": pose,
            "feature": feature,
            "valid_samples": len(values),
            "minimum": minimum,
            "q10": q10,
            "q25": q25,
            "median": median,
            "q75": q75,
            "q90": q90,
            "maximum": maximum,
            "mean": mean,
            "std": std
        })

        print(
            f"\nFeature: {feature}"
        )

        print(
            f"  Samples : {len(values)}"
        )

        print(
            f"  Min     : {minimum:.4f}"
        )

        print(
            f"  Q10     : {q10:.4f}"
        )

        print(
            f"  Q25     : {q25:.4f}"
        )

        print(
            f"  Median  : {median:.4f}"
        )

        print(
            f"  Q75     : {q75:.4f}"
        )

        print(
            f"  Q90     : {q90:.4f}"
        )

        print(
            f"  Max     : {maximum:.4f}"
        )

        print(
            f"  Mean    : {mean:.4f}"
        )

        print(
            f"  Std     : {std:.4f}"
        )


# ============================================================
# SAVE
# ============================================================

thresholds_df = pd.DataFrame(
    results
)

thresholds_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("POSE RULE THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nSaved to:\n{OUTPUT_CSV}"
)

print(
    f"\nTotal pose-feature rows: "
    f"{len(thresholds_df)}"
)