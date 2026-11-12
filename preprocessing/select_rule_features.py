import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "reports",
    "train_pose_features.csv"
)

ANALYSIS_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "pose_analysis"
)

OUTPUT_CSV = os.path.join(
    ANALYSIS_DIR,
    "pose_rule_candidates.csv"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_GLOBAL_MISSING_PERCENT = 30
MAX_POSE_MISSING_PERCENT = 20

EXCLUDED_RULE_FEATURE_PATTERNS = [
    "_visibility",
    "width_norm"
]

TOP_FEATURES_PER_POSE = 8


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

EXCLUDE_COLUMNS = {
    "image",
    "pose"
}

FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in EXCLUDE_COLUMNS
]


# ============================================================
# LOAD GLOBAL SEPARATION SCORES
# ============================================================

separation_file = os.path.join(
    ANALYSIS_DIR,
    "feature_separation_scores.csv"
)

separation_df = pd.read_csv(separation_file)

# Keep only features with acceptable global missingness.
separation_df = separation_df[
    separation_df["missing_percent"] <= MAX_GLOBAL_MISSING_PERCENT
].copy()

separation_df = separation_df.sort_values(
    "separation_score",
    ascending=False
)


# ============================================================
# CALCULATE POSE-SPECIFIC DISCRIMINATION
# ============================================================

results = []

poses = sorted(df["pose"].unique())

for pose in poses:

    print("\n" + "=" * 70)
    print(f"POSE: {pose}")
    print("=" * 70)

    pose_df = df[df["pose"] == pose]
    other_df = df[df["pose"] != pose]

    candidates = []

    for feature in separation_df["feature"]:

        # Ignore visibility-related features and
        # body-width normalization features.
        if any(
            pattern in feature
            for pattern in EXCLUDED_RULE_FEATURE_PATTERNS
        ):
            continue

        # Calculate missing percentage specifically
        # for the current pose.
        pose_missing_percent = (
            pose_df[feature].isna().mean() * 100
        )

        # Ignore features that are too frequently
        # unavailable for this particular pose.
        if pose_missing_percent > MAX_POSE_MISSING_PERCENT:
            continue

        pose_values = pose_df[feature].dropna()
        other_values = other_df[feature].dropna()

        # Make sure enough valid samples are available.
        if len(pose_values) < 10:
            continue

        if len(other_values) < 10:
            continue

        # ----------------------------------------------------
        # Median values
        # ----------------------------------------------------

        pose_median = pose_values.median()
        other_median = other_values.median()

        # ----------------------------------------------------
        # Standard deviations
        # ----------------------------------------------------

        pose_std = pose_values.std()
        other_std = other_values.std()

        # ----------------------------------------------------
        # Pooled standard deviation
        # ----------------------------------------------------

        pooled_std = np.sqrt(
            (
                pose_std ** 2 +
                other_std ** 2
            ) / 2
        )

        # ----------------------------------------------------
        # Effect size
        # ----------------------------------------------------

        if pooled_std < 1e-8:
            effect_size = 0

        else:
            effect_size = (
                abs(
                    pose_median - other_median
                ) / pooled_std
            )

        # ----------------------------------------------------
        # Global separation score
        # ----------------------------------------------------

        global_score = separation_df.loc[
            separation_df["feature"] == feature,
            "separation_score"
        ].iloc[0]

        # ----------------------------------------------------
        # Store candidate
        # ----------------------------------------------------

        candidates.append({
            "pose": pose,
            "feature": feature,
            "pose_median": pose_median,
            "other_median": other_median,
            "difference": abs(
                pose_median - other_median
            ),
            "effect_size": effect_size,
            "pose_missing_percent": pose_missing_percent,
            "global_separation_score": global_score
        })

    # ========================================================
    # SORT CANDIDATES
    # ========================================================

    candidates_df = pd.DataFrame(candidates)

    candidates_df = candidates_df.sort_values(
        [
            "effect_size",
            "global_separation_score"
        ],
        ascending=False
    )

    # Select the best features for this pose.
    top_candidates = candidates_df.head(
        TOP_FEATURES_PER_POSE
    )

    print(
        top_candidates.round(3).to_string(
            index=False
        )
    )

    results.append(top_candidates)


# ============================================================
# COMBINE RESULTS
# ============================================================

final_df = pd.concat(
    results,
    ignore_index=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

final_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("POSE RULE FEATURE SELECTION COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_CSV)

print(
    f"\nTotal candidate rows: {len(final_df)}"
)