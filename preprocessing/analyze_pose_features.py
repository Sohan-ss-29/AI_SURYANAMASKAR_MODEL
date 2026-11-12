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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "pose_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("SURYA NAMASKARA - POSE FEATURE ANALYSIS")
print("=" * 70)

print(f"\nInput file:")
print(INPUT_CSV)

print(f"\nTotal feature rows: {len(df)}")

print("\nClass distribution:")
print(df["pose"].value_counts().sort_index())


# ============================================================
# FEATURE COLUMNS
# ============================================================

EXCLUDE_COLUMNS = {
    "image",
    "pose"
}

FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in EXCLUDE_COLUMNS
]


# ============================================================
# 1. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("1. MISSING VALUE ANALYSIS")
print("=" * 70)

missing = df[FEATURE_COLUMNS].isna().sum()
missing_percent = (
    missing / len(df) * 100
).round(2)

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent
})

missing_table = missing_table.sort_values(
    "missing_percent",
    ascending=False
)

print("\nMissing values:")
print(missing_table)

missing_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missing_value_analysis.csv"
    )
)


# ============================================================
# 2. GLOBAL FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("2. GLOBAL FEATURE STATISTICS")
print("=" * 70)

global_stats = df[FEATURE_COLUMNS].describe(
    percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
).T

global_stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "global_feature_statistics.csv"
    )
)

print(global_stats)


# ============================================================
# 3. PER-POSE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("3. PER-POSE STATISTICS")
print("=" * 70)

pose_stats = []

for pose in sorted(df["pose"].unique()):

    pose_df = df[df["pose"] == pose]

    print("\n" + "-" * 70)
    print(f"POSE: {pose}")
    print("-" * 70)

    for feature in FEATURE_COLUMNS:

        values = pose_df[feature].dropna()

        if len(values) == 0:
            continue

        row = {
            "pose": pose,
            "feature": feature,
            "count": len(values),
            "missing_percent":
                round(
                    pose_df[feature].isna().mean() * 100,
                    2
                ),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "p10": values.quantile(0.10),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "p90": values.quantile(0.90),
            "max": values.max()
        }

        pose_stats.append(row)

pose_stats_df = pd.DataFrame(pose_stats)

pose_stats_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_pose_statistics.csv"
    ),
    index=False
)

print("\nPer-pose statistics saved.")


# ============================================================
# 4. MEDIAN TABLE
# ============================================================

print("\n" + "=" * 70)
print("4. MEDIAN FEATURE VALUES BY POSE")
print("=" * 70)

median_table = (
    df.groupby("pose")[FEATURE_COLUMNS]
    .median()
    .round(2)
)

print(median_table)

median_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "median_features_by_pose.csv"
    )
)


# ============================================================
# 5. MEAN TABLE
# ============================================================

print("\n" + "=" * 70)
print("5. MEAN FEATURE VALUES BY POSE")
print("=" * 70)

mean_table = (
    df.groupby("pose")[FEATURE_COLUMNS]
    .mean()
    .round(2)
)

print(mean_table)

mean_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "mean_features_by_pose.csv"
    )
)


# ============================================================
# 6. FEATURE SEPARATION SCORE
# ============================================================

print("\n" + "=" * 70)
print("6. FEATURE SEPARATION ANALYSIS")
print("=" * 70)

separation_results = []

for feature in FEATURE_COLUMNS:

    feature_df = df[
        ["pose", feature]
    ].dropna()

    if len(feature_df) < 10:
        continue

    overall_mean = feature_df[feature].mean()

    between_variance = 0.0
    within_variance = 0.0

    total_count = len(feature_df)

    for pose in feature_df["pose"].unique():

        group = feature_df[
            feature_df["pose"] == pose
        ][feature]

        n = len(group)

        group_mean = group.mean()

        between_variance += (
            n *
            (group_mean - overall_mean) ** 2
        )

        within_variance += (
            ((group - group_mean) ** 2).sum()
        )

    between_variance /= total_count
    within_variance /= total_count

    if within_variance > 1e-8:
        separation_score = (
            between_variance /
            within_variance
        )
    else:
        separation_score = np.inf

    separation_results.append({
        "feature": feature,
        "separation_score":
            separation_score,
        "missing_percent":
            df[feature].isna().mean() * 100
    })


separation_df = pd.DataFrame(
    separation_results
)

separation_df = separation_df.sort_values(
    "separation_score",
    ascending=False
)

print(
    "\nFeatures ranked by between-pose "
    "separation / within-pose variation:"
)

print(
    separation_df.head(20).round(3)
)

separation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_separation_scores.csv"
    ),
    index=False
)


# ============================================================
# 7. RELIABLE + USEFUL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("7. RELIABLE + USEFUL FEATURES")
print("=" * 70)

reliable_features = separation_df[
    separation_df["missing_percent"] <= 30
].copy()

reliable_features = reliable_features.sort_values(
    "separation_score",
    ascending=False
)

print(
    "\nTotal features analyzed:",
    len(FEATURE_COLUMNS)
)

print(
    "Reliable features (<=30% missing):",
    len(reliable_features)
)

print(
    "\nTop features with <=30% missing values:"
)

print(
    reliable_features.head(20).round(3)
)

reliable_features.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "reliable_useful_features.csv"
    ),
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nAnalysis files saved to:")

print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(" -", filename)