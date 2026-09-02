import os
from PIL import Image
import imagehash

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "suryanamaskar_dataset"
)

DUPLICATE_REVIEW = os.path.join(
    PROJECT_ROOT,
    "review",
    "duplicate"
)

# ============================================================
# SETTINGS
# ============================================================

HASH_SIZE = 16

# ============================================================
# FUNCTIONS
# ============================================================

def calculate_hash(image_path):
    try:
        image = Image.open(image_path)

        return imagehash.phash(
            image,
            hash_size=HASH_SIZE
        )

    except Exception:
        return None


def get_images(folder):
    return [
        file
        for file in os.listdir(folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


# ============================================================
# ANALYSIS
# ============================================================

print("=" * 70)
print("CORRECTED DUPLICATE ANALYSIS")
print("=" * 70)

overall_distances = []

# Store all exact duplicate filenames
all_exact_duplicates = []


for pose_name in sorted(
    os.listdir(DUPLICATE_REVIEW)
):

    duplicate_pose_path = os.path.join(
        DUPLICATE_REVIEW,
        pose_name
    )

    original_pose_path = os.path.join(
        DATASET_PATH,
        pose_name
    )

    if not os.path.isdir(
        duplicate_pose_path
    ):
        continue

    if not os.path.isdir(
        original_pose_path
    ):
        continue

    duplicate_images = get_images(
        duplicate_pose_path
    )

    original_images = get_images(
        original_pose_path
    )

    print("\n" + "-" * 70)
    print(f"POSE: {pose_name}")
    print("-" * 70)

    print(
        f"Flagged duplicates : "
        f"{len(duplicate_images)}"
    )

    # --------------------------------------------------------
    # Identify flagged filenames
    # --------------------------------------------------------

    duplicate_names = set(
        duplicate_images
    )

    # --------------------------------------------------------
    # Calculate hashes ONLY for accepted/original images
    # --------------------------------------------------------

    accepted_hashes = []

    for filename in original_images:

        if filename in duplicate_names:
            continue

        image_path = os.path.join(
            original_pose_path,
            filename
        )

        image_hash = calculate_hash(
            image_path
        )

        if image_hash is not None:

            accepted_hashes.append(
                (filename, image_hash)
            )

    print(
        f"Accepted images    : "
        f"{len(accepted_hashes)}"
    )

    # --------------------------------------------------------
    # Find closest ACCEPTED image
    # --------------------------------------------------------

    pose_distances = []

    for index, filename in enumerate(
        duplicate_images
    ):

        duplicate_path = os.path.join(
            duplicate_pose_path,
            filename
        )

        duplicate_hash = calculate_hash(
            duplicate_path
        )

        if duplicate_hash is None:
            continue

        best_distance = None
        best_match = None

        for (
            accepted_filename,
            accepted_hash
        ) in accepted_hashes:

            distance = (
                duplicate_hash -
                accepted_hash
            )

            if (
                best_distance is None
                or distance < best_distance
            ):

                best_distance = distance
                best_match = accepted_filename

        if best_distance is not None:

            pose_distances.append(
                best_distance
            )

            overall_distances.append(
                best_distance
            )

        # ----------------------------------------------------
        # Print first 15 examples
        # ----------------------------------------------------

        if index < 15:

            print(
                f"\n{filename}"
            )

            print(
                f"  Closest accepted : "
                f"{best_match}"
            )

            print(
                f"  Distance         : "
                f"{best_distance}"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # Print EVERY exact duplicate (distance 0)
        # ----------------------------------------------------

        if best_distance == 0:

            all_exact_duplicates.append(
                (pose_name, filename, best_match)
            )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print("\nDistance distribution:")

    for distance in range(0, 11):

        count = pose_distances.count(
            distance
        )

        print(
            f"Distance {distance}: "
            f"{count}"
        )

    if pose_distances:

        print(
            f"\nMinimum distance : "
            f"{min(pose_distances)}"
        )

        print(
            f"Maximum distance : "
            f"{max(pose_distances)}"
        )

        print(
            f"Average distance : "
            f"{sum(pose_distances) / len(pose_distances):.2f}"
        )


# ============================================================
# EXACT DUPLICATES
# ============================================================

print("\n")
print("=" * 70)
print("ALL EXACT DUPLICATES (DISTANCE = 0)")
print("=" * 70)

print(
    f"Total exact duplicates: "
    f"{len(all_exact_duplicates)}"
)

for pose_name, duplicate, accepted in all_exact_duplicates:

    print(
        f"\nPose: {pose_name}"
    )

    print(
        f"Duplicate to remove : "
        f"{duplicate}"
    )

    print(
        f"Matching image      : "
        f"{accepted}"
    )

# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL DUPLICATE ANALYSIS")
print("=" * 70)

print(
    f"Total flagged duplicates: "
    f"{len(overall_distances)}"
)

print("\nDistance distribution:")

for distance in range(0, 11):

    count = overall_distances.count(
        distance
    )

    print(
        f"Distance {distance}: "
        f"{count}"
    )

if overall_distances:

    print(
        f"\nMinimum distance : "
        f"{min(overall_distances)}"
    )

    print(
        f"Maximum distance : "
        f"{max(overall_distances)}"
    )

    print(
        f"Average distance : "
        f"{sum(overall_distances) / len(overall_distances):.2f}"
    )

print("\nNo files were deleted or modified.")

print("=" * 70)