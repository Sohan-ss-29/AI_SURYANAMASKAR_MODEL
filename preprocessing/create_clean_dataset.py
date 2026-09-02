import os
import shutil

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DATASET = os.path.join(
    PROJECT_ROOT,
    "suryanamaskar_dataset"
)

CLEAN_DATASET = os.path.join(
    PROJECT_ROOT,
    "clean_dataset"
)

REVIEW_FOLDER = os.path.join(
    PROJECT_ROOT,
    "review"
)

# ============================================================
# POSES
# ============================================================

POSES = [
    "1_Pranamasana",
    "2_Hasta_Uttanasana",
    "3_Padahastasana",
    "4_AshwaSanchalanasana",
    "5_Parvatasana",
    "6_Ashtanga_Namaskara",
    "7_Bhujangasana"
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# ============================================================
# EXACT DUPLICATES
# These are the 13 files with pHash distance = 0
# ============================================================

EXACT_DUPLICATES = {

    "1_Pranamasana": {
        "druthi1_0035.jpg",
        "shivani1_0069.jpg"
    },

    "2_Hasta_Uttanasana": {
        "22.jpg"
    },

    "3_Padahastasana": {
        "9.jpg",
        "Standing_Forward_Bend_pose_or_Uttanasana__image_189.jpg",
        "Standing_Forward_Bend_pose_or_Uttanasana__image_211.jpg",
        "Standing_Forward_Bend_pose_or_Uttanasana__image_264.jpg",
        "Standing_Forward_Bend_pose_or_Uttanasana__image_41.jpg"
    },

    "6_Ashtanga_Namaskara": {
        "druthi3_0011.jpg",
        "person 1_0031.jpg",
        "person 1_0056.jpg",
        "shivani1_0037.jpg",
        "shivani_0049.jpg"
    }
}

# ============================================================
# GET FILES FROM REVIEW FOLDERS
# ============================================================

def get_review_filenames(review_type, pose_name):

    folder = os.path.join(
        REVIEW_FOLDER,
        review_type,
        pose_name
    )

    if not os.path.exists(folder):
        return set()

    return {
        filename
        for filename in os.listdir(folder)
        if filename.lower().endswith(
            IMAGE_EXTENSIONS
        )
    }


# ============================================================
# CREATE CLEAN DATASET
# ============================================================

print("=" * 70)
print("CREATING FINAL CLEAN DATASET")
print("=" * 70)

os.makedirs(
    CLEAN_DATASET,
    exist_ok=True
)

total_raw = 0
total_blurry = 0
total_no_pose = 0
total_exact_duplicates = 0
total_clean = 0


for pose in POSES:

    source_folder = os.path.join(
        RAW_DATASET,
        pose
    )

    destination_folder = os.path.join(
        CLEAN_DATASET,
        pose
    )

    os.makedirs(
        destination_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Get actual blurry/no-pose filenames from review folders
    # --------------------------------------------------------

    blurry_files = get_review_filenames(
        "blurry",
        pose
    )

    no_pose_files = get_review_filenames(
        "no_pose",
        pose
    )

    exact_duplicates = EXACT_DUPLICATES.get(
        pose,
        set()
    )

    pose_raw = 0
    pose_blurry = 0
    pose_no_pose = 0
    pose_duplicates = 0
    pose_clean = 0

    if not os.path.exists(source_folder):

        print(
            f"\nWARNING: Missing folder: {pose}"
        )

        continue

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for filename in os.listdir(
        source_folder
    ):

        if not filename.lower().endswith(
            IMAGE_EXTENSIONS
        ):
            continue

        pose_raw += 1
        total_raw += 1

        # ----------------------------------------------------
        # Remove blurry
        # ----------------------------------------------------

        if filename in blurry_files:

            pose_blurry += 1
            total_blurry += 1

            continue

        # ----------------------------------------------------
        # Remove no-pose
        # ----------------------------------------------------

        if filename in no_pose_files:

            pose_no_pose += 1
            total_no_pose += 1

            continue

        # ----------------------------------------------------
        # Remove ONLY exact pHash duplicates
        # ----------------------------------------------------

        if filename in exact_duplicates:

            pose_duplicates += 1
            total_exact_duplicates += 1

            continue

        # ----------------------------------------------------
        # Copy clean image
        # ----------------------------------------------------

        source_path = os.path.join(
            source_folder,
            filename
        )

        destination_path = os.path.join(
            destination_folder,
            filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        pose_clean += 1
        total_clean += 1

    # --------------------------------------------------------
    # Pose summary
    # --------------------------------------------------------

    print(
        f"\n{pose}"
    )

    print(
        f"  Raw:              {pose_raw}"
    )

    print(
        f"  Blurry removed:   {pose_blurry}"
    )

    print(
        f"  No-pose removed:  {pose_no_pose}"
    )

    print(
        f"  Exact dupes:      {pose_duplicates}"
    )

    print(
        f"  Clean:            {pose_clean}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

total_removed = (
    total_blurry
    + total_no_pose
    + total_exact_duplicates
)

print("\n")
print("=" * 70)
print("FINAL CLEAN DATASET")
print("=" * 70)

print(
    f"Raw images:              {total_raw}"
)

print(
    f"Blurry removed:          {total_blurry}"
)

print(
    f"No-pose removed:         {total_no_pose}"
)

print(
    f"Exact duplicates removed:{total_exact_duplicates}"
)

print(
    f"Total removed:           {total_removed}"
)

print(
    f"Final clean images:      {total_clean}"
)

print("\nExpected:")

print(
    "Raw images:              2233"
)

print(
    "Blurry:                  17"
)

print(
    "No-pose:                 15"
)

print(
    "Exact duplicates:        13"
)

print(
    "Final clean:             2188"
)

print("\nClean dataset location:")

print(
    CLEAN_DATASET
)

print("\nOriginal dataset was NOT modified.")

print("=" * 70)