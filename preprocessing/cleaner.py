import os
import shutil
import csv
import cv2
import imagehash
from PIL import Image
import mediapipe as mp

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

REVIEW_PATH = os.path.join(
    PROJECT_ROOT,
    "review"
)

REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "reports",
    "dataset_report.csv"
)

# ============================================================
# SETTINGS
# ============================================================

BLUR_THRESHOLD = 120

HASH_SIZE = 16

HASH_THRESHOLD = 5

MIN_VISIBILITY = 0.60

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# ============================================================
# POSE-SPECIFIC CLEANING
# ============================================================

STRICT_CLEANING_POSES = {
    "1_Pranamasana",
    "2_Hasta_Uttanasana",
    "6_Ashtanga_Namaskara"
}

SKIP_BLUR_POSES = {
    "3_Padahastasana",
    "4_AshwaSanchalanasana",
    "5_Parvatasana",
    "7_Bhujangasana"
}

# ============================================================
# REVIEW FOLDERS
# ============================================================

BLUR_REVIEW = os.path.join(
    REVIEW_PATH,
    "blurry"
)

DUPLICATE_REVIEW = os.path.join(
    REVIEW_PATH,
    "duplicate"
)

NOPOSE_REVIEW = os.path.join(
    REVIEW_PATH,
    "no_pose"
)

os.makedirs(BLUR_REVIEW, exist_ok=True)
os.makedirs(DUPLICATE_REVIEW, exist_ok=True)
os.makedirs(NOPOSE_REVIEW, exist_ok=True)

os.makedirs(
    os.path.dirname(REPORT_PATH),
    exist_ok=True
)

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_pose = mp.solutions.pose

pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_blur_score(image):
    """
    Calculate Variance of Laplacian.

    Higher score = sharper image.
    Lower score = blurrier image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def calculate_hash(image_path):
    """
    Calculate perceptual hash.
    """

    try:

        image = Image.open(image_path)

        return imagehash.phash(
            image,
            hash_size=HASH_SIZE
        )

    except Exception:

        return None


def check_body_visibility(image):
    """
    Forgiving MediaPipe validation.

    The purpose is NOT to demand a perfect pose.

    We only want to reject images where MediaPipe
    cannot reasonably detect the person/body.

    Returns:
        True  -> image is usable
        False -> image is probably unusable
    """

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    result = pose_detector.process(rgb)

    # ----------------------------------------------------
    # No person detected
    # ----------------------------------------------------

    if not result.pose_landmarks:
        return False

    landmarks = result.pose_landmarks.landmark

    # ----------------------------------------------------
    # Important upper-body landmarks
    # ----------------------------------------------------

    # We intentionally DO NOT require:
    # nose
    # knees
    # ankles

    # because some yoga poses naturally hide/occlude
    # these landmarks.

    # ----------------------------------------------------

    important_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,

        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,

        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,

        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    # ----------------------------------------------------
    # Forgiving visibility threshold
    # ----------------------------------------------------

    visibility_threshold = 0.30

    visible_count = 0

    for landmark_id in important_landmarks:

        landmark = landmarks[
            landmark_id.value
        ]

        if landmark.visibility >= visibility_threshold:
            visible_count += 1

    # ----------------------------------------------------
    # KEEP IMAGE if at least 4 of the 8 important
    # landmarks are reasonably visible.
    # ----------------------------------------------------

    if visible_count >= 4:
        return True

    return False


def copy_to_review(
        image_path,
        review_folder,
        pose_name,
        filename
):
    """
    Copy suspicious image to review folder.
    """

    pose_review_folder = os.path.join(
        review_folder,
        pose_name
    )

    os.makedirs(
        pose_review_folder,
        exist_ok=True
    )

    destination = os.path.join(
        pose_review_folder,
        filename
    )

    # Avoid overwriting an existing review image
    if os.path.exists(destination):

        base, ext = os.path.splitext(filename)

        counter = 1

        while os.path.exists(destination):

            new_name = (
                f"{base}_{counter}{ext}"
            )

            destination = os.path.join(
                pose_review_folder,
                new_name
            )

            counter += 1

    shutil.copy2(
        image_path,
        destination
    )


# ============================================================
# DATASET CLEANING
# ============================================================

print("=" * 70)
print("SURYA NAMASKAR DATASET CLEANER V1")
print("=" * 70)

print("\nDataset:")
print(DATASET_PATH)

print("\nImportant:")
print("Original images WILL NOT be deleted.")
print("Suspicious images will be copied to review/.")

print("\n" + "=" * 70)

# ============================================================
# GET POSE FOLDERS
# ============================================================

pose_folders = [
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(
            DATASET_PATH,
            folder
        )
    )
]

pose_folders.sort()

# ============================================================
# REPORT STORAGE
# ============================================================

report_rows = []

overall_original = 0
overall_blurry = 0
overall_duplicate = 0
overall_no_pose = 0
overall_valid = 0

# ============================================================
# PROCESS EACH POSE
# ============================================================

for pose_name in pose_folders:

    pose_path = os.path.join(
        DATASET_PATH,
        pose_name
    )

    images = [
        file
        for file in os.listdir(pose_path)
        if file.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ]

    images.sort()

    print("\n" + "-" * 70)
    print(f"POSE: {pose_name}")
    print("-" * 70)

    print(
        f"Original images: {len(images)}"
    )

    original_count = len(images)

    blurry_count = 0
    duplicate_count = 0
    no_pose_count = 0
    valid_count = 0

    # ========================================================
    # HASH STORAGE
    # ========================================================

    image_hashes = []

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for index, filename in enumerate(images):

        image_path = os.path.join(
            pose_path,
            filename
        )

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:

            print(
                f"[NO IMAGE] {filename}"
            )

            copy_to_review(
                image_path,
                NOPOSE_REVIEW,
                pose_name,
                filename
            )

            no_pose_count += 1

            continue

        # ----------------------------------------------------
        # BLUR CHECK
        # ----------------------------------------------------

        # Only perform blur detection on poses where
        # custom video frames were added.
        #
        # Poses 3, 4, 5 and 7 are already curated
        # Yoga-82 images, so blur checking is skipped.

        if pose_name not in SKIP_BLUR_POSES:

            blur_score = calculate_blur_score(
                image
            )

            if blur_score < BLUR_THRESHOLD:

                copy_to_review(
                    image_path,
                    BLUR_REVIEW,
                    pose_name,
                    filename
                )

                blurry_count += 1

                continue

        # ----------------------------------------------------
        # DUPLICATE CHECK
        # ----------------------------------------------------

        current_hash = calculate_hash(
            image_path
        )

        if current_hash is None:

            copy_to_review(
                image_path,
                NOPOSE_REVIEW,
                pose_name,
                filename
            )

            no_pose_count += 1

            continue

        is_duplicate = False

        for previous_hash in image_hashes:

            distance = (
                current_hash -
                previous_hash
            )

            if distance <= HASH_THRESHOLD:

                is_duplicate = True
                break

        if is_duplicate:

            copy_to_review(
                image_path,
                DUPLICATE_REVIEW,
                pose_name,
                filename
            )

            duplicate_count += 1

            continue

        image_hashes.append(
            current_hash
        )

        # ----------------------------------------------------
        # MEDIAPIPE CHECK
        # ----------------------------------------------------

        # Only use MediaPipe validation for the poses
        # containing our custom video data.

        if pose_name in STRICT_CLEANING_POSES:

            valid_body = check_body_visibility(
                image
            )

            if not valid_body:

                copy_to_review(
                    image_path,
                    NOPOSE_REVIEW,
                    pose_name,
                    filename
                )

                no_pose_count += 1

                continue

        # ----------------------------------------------------
        # VALID IMAGE
        # ----------------------------------------------------

        valid_count += 1

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (index + 1) % 25 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(images)}"
            )

    # ========================================================
    # POSE SUMMARY
    # ========================================================

    print("\nResults:")

    print(
        f"Original   : {original_count}"
    )

    print(
        f"Blurry     : {blurry_count}"
    )

    print(
        f"Duplicate  : {duplicate_count}"
    )

    print(
        f"No Pose    : {no_pose_count}"
    )

    print(
        f"Valid      : {valid_count}"
    )

    # ========================================================
    # REPORT ROW
    # ========================================================

    report_rows.append([
        pose_name,
        original_count,
        blurry_count,
        duplicate_count,
        no_pose_count,
        valid_count
    ])

    overall_original += original_count
    overall_blurry += blurry_count
    overall_duplicate += duplicate_count
    overall_no_pose += no_pose_count
    overall_valid += valid_count


# ============================================================
# SAVE CSV REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Pose",
        "Original Images",
        "Blurry",
        "Duplicates",
        "No Pose",
        "Valid Images"
    ])

    for row in report_rows:
        writer.writerow(row)

    writer.writerow([
        "TOTAL",
        overall_original,
        overall_blurry,
        overall_duplicate,
        overall_no_pose,
        overall_valid
    ])


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET CLEANING ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Original Images : {overall_original}"
)

print(
    f"Blurry Images   : {overall_blurry}"
)

print(
    f"Duplicates      : {overall_duplicate}"
)

print(
    f"No Pose         : {overall_no_pose}"
)

print(
    f"Valid Images    : {overall_valid}"
)

print("\nReport saved to:")

print(REPORT_PATH)

print("\nReview folders:")

print(BLUR_REVIEW)
print(DUPLICATE_REVIEW)
print(NOPOSE_REVIEW)

print("\n" + "=" * 70)
print("NO ORIGINAL IMAGES WERE DELETED.")
print("=" * 70)