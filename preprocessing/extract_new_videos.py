import os
import cv2

# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VIDEOS_PATH = os.path.join(PROJECT_PATH, "videos")
DATASET_PATH = os.path.join(PROJECT_PATH, "suryanamaskar_dataset")

FRAME_SKIP = 10
IMAGE_SIZE = (224, 224)

# =====================================================
# NEW VIDEOS ONLY
# =====================================================

NEW_VIDEOS = {
    "1. pranamasana": {
        "output": "1_Pranamasana",
        "videos": [
            "druthi1",
            "druthi2",
            "druthi3",
            "shivani1",
            "shivani2"
        ]
    },

    "2. hasta uttanasana": {
        "output": "2_Hasta_Uttanasana",
        "videos": [
            "shivani2",
            "shivani3",
            "druthi4",
            "druthi5"
        ]
    },

    "6. ashtanga namaskara": {
        "output": "6_Ashtanga_Namaskara",
        "videos": [
            "druthi3",
            "druthi4",
            "shivani",
            "shivani1"
        ]
    }
}

# =====================================================

total_saved = 0

print("=" * 65)
print("SURYA NAMASKAR - NEW VIDEO EXTRACTOR")
print("=" * 65)

for pose_folder, pose_info in NEW_VIDEOS.items():

    video_folder = os.path.join(
        VIDEOS_PATH,
        pose_folder
    )

    output_folder = os.path.join(
        DATASET_PATH,
        pose_info["output"]
    )

    # Check video folder
    if not os.path.exists(video_folder):
        print(f"\n❌ Video folder not found:")
        print(f"   {video_folder}")
        continue

    # Make sure dataset folder exists
    os.makedirs(output_folder, exist_ok=True)

    print("\n" + "-" * 65)
    print(f"POSE: {pose_folder}")
    print(f"OUTPUT: {pose_info['output']}")
    print("-" * 65)

    pose_total = 0

    for video_name in pose_info["videos"]:

        # -------------------------------------------------
        # Find video file
        # -------------------------------------------------

        video_file = None

        for extension in [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"]:

            possible_file = os.path.join(
                video_folder,
                video_name + extension
            )

            if os.path.exists(possible_file):
                video_file = possible_file
                break

        if video_file is None:
            print(f"❌ {video_name} -> VIDEO NOT FOUND")
            continue

        # -------------------------------------------------
        # Open video
        # -------------------------------------------------

        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"❌ {video_name} -> Could not open video")
            continue

        # -------------------------------------------------
        # Find existing images for this video
        # -------------------------------------------------

        existing_images = [
            f for f in os.listdir(output_folder)
            if f.startswith(video_name + "_")
            and f.lower().endswith(".jpg")
        ]

        # -------------------------------------------------
        # SAFETY CHECK
        # -------------------------------------------------

        if existing_images:

            print(
                f"⚠️ {video_name} -> "
                f"{len(existing_images)} existing images found. "
                f"SKIPPING to prevent duplicates."
            )

            cap.release()
            continue

        # -------------------------------------------------
        # Extract frames
        # -------------------------------------------------

        frame_count = 0
        image_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:

                frame = cv2.resize(
                    frame,
                    IMAGE_SIZE
                )

                filename = (
                    f"{video_name}_{image_count:04d}.jpg"
                )

                output_path = os.path.join(
                    output_folder,
                    filename
                )

                cv2.imwrite(
                    output_path,
                    frame
                )

                image_count += 1
                pose_total += 1
                total_saved += 1

            frame_count += 1

        cap.release()

        print(
            f"✅ {video_name}.mp4 -> "
            f"{image_count} images"
        )

    print(
        f"\nTotal images added for this pose: "
        f"{pose_total}"
    )

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n" + "=" * 65)
print("NEW VIDEO EXTRACTION COMPLETED")
print("=" * 65)
print(f"Total NEW images extracted: {total_saved}")
print("=" * 65)