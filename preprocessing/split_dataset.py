import os
import shutil
import random
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = "clean_dataset"
OUTPUT_DIR = "dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

# These prefixes identify frames extracted from videos.
VIDEO_PREFIXES = [
    "druthi1",
    "druthi2",
    "druthi3",
    "druthi4",
    "druthi5",
    "shivani",
    "shivani1",
    "shivani2",
    "shivani3",
    "teju",
    "person 1",
]

SPLITS = ["train", "val", "test"]


# ============================================================
# GROUP IDENTIFICATION
# ============================================================

def get_group(filename):
    """
    Return the video group for custom-video frames.

    Example:
        druthi2_0045.jpg -> VIDEO_druthi2
        shivani1_0069.jpg -> VIDEO_shivani1

    Yoga-82 images return None because they are individual images.
    """

    name = os.path.splitext(filename)[0]

    for prefix in VIDEO_PREFIXES:
        if name.startswith(prefix + "_"):
            return f"VIDEO_{prefix}"

    return None


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """
    Returns:

    video_groups:
        {
            VIDEO_druthi1: [(pose, filename), ...],
            ...
        }

    individual_images:
        {
            pose: [filename, ...]
        }
    """

    video_groups = defaultdict(list)
    individual_images = defaultdict(list)

    for pose in sorted(os.listdir(SOURCE_DIR)):

        pose_dir = os.path.join(SOURCE_DIR, pose)

        if not os.path.isdir(pose_dir):
            continue

        for filename in sorted(os.listdir(pose_dir)):

            filepath = os.path.join(pose_dir, filename)

            if not os.path.isfile(filepath):
                continue

            group = get_group(filename)

            if group is not None:
                # IMPORTANT:
                # Group globally by video.
                video_groups[group].append((pose, filename))
            else:
                # Yoga-82 images remain individually splittable.
                individual_images[pose].append(filename)

    return video_groups, individual_images


# ============================================================
# SPLIT VIDEO GROUPS GLOBALLY
# ============================================================

def split_video_groups(video_groups):
    """
    Assign every complete video group to exactly ONE split.

    This guarantees that frames from the same video can never
    appear in train + validation + test simultaneously.
    """

    rng = random.Random(SEED)

    groups = list(video_groups.keys())

    # Shuffle first so assignment isn't biased by filename order.
    rng.shuffle(groups)

    # Large groups first gives better overall balance.
    groups.sort(
        key=lambda g: len(video_groups[g]),
        reverse=True
    )

    total_images = sum(
        len(video_groups[g])
        for g in groups
    )

    targets = {
        "train": total_images * TRAIN_RATIO,
        "val": total_images * VAL_RATIO,
        "test": total_images * TEST_RATIO,
    }

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    assignments = {}

    for group in groups:

        group_size = len(video_groups[group])

        best_split = None
        best_score = None

        for split in SPLITS:

            # Current deficit relative to target.
            deficit = targets[split] - counts[split]

            # Prefer splits that are currently furthest below target.
            score = deficit

            # Strong penalty for overshooting a target.
            new_count = counts[split] + group_size

            if new_count > targets[split]:
                overshoot = new_count - targets[split]
                score -= overshoot * 2

            if best_score is None or score > best_score:
                best_score = score
                best_split = split

        assignments[group] = best_split
        counts[best_split] += group_size

    return assignments


# ============================================================
# SPLIT INDIVIDUAL YOGA-82 IMAGES
# ============================================================

def split_individual_images(individual_images, current_counts):
    """
    Split Yoga-82 images individually.

    Video groups have already been assigned, so we use the
    remaining class-specific images to bring each pose closer
    to the desired 70/15/15 distribution.
    """

    rng = random.Random(SEED)

    assignments = defaultdict(lambda: {
        "train": [],
        "val": [],
        "test": []
    })

    for pose in sorted(individual_images):

        images = list(individual_images[pose])
        rng.shuffle(images)

        total = len(images)

        # Target final class counts.
        # Video counts are accounted for first.
        video_train = current_counts[pose]["train"]
        video_val = current_counts[pose]["val"]
        video_test = current_counts[pose]["test"]

        video_total = video_train + video_val + video_test

        final_total = video_total + total

        target_train = round(final_total * TRAIN_RATIO)
        target_val = round(final_total * VAL_RATIO)
        target_test = final_total - target_train - target_val

        needed_train = max(0, target_train - video_train)
        needed_val = max(0, target_val - video_val)
        needed_test = max(0, target_test - video_test)

        # If rounding/constraints leave a mismatch, distribute
        # remaining images by current deficit.
        remaining = len(images)

        take_train = min(needed_train, remaining)
        remaining -= take_train

        take_val = min(needed_val, remaining)
        remaining -= take_val

        take_test = min(needed_test, remaining)
        remaining -= take_test

        # Any remaining images go to the split with the lowest
        # normalized count.
        index = 0

        assignments[pose]["train"] = images[
            index:index + take_train
        ]
        index += take_train

        assignments[pose]["val"] = images[
            index:index + take_val
        ]
        index += take_val

        assignments[pose]["test"] = images[
            index:index + take_test
        ]
        index += take_test

        while index < len(images):

            current = {
                "train": video_train + len(assignments[pose]["train"]),
                "val": video_val + len(assignments[pose]["val"]),
                "test": video_test + len(assignments[pose]["test"]),
            }

            targets_now = {
                "train": final_total * TRAIN_RATIO,
                "val": final_total * VAL_RATIO,
                "test": final_total * TEST_RATIO,
            }

            best_split = min(
                SPLITS,
                key=lambda s: current[s] / targets_now[s]
            )

            assignments[pose][best_split].append(images[index])

            index += 1

    return assignments


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STRICT GLOBAL GROUP-AWARE DATASET SPLIT")
    print("=" * 70)

    print(f"\nTrain ratio: {TRAIN_RATIO}")
    print(f"Validation ratio: {VAL_RATIO}")
    print(f"Test ratio: {TEST_RATIO}")
    print(f"Random seed: {SEED}")

    # --------------------------------------------------------
    # Remove previous split
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_DIR):
        print("\nRemoving previous dataset_split...")
        shutil.rmtree(OUTPUT_DIR)

    for split in SPLITS:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split),
            exist_ok=True
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    video_groups, individual_images = load_dataset()

    print("\nVideo groups found:", len(video_groups))

    for group in sorted(video_groups):
        print(
            f"  {group}: "
            f"{len(video_groups[group])} frames"
        )

    # --------------------------------------------------------
    # Assign complete video groups
    # --------------------------------------------------------

    video_assignments = split_video_groups(video_groups)

    # Count video images per pose and split.
    current_counts = defaultdict(
        lambda: {
            "train": 0,
            "val": 0,
            "test": 0
        }
    )

    for group, split in video_assignments.items():

        for pose, filename in video_groups[group]:
            current_counts[pose][split] += 1

    # --------------------------------------------------------
    # Split Yoga-82 individual images
    # --------------------------------------------------------

    individual_assignments = split_individual_images(
        individual_images,
        current_counts
    )

    # --------------------------------------------------------
    # Copy everything
    # --------------------------------------------------------

    final_counts = defaultdict(
        lambda: {
            "train": 0,
            "val": 0,
            "test": 0
        }
    )

    # Copy video groups.
    for group, split in video_assignments.items():

        for pose, filename in video_groups[group]:

            src = os.path.join(
                SOURCE_DIR,
                pose,
                filename
            )

            dst_dir = os.path.join(
                OUTPUT_DIR,
                split,
                pose
            )

            os.makedirs(dst_dir, exist_ok=True)

            dst = os.path.join(
                dst_dir,
                filename
            )

            shutil.copy2(src, dst)

            final_counts[pose][split] += 1

    # Copy individual Yoga-82 images.
    for pose, split_data in individual_assignments.items():

        for split, filenames in split_data.items():

            for filename in filenames:

                src = os.path.join(
                    SOURCE_DIR,
                    pose,
                    filename
                )

                dst_dir = os.path.join(
                    OUTPUT_DIR,
                    split,
                    pose
                )

                os.makedirs(dst_dir, exist_ok=True)

                dst = os.path.join(
                    dst_dir,
                    filename
                )

                shutil.copy2(src, dst)

                final_counts[pose][split] += 1

    # --------------------------------------------------------
    # Print final statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL SPLIT")
    print("=" * 70)

    total_train = 0
    total_val = 0
    total_test = 0

    for pose in sorted(final_counts):

        train = final_counts[pose]["train"]
        val = final_counts[pose]["val"]
        test = final_counts[pose]["test"]

        total = train + val + test

        total_train += train
        total_val += val
        total_test += test

        print(f"\n{pose}")
        print(f"  Total:  {total}")
        print(f"  Train:  {train}")
        print(f"  Val:    {val}")
        print(f"  Test:   {test}")

    grand_total = total_train + total_val + total_test

    print("\n")
    print("=" * 70)
    print("OVERALL TOTAL")
    print("=" * 70)

    print(f"Train:      {total_train}")
    print(f"Validation: {total_val}")
    print(f"Test:       {total_test}")
    print(f"Total:      {grand_total}")
    print("Expected:   2188")

    if grand_total == 2188:
        print("\n✅ ALL 2188 IMAGES ACCOUNTED FOR")
    else:
        print(
            f"\n❌ ERROR: Expected 2188 but got {grand_total}"
        )

    print("\nDataset location:")
    print(os.path.abspath(OUTPUT_DIR))

    print("\nOriginal clean_dataset was NOT modified.")

    print("\n")
    print("=" * 70)
    print("VIDEO GROUP ASSIGNMENTS")
    print("=" * 70)

    for group in sorted(video_assignments):
        print(
            f"{group:<25} -> "
            f"{video_assignments[group]}"
        )


if __name__ == "__main__":
    main()