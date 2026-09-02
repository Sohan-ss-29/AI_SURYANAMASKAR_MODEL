import os
from collections import defaultdict

BASE_DIR = "dataset_split"

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


def get_group(filename):
    name = os.path.splitext(filename)[0]

    for prefix in VIDEO_PREFIXES:
        if name.startswith(prefix + "_"):
            return f"VIDEO_{prefix}"

    return None


groups = defaultdict(set)

for split in ["train", "val", "test"]:
    split_dir = os.path.join(BASE_DIR, split)

    for pose in os.listdir(split_dir):
        pose_dir = os.path.join(split_dir, pose)

        if not os.path.isdir(pose_dir):
            continue

        for filename in os.listdir(pose_dir):
            group = get_group(filename)

            if group is not None:
                groups[group].add(split)


print("=" * 70)
print("DATASET LEAKAGE CHECK")
print("=" * 70)

leaks = []

for group, splits in sorted(groups.items()):
    if len(splits) > 1:
        leaks.append((group, sorted(splits)))

if leaks:
    print("\n❌ DATA LEAKAGE FOUND\n")

    for group, splits in leaks:
        print(f"{group}: {', '.join(splits)}")

    print(f"\nTotal leaking groups: {len(leaks)}")
else:
    print("\n✅ NO VIDEO GROUP LEAKAGE FOUND")

print("\nVideo groups checked:", len(groups))
print("=" * 70)