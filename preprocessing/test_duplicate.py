import os

from duplicate_detector import DuplicateDetector

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

folder = os.path.join(
    project_root,
    "suryanamaskar_dataset",
    "1_Pranamasana"
)

images = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

detector = DuplicateDetector()

img1 = os.path.join(folder, images[0])
img2 = os.path.join(folder, images[1])

duplicate, distance = detector.is_duplicate(img1, img2)

print("Image 1 :", images[0])
print("Image 2 :", images[1])
print("Hash Distance :", distance)
print("Duplicate :", duplicate)