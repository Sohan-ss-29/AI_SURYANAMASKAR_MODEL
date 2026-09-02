import os
from blur_detector import BlurDetector

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset folder
folder = os.path.join(
    project_root,
    "suryanamaskar_dataset",
    "1_Pranamasana"
)

print("Dataset Folder:")
print(folder)

# Get first image automatically
images = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if len(images) == 0:
    print("No images found!")
    exit()

image_path = os.path.join(folder, images[0])

print("\nTesting Image:")
print(image_path)

detector = BlurDetector()

blurry, score = detector.is_blurry(image_path)

print("\nBlur Score :", score)
print("Blurry     :", blurry)