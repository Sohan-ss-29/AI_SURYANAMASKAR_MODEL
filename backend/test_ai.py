import cv2
import sys
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, BASE_DIR)

from backend.ai_processor import AIProcessor


IMAGE_PATH = os.path.join(
    BASE_DIR,
    "dataset_split",
    "test",
    "1_Pranamasana",
    "druthi1_0000.jpg"
)


print("=" * 50)
print("Testing Surya AI inference")
print("=" * 50)

ai = AIProcessor()

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("ERROR: Image could not be loaded:")
    print(IMAGE_PATH)
    sys.exit(1)

print("Image loaded successfully.")

result = ai.process_frame(frame)

print("\nAI RESULT")
print("=" * 50)

for key, value in result.items():
    print(f"{key}: {value}")

ai.close()

print("=" * 50)
print("AI inference test completed.")