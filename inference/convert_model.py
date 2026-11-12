import keras
from keras.applications.mobilenet_v2 import preprocess_input


SOURCE_MODEL = "models/suryanamaskar_mobilenetv2_best.keras"
OUTPUT_MODEL = "models/suryanamaskar_mobilenetv2_converted.keras"


print("Loading original model...")

model = keras.models.load_model(
    SOURCE_MODEL,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

print("Original model loaded successfully.")

print("\nSaving converted model...")

model.save(
    OUTPUT_MODEL,
    include_optimizer=False
)

print("\n========================================")
print("MODEL CONVERSION SUCCESSFUL")
print("========================================")
print("Output:", OUTPUT_MODEL)