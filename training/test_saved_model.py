import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "models/suryanamaskar_mobilenetv2_best.keras"

print("TensorFlow:", tf.__version__)
print("Loading saved model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

print("\n✅ MODEL LOADED SUCCESSFULLY")
print("Model name:", model.name)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Total parameters:", model.count_params())