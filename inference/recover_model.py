import os
import zipfile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MODEL_PATH = "models/suryanamaskar_mobilenetv2_best.keras"
WEIGHTS_PATH = "models/model.weights.h5"
OUTPUT_PATH = "models/suryanamaskar_mobilenetv2_recovered.h5"


print("TensorFlow:", tf.__version__)


# ---------------------------------------------------------
# 1. Extract weights if needed
# ---------------------------------------------------------

if not os.path.exists(WEIGHTS_PATH):
    print("\nExtracting weights...")

    with zipfile.ZipFile(MODEL_PATH, "r") as z:
        z.extract("model.weights.h5", "models")

    print("Weights extracted.")

else:
    print("\nWeights already extracted.")


# ---------------------------------------------------------
# 2. Build the original architecture
# ---------------------------------------------------------

print("\nBuilding MobileNetV2 model...")


inputs = keras.Input(
    shape=(224, 224, 3),
    name="input_image"
)


# Data augmentation
augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            mode="horizontal"
        ),
        layers.RandomRotation(
            factor=0.05
        ),
        layers.RandomZoom(
            height_factor=0.1,
            width_factor=0.1
        ),
        layers.RandomContrast(
            factor=0.1
        ),
    ],
    name="data_augmentation"
)

x = augmentation(inputs)


# MobileNetV2 preprocessing
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0,
    name="mobilenetv2_preprocess"
)(x)


# MobileNetV2 backbone
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = base_model(x)


# Classification head
x = layers.GlobalAveragePooling2D(
    name="global_average_pooling2d"
)(x)

x = layers.Dropout(
    0.3,
    name="dropout"
)(x)

outputs = layers.Dense(
    7,
    activation="softmax",
    name="pose_classifier"
)(x)


model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="functional"
)


print("\nModel created.")

print("\nLoading trained weights...")


# ---------------------------------------------------------
# 3. Load trained weights
# ---------------------------------------------------------

model.load_weights(
    WEIGHTS_PATH
)

print("\n========================================")
print("TRAINED WEIGHTS LOADED SUCCESSFULLY!")
print("========================================")


# ---------------------------------------------------------
# 4. Save recovered model
# ---------------------------------------------------------

print("\nSaving recovered model...")

model.save(
    OUTPUT_PATH,
    include_optimizer=False
)

print("\n========================================")
print("MODEL RECOVERY COMPLETE")
print("========================================")

print("Saved model:")
print(OUTPUT_PATH)