import os
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 15

INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

DATASET_DIR = "dataset_split"
MODEL_DIR = "models"
REPORT_DIR = "reports"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "suryanamaskar_mobilenetv2_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "suryanamaskar_mobilenetv2_final.keras"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# GPU / CPU INFORMATION
# ============================================================

print("=" * 70)
print("SURYA NAMASKARA - MOBILENETV2 TRAINING")
print("=" * 70)

print("\nTensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)
else:
    print("GPU not detected.")
    print("Training will run on CPU.")


# ============================================================
# DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "val")
test_dir = os.path.join(DATASET_DIR, "test")


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

print("\nNumber of classes:", num_classes)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=4)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

train_labels = []

for _, labels in train_ds.unbatch():
    train_labels.append(int(labels.numpy()))

train_labels = np.array(train_labels)

class_indices = np.arange(num_classes)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=class_indices,
    y=train_labels
)

class_weights = {
    int(i): float(weight)
    for i, weight in enumerate(class_weights_array)
}

for i, name in enumerate(class_names):
    print(
        f"{name:<30} "
        f"weight = {class_weights[i]:.4f}"
    )


# ============================================================
# BUILD MOBILE NET V2
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MOBILENETV2")
print("=" * 70)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


inputs = keras.Input(
    shape=(224, 224, 3),
    name="input_image"
)

x = data_augmentation(inputs)

x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0,
    name="mobilenetv2_preprocess"
)(x)
x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.30,
    name="dropout"
)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="pose_classifier"
)(x)

model = keras.Model(
    inputs,
    outputs,
    name="SuryaNamaskara_MobileNetV2"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# STAGE 1 - TRANSFER LEARNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 - TRANSFER LEARNING")
print("=" * 70)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        reduce_lr_callback
    ]
)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 - FINE TUNING")
print("=" * 70)

base_model.trainable = True


# Freeze the lower layers.
# Only the upper part of MobileNetV2 will be fine-tuned.

fine_tune_from = 100

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False


print(
    f"\nFine-tuning MobileNetV2 from layer "
    f"{fine_tune_from} onward."
)

trainable_layers = sum(
    layer.trainable
    for layer in base_model.layers
)

print(
    "Trainable MobileNetV2 layers:",
    trainable_layers
)


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        reduce_lr_callback
    ]
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

best_model = keras.models.load_model(
    BEST_MODEL_PATH
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = best_model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)


report_path = os.path.join(
    REPORT_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as f:
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

cm_path = os.path.join(
    REPORT_DIR,
    "confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title("Surya Namaskara - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    range(num_classes),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(num_classes),
    class_names
)

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_plot_path = os.path.join(
    REPORT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# TRAINING HISTORY
# ============================================================

def combine_history(h1, h2):

    history = {}

    for key in h1.history.keys():
        history[key] = (
            h1.history[key]
            + h2.history.get(key, [])
        )

    return history


combined_history = combine_history(
    history_stage1,
    history_stage2
)


# ============================================================
# ACCURACY PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    combined_history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    combined_history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Training and Validation Accuracy")
plt.legend()
plt.grid(True)

accuracy_plot_path = os.path.join(
    REPORT_DIR,
    "training_accuracy.png"
)

plt.savefig(
    accuracy_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# LOSS PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    combined_history["loss"],
    label="Training Loss"
)

plt.plot(
    combined_history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 Training and Validation Loss")
plt.legend()
plt.grid(True)

loss_plot_path = os.path.join(
    REPORT_DIR,
    "training_loss.png"
)

plt.savefig(
    loss_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    REPORT_DIR,
    "training_history.json"
)

with open(history_path, "w") as f:
    json.dump(
        {
            key: [float(v) for v in values]
            for key, values in combined_history.items()
        },
        f,
        indent=4
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

best_model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"\nBest model:")
print(BEST_MODEL_PATH)

print(f"\nFinal model:")
print(FINAL_MODEL_PATH)

print(f"\nClass names:")
print(class_names_path)

print(f"\nClassification report:")
print(report_path)

print(f"\nConfusion matrix:")
print(confusion_plot_path)

print(f"\nAccuracy plot:")
print(accuracy_plot_path)

print(f"\nLoss plot:")
print(loss_plot_path)

print("\n" + "=" * 70)
print(f"FINAL TEST ACCURACY: {test_accuracy * 100:.2f}%")
print("=" * 70)