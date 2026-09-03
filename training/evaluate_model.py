import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

MODEL_PATH = "models/suryanamaskar_mobilenetv2_best.keras"
TEST_DIR = "dataset_split/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "1_Pranamasana",
    "2_Hasta_Uttanasana",
    "3_Padahastasana",
    "4_AshwaSanchalanasana",
    "5_Parvatasana",
    "6_Ashtanga_Namaskara",
    "7_Bhujangasana"
]

print("TensorFlow:", tf.__version__)
print("\nLoading model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

print("✅ Model loaded")

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=CLASS_NAMES
)

print("Test images:", tf.data.experimental.cardinality(test_ds).numpy() * BATCH_SIZE)

# Collect predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Accuracy
accuracy = np.mean(y_true == y_pred)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print("=" * 70)
print(cm)

print("\nActual class order:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")

print("\n✅ Evaluation completed successfully.")