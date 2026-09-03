import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/suryanamaskar_mobilenetv2_best.keras"

CLASS_NAMES = [
    "1_Pranamasana",
    "2_Hasta_Uttanasana",
    "3_Padahastasana",
    "4_AshwaSanchalanasana",
    "5_Parvatasana",
    "6_Ashtanga_Namaskara",
    "7_Bhujangasana",
]

IMAGE_SIZE = (224, 224)

# Only display predictions above this confidence
CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

print("TensorFlow:", tf.__version__)
print("Loading model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

print("✅ Model loaded successfully")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    raise SystemExit


print("\n✅ Webcam started")
print("Press 'q' to quit.")


# ============================================================
# REAL-TIME INFERENCE
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read webcam frame.")
        break

    # Flip horizontally for a mirror-like experience
    frame = cv2.flip(frame, 1)

    # --------------------------------------------------------
    # Prepare image for MobileNetV2
    # --------------------------------------------------------

    image = cv2.resize(frame, IMAGE_SIZE)

    # OpenCV: BGR
    # Model: RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32)

    
    # The saved model already contains MobileNetV2 preprocessing

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(image, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index])

    predicted_class = CLASS_NAMES[predicted_index]

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        display_name = predicted_class.split("_", 1)[1]

        text = f"{display_name}"
        confidence_text = f"Confidence: {confidence * 100:.1f}%"

    else:

        text = "Uncertain"
        confidence_text = f"Confidence: {confidence * 100:.1f}%"

    # Background rectangle
    cv2.rectangle(
        frame,
        (10, 10),
        (500, 95),
        (0, 0, 0),
        -1
    )

    # Pose name
    cv2.putText(
        frame,
        text,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # Confidence
    cv2.putText(
        frame,
        confidence_text,
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Show frame
    cv2.imshow("Surya Namaskara - Real-Time Classifier", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Webcam stopped.")