# AI-Based Surya Namaskara Correction System

An AI-powered computer vision system for recognizing and evaluating Surya Namaskara yoga postures using image classification, pose estimation, and real-time feedback.

## Project Overview

Surya Namaskara is a sequence of yoga postures that requires correct body alignment and controlled movement. Incorrect posture during the sequence can reduce its effectiveness and may lead to physical strain.

This project aims to develop an AI-based system that can:

- Recognize individual Surya Namaskara postures.
- Classify the detected posture using a deep learning model.
- Analyze body landmarks using MediaPipe Pose.
- Evaluate posture correctness.
- Provide real-time visual and audio feedback.
- Track the progression through the Surya Namaskara sequence.

The system combines **deep learning-based image classification** with **computer vision-based pose analysis**.

---

## Surya Namaskara Poses

The current dataset contains seven major postures:

| Class | Pose |
|---|---|
| 1 | Pranamasana |
| 2 | Hasta Uttanasana |
| 3 | Padahastasana |
| 4 | Ashwa Sanchalanasana |
| 5 | Parvatasana |
| 6 | Ashtanga Namaskara |
| 7 | Bhujangasana |

---

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- MediaPipe Pose
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Git / GitHub

---

## System Architecture

```text
Input Image / Video
        |
        v
Image Preprocessing
        |
        v
MobileNetV2
Transfer Learning Model
        |
        v
7-Class Pose Classification
        |
        v
Detected Yoga Pose
        |
        +----------------------+
        |                      |
        v                      v
MediaPipe Pose          Posture Analysis
        |                      |
        v                      v
Body Landmarks         Alignment / Form
        |                      |
        +----------+-----------+
                   |
                   v
          Accuracy Evaluation
                   |
                   v
        Visual / Voice Feedback

## Dataset

The project uses a combination of publicly available yoga images and custom video data.

Dataset sources
Selected images from the Yoga-82 dataset.
Custom videos recorded for poses that required additional or more diverse examples.

Custom video frames were extracted and resized to 224 × 224 pixels for compatibility with the MobileNetV2 input pipeline.

The raw dataset initially contained:

2,233 images

After preprocessing and cleaning:

2,188 images

Dataset Preprocessing

The preprocessing pipeline was designed to improve dataset quality before model training.

Raw Dataset
     |
     v
Readable / Corrupt Image Check
     |
     v
Blur Detection
     |
     v
Duplicate Detection
     |
     v
MediaPipe Pose Validation
     |
     v
Manual Review
     |
     v
Clean Dataset
     |
     v
Group-Aware Dataset Split
Cleaning

The preprocessing pipeline identifies:

Blurry images
Duplicate images
Images where a human pose cannot be detected
Invalid or unreadable images

The original dataset is preserved. Suspicious samples are moved to review directories rather than deleting the original data.

Duplicate handling

Exact duplicate images were removed from the clean dataset.

Near-duplicate frames were retained when they represented meaningful variations of the same posture.

MediaPipe validation

MediaPipe Pose is used as a validation step for the custom-video images.

The validation checks important upper/lower body landmarks and accepts an image when sufficient reliable landmarks are detected.

Final Dataset

After cleaning:

Total clean images: 2,188
Pose	Images
Pranamasana	550
Hasta Uttanasana	431
Padahastasana	286
Ashwa Sanchalanasana	161
Parvatasana	199
Ashtanga Namaskara	440
Bhujangasana	121
Total	2,188
Dataset Splitting

The dataset is divided into:

Training
Validation
Testing

A group-aware splitting strategy is used for frames extracted from custom videos.

This is important because consecutive frames from the same video can be visually very similar. Frames belonging to the same video are therefore kept in the same dataset split.

Final split
Split	Images
Training	1,532
Validation	312
Testing	344
Total	2,188
Video leakage verification

A dedicated verification script checks whether frames belonging to the same custom video appear in multiple splits.

The final dataset split currently passes this check:

NO VIDEO GROUP LEAKAGE FOUND
Deep Learning Model

The project uses MobileNetV2 with transfer learning.

MobileNetV2 is initialized using ImageNet pretrained weights and adapted for the seven Surya Namaskara classes.

The planned architecture is:

MobileNetV2
(ImageNet pretrained)
        |
        v
Global Average Pooling
        |
        v
Dropout
        |
        v
Dense Layer
        |
        v
7-Class Softmax Output

Training will use a two-stage approach:

Stage 1 — Transfer Learning

The pretrained MobileNetV2 feature extractor is initially frozen while the new classification layers are trained.

Stage 2 — Fine Tuning

Selected deeper MobileNetV2 layers are unfrozen and trained with a small learning rate to adapt the model to the Surya Namaskara dataset.

Model Evaluation

The trained model will be evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion matrix
Per-class performance

The test dataset is kept separate from training and validation data.

Real-Time Inference

After training, the model will be integrated into the real-time computer vision pipeline.

Webcam
   |
   v
OpenCV Frame
   |
   v
MobileNetV2
   |
   v
Pose Classification
   |
   v
MediaPipe Pose
   |
   v
Landmark Analysis
   |
   v
Posture Accuracy
   |
   v
Feedback

The system will provide feedback based on the detected posture and body alignment.

Project Structure
AI_SURYANAMASKAR_MODEL/
│
├── README.md
├── .gitignore
├── requirements.txt
├── LICENSE
│
├── preprocessing/
│   ├── config.py
│   ├── blur_detector.py
│   ├── duplicate_detector.py
│   ├── mediapipe_validator.py
│   ├── report_generator.py
│   ├── cleaner.py
│   ├── test_blur.py
│   ├── test_duplicate.py
│   ├── extract_new_videos.py
│   ├── analyze_duplicates.py
│   ├── create_clean_dataset.py
│   ├── split_dataset.py
│   └── verify_split.py
│
├── training/
│
├── inference/
│
├── models/
│
├── reports/
│   └── dataset_report.csv
│
├── suryanamaskar_dataset/   # Local dataset, not committed
├── clean_dataset/           # Local dataset, not committed
├── dataset_split/           # Local dataset, not committed
└── videos/                  # Local videos, not committed
Installation

Clone the repository:

git clone <repository-url>
cd AI_SURYANAMASKAR_MODEL

Create and activate the required Python environment.

Install dependencies:

pip install -r requirements.txt
Dataset Setup

The datasets and custom videos are intentionally excluded from the Git repository because of their size.

The expected local directories are:

suryanamaskar_dataset/
clean_dataset/
dataset_split/
videos/

The preprocessing scripts can be used to reproduce the cleaning and dataset preparation pipeline.

Preprocessing

Run the preprocessing scripts from the project root.

Example:

python preprocessing/create_clean_dataset.py

Create the group-aware dataset split:

python preprocessing/split_dataset.py

Verify that no custom video appears across multiple splits:

python preprocessing/verify_split.py
Current Status
Completed
 Initial dataset collection
 Custom video frame extraction
 Dataset cleaning
 Duplicate analysis
 MediaPipe pose validation
 Clean dataset creation
 Group-aware train/validation/test splitting
 Video leakage verification
In Progress
 MobileNetV2 transfer learning
 Model fine-tuning
 Test-set evaluation
 Confusion matrix generation
 Real-time inference
 Posture accuracy evaluation
 Voice feedback integration
Future Improvements
Increase the diversity of training subjects and environments.
Add more Surya Namaskara poses.
Improve posture correctness estimation.
Improve robustness to different camera angles and lighting conditions.
Optimize the trained model for real-time inference.
Add temporal movement analysis across consecutive frames.
Deploy the system as a desktop or web application.
Disclaimer

This project is intended for educational and research purposes. The system provides computer-vision-based posture feedback and should not be considered a substitute for professional medical or yoga instruction.


### Important

I've deliberately **not claimed model accuracy yet**, because we haven't trained MobileNetV2. That's better for your GitHub/research credibility than putting a made-up accuracy number in the README.

Also, the README already reflects the **actual dataset numbers we established**:

```text
Raw:   2,233
Clean: 2,188
Train: 1,532
Val:     312
Test:    344