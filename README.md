# AI-Based Surya Namaskara Correction System

An AI-powered computer vision system for recognizing and evaluating **Surya Namaskara yoga postures** using deep learning, pose estimation, and real-time feedback.

The project combines **MobileNetV2-based image classification** with **MediaPipe Pose landmark analysis** to recognize yoga poses and eventually provide posture correction, sequence tracking, and visual/voice feedback.

---

## 📌 Project Overview

Surya Namaskara is a sequence of yoga postures that requires correct body alignment and controlled movement. Incorrect posture during the sequence can reduce its effectiveness and may increase the risk of strain.

This project aims to develop an AI-based system capable of:

- Recognizing individual Surya Namaskara postures
- Classifying the detected posture using a deep learning model
- Analyzing body landmarks using MediaPipe Pose
- Evaluating posture correctness
- Tracking progression through the Surya Namaskara sequence
- Providing real-time visual feedback
- Providing voice-based correction and guidance

The system combines:

> **Deep Learning Image Classification + Pose Estimation + Posture Analysis + Real-Time Feedback**

---

## 🧘 Surya Namaskara Poses

The current model recognizes **7 major Surya Namaskara postures**.

| Class | Pose |
|:---:|---|
| 1 | Pranamasana |
| 2 | Hasta Uttanasana |
| 3 | Padahastasana |
| 4 | Ashwa Sanchalanasana |
| 5 | Parvatasana |
| 6 | Ashtanga Namaskara |
| 7 | Bhujangasana |

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **MobileNetV2**
- **OpenCV**
- **MediaPipe Pose**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **Pillow**
- **ImageHash**
- **Git & GitHub**

---

## 🏗️ System Architecture

```text
Input Image / Video
        │
        ▼
Image Preprocessing
        │
        ▼
MobileNetV2
Transfer Learning Model
        │
        ▼
7-Class Pose Classification
        │
        ▼
Detected Yoga Pose
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
MediaPipe Pose          Posture Analysis
        │                      │
        ▼                      ▼
Body Landmarks         Alignment / Form
        │                      │
        └──────────┬───────────┘
                   │
                   ▼
          Accuracy Evaluation
                   │
                   ▼
          Visual / Voice Feedback
```

---

# 📊 Dataset

The project uses a combination of **publicly available yoga images** and **custom-recorded video data**.

### Dataset Sources

- Selected yoga posture images from publicly available datasets
- Custom videos recorded to increase the number and diversity of examples for selected poses

Frames extracted from custom videos were prepared for compatibility with the MobileNetV2 input pipeline.

### Dataset Statistics

The original dataset contained:

```text
2,233 raw images
```

After preprocessing and cleaning:

```text
2,188 clean images
```

Therefore, a total of **45 images** were excluded from the final clean dataset.

---

## 🧹 Dataset Preprocessing

A preprocessing pipeline was developed to improve dataset quality before model training.

```text
Raw Dataset
     │
     ▼
Readable / Corrupt Image Check
     │
     ▼
Blur Detection
     │
     ▼
Duplicate Detection
     │
     ▼
MediaPipe Pose Validation
     │
     ▼
Manual / Review Analysis
     │
     ▼
Clean Dataset
     │
     ▼
Group-Aware Dataset Split
```

### Cleaning Pipeline

The preprocessing system checks for:

- Blurry images
- Duplicate images
- Images where a valid human pose cannot be detected
- Invalid or unreadable image files

The original dataset is **never modified or deleted** during the cleaning process.

Suspicious samples are copied into review directories for inspection.

---

## 🔍 Blur Detection

Blur detection is applied selectively to poses where image sharpness needs automated validation.

After inspection of the dataset, blur checking was intentionally disabled for certain poses where the available images were considered usable despite characteristics that could trigger conventional blur-detection thresholds.

Final blurry images excluded:

```text
17 images
```

---

## 🧍 MediaPipe Pose Validation

MediaPipe Pose is used during preprocessing to detect whether sufficient human body information is present in an image.

The validation logic is intentionally designed to be **permissive rather than requiring near-perfect landmark visibility**, because valid yoga images may contain:

- Partially hidden limbs
- Unusual body orientations
- Floor-based poses
- Side views
- Difficult joint visibility

Final no-pose images excluded:

```text
15 images
```

---

## ♻️ Duplicate Detection

Perceptual hashing is used to analyze visually similar images.

The initial duplicate detector flagged:

```text
316 potential duplicates
```

These were further analyzed using perceptual-hash distance.

Distribution of the flagged images:

| pHash Distance | Images |
|:---:|---:|
| 0 | 13 |
| 2 | 127 |
| 4 | 176 |

Only **distance = 0** images were treated as exact duplicates for final removal.

Near-duplicate frames were retained because they may contain meaningful differences in:

- Body alignment
- Limb position
- Camera perspective
- Movement
- Lighting
- Posture transition

Final exact duplicates excluded:

```text
13 images
```

---

## ✅ Final Clean Dataset

Cleaning summary:

| Category | Images |
|---|---:|
| Raw images | 2,233 |
| Blurry removed | 17 |
| No-pose removed | 15 |
| Exact duplicates removed | 13 |
| **Final clean images** | **2,188** |

### Class Distribution

| Pose | Images |
|---|---:|
| Pranamasana | 550 |
| Hasta Uttanasana | 431 |
| Padahastasana | 286 |
| Ashwa Sanchalanasana | 161 |
| Parvatasana | 199 |
| Ashtanga Namaskara | 440 |
| Bhujangasana | 121 |
| **Total** | **2,188** |

---

# 🔀 Dataset Splitting

The clean dataset is divided into:

- **Training set — 70%**
- **Validation set — approximately 15%**
- **Test set — approximately 15%**

A **strict global group-aware splitting strategy** is used for frames extracted from custom videos.

This is important because consecutive frames originating from the same video can be extremely similar.

Allowing frames from the same video to appear in both training and testing data could produce **data leakage** and artificially inflate model performance.

Therefore:

> Frames belonging to the same source video are assigned to only one dataset split.

### Final Dataset Split

| Split | Images |
|---|---:|
| Training | 1,532 |
| Validation | 312 |
| Testing | 344 |
| **Total** | **2,188** |

---

## 🔒 Video Leakage Verification

A dedicated verification script checks whether any custom-video group appears across multiple dataset splits.

The final split successfully passes the leakage check:

```text
NO VIDEO GROUP LEAKAGE FOUND
```

A total of **11 custom video groups** were verified.

This ensures that the held-out test dataset provides a more realistic measurement of model generalization.

---

# 🧠 Deep Learning Model

The project currently uses **MobileNetV2 with transfer learning**.

MobileNetV2 is initialized using **ImageNet pretrained weights** and adapted to classify the seven Surya Namaskara poses.

### Model Architecture

```text
Input Image
224 × 224 × 3
      │
      ▼
Data Augmentation
      │
      ▼
MobileNetV2 Preprocessing
      │
      ▼
MobileNetV2
(ImageNet Pretrained)
      │
      ▼
Global Average Pooling
      │
      ▼
Dropout
      │
      ▼
Classification Head
      │
      ▼
7-Class Softmax Output
```

The trained model contains approximately:

```text
2.27 million parameters
```

---

## 🚀 Training Strategy

Training is performed using a **two-stage transfer-learning strategy**.

### Stage 1 — Transfer Learning

The pretrained MobileNetV2 feature extractor is initially frozen.

Only the newly added classification layers are trained on the Surya Namaskara dataset.

This allows the model to reuse general visual features learned from ImageNet.

### Stage 2 — Fine-Tuning

Selected deeper MobileNetV2 layers are unfrozen.

Training then continues using a smaller learning rate so that the pretrained features can adapt more specifically to Surya Namaskara postures without aggressively modifying previously learned representations.

---

# 📈 Model Evaluation

The trained model was evaluated using the completely held-out test dataset containing:

```text
344 images
```

### Current Test Performance

| Metric | Result |
|---|---:|
| **Test Accuracy** | **82.56%** |
| Weighted Precision | 85.84% |
| Weighted Recall | 82.56% |
| Weighted F1-score | 82.57% |
| Macro Precision | 76.07% |
| Macro Recall | 79.18% |
| Macro F1-score | 75.91% |

### Per-Class Performance

| Pose | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Pranamasana | 99.14% | 100.00% | 99.57% |
| Hasta Uttanasana | 77.78% | 87.50% | 82.35% |
| Padahastasana | 79.49% | 72.09% | 75.61% |
| Ashwa Sanchalanasana | 55.26% | 87.50% | 67.74% |
| Parvatasana | 72.97% | 90.00% | 80.60% |
| Ashtanga Namaskara | 100.00% | 56.06% | 71.84% |
| Bhujangasana | 47.83% | 61.11% | 53.66% |

The current model therefore establishes a **baseline test accuracy of 82.56%**.

Further error analysis and model optimization are planned, particularly for classes with lower precision or recall.

---

## 🔬 Planned Error Analysis

Before finalizing the classifier, further analysis will include:

- Confusion matrix visualization
- Inspection of incorrectly classified test images
- Analysis of visually similar poses
- Investigation of class imbalance
- Evaluation of additional augmentation strategies
- Potential additional fine-tuning

The goal is to improve generalization while preserving the strict no-video-leakage evaluation methodology.

---

# 🎥 Real-Time Inference

The trained classifier will later be integrated into a real-time computer vision pipeline.

```text
Webcam
   │
   ▼
OpenCV Frame
   │
   ▼
MobileNetV2
   │
   ▼
Pose Classification
   │
   ▼
MediaPipe Pose
   │
   ▼
Body Landmark Analysis
   │
   ▼
Posture Accuracy
   │
   ▼
Sequence Tracking
   │
   ▼
Visual / Voice Feedback
```

The planned system will combine **pose classification** with **landmark-based posture analysis** rather than relying only on image classification.

---

# 🔄 Planned Surya Namaskara Pipeline

```text
MobileNetV2 Classification
          │
          ▼
Real-Time Webcam
          │
          ▼
Pose Recognition
          │
          ▼
Surya Namaskara Sequence / FSM
          │
          ▼
MediaPipe Landmark Analysis
          │
          ▼
Posture Accuracy / Correction
          │
          ▼
Visual + Voice Feedback
```

A finite-state-machine-style sequence tracker is planned to ensure that poses are performed in the correct Surya Namaskara order.

---

# 📁 Project Structure

```text
AI_SURYANAMASKAR_MODEL/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── preprocessing/
│   ├── analyze_duplicates.py
│   ├── blur_detector.py
│   ├── cleaner.py
│   ├── config.py
│   ├── create_clean_dataset.py
│   ├── duplicate_detector.py
│   ├── extract_new_videos.py
│   ├── mediapipe_validator.py
│   ├── report_generator.py
│   ├── split_dataset.py
│   ├── test_blur.py
│   ├── test_duplicate.py
│   ├── test_mediapipe.py
│   └── verify_split.py
│
├── training/
│   ├── train_mobilenetv2.py
│   ├── test_saved_model.py
│   └── evaluate_model.py
│
├── inference/
│
├── models/
│   └── *.keras                 # Local trained models
│
├── reports/
│   └── dataset_report.csv
│
├── suryanamaskar_dataset/     # Local raw dataset
├── clean_dataset/             # Local cleaned dataset
├── dataset_split/             # Local train/val/test data
└── videos/                    # Local custom videos
```

Large datasets, videos, generated splits, and trained model files can be excluded from the repository through `.gitignore`.

---

# ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sohan-ss-29/AI_SURYANAMASKAR_MODEL.git
cd AI_SURYANAMASKAR_MODEL
```

### 2. Create a Python Environment

Using Conda:

```bash
conda create -n yoga python=3.11
conda activate yoga
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📂 Dataset Setup

The image datasets and custom videos are intentionally excluded from the Git repository because of their size.

The expected local directories are:

```text
suryanamaskar_dataset/
clean_dataset/
dataset_split/
videos/
```

The preprocessing scripts can then be used to reproduce the cleaning and dataset preparation pipeline.

---

# 🧹 Running Preprocessing

Run commands from the project root.

### Create the Clean Dataset

```bash
python preprocessing/create_clean_dataset.py
```

### Create the Group-Aware Split

```bash
python preprocessing/split_dataset.py
```

### Verify Video Leakage

```bash
python preprocessing/verify_split.py
```

Expected result:

```text
NO VIDEO GROUP LEAKAGE FOUND
```

---

# 🏋️ Training

Train the MobileNetV2 classifier using:

```bash
python training/train_mobilenetv2.py
```

The training pipeline uses transfer learning followed by fine-tuning.

---

# 🧪 Model Evaluation

Evaluate the saved model against the held-out test dataset using:

```bash
python training/evaluate_model.py
```

Current baseline:

```text
Test images   : 344
Test accuracy : 82.56%
```

---

# 📌 Current Project Status

### ✅ Completed

- [x] Initial dataset collection
- [x] Custom video frame extraction
- [x] Dataset cleaning pipeline
- [x] Blur analysis
- [x] Duplicate analysis
- [x] Exact duplicate removal
- [x] MediaPipe pose validation
- [x] Clean dataset generation
- [x] Strict global group-aware dataset splitting
- [x] Video leakage verification
- [x] MobileNetV2 transfer learning
- [x] MobileNetV2 fine-tuning
- [x] Test-set evaluation
- [x] Baseline test accuracy established — **82.56%**

### 🔄 Next Steps

- [ ] Generate confusion matrix visualization
- [ ] Analyze misclassified test images
- [ ] Improve weaker pose classes
- [ ] Finalize optimized classification model
- [ ] Real-time webcam inference
- [ ] Surya Namaskara sequence tracking
- [ ] MediaPipe-based posture correctness analysis
- [ ] Visual correction feedback
- [ ] Voice feedback integration

---

# 🔮 Future Improvements

Potential improvements include:

- Increasing subject diversity in the training dataset
- Increasing examples for underrepresented classes
- Improving classification of visually similar poses
- Adding more Surya Namaskara postures
- Improving robustness to different camera angles
- Improving robustness to lighting and background changes
- Combining classification confidence with MediaPipe landmark geometry
- Adding temporal analysis across consecutive video frames
- Implementing full Surya Namaskara sequence recognition
- Optimizing the trained model for real-time inference
- Deploying the system as a desktop, mobile, or web application

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes**.

The system provides computer-vision-based posture analysis and feedback and should not be considered a substitute for professional medical advice, physiotherapy, or qualified yoga instruction.

---

## 👨‍💻 Author

**Sohan Suhas**

GitHub: **Sohan-ss-29**

---

⭐ If you find this project useful or interesting, consider starring the repository.
