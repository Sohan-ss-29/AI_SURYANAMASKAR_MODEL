# AI-Based Surya Namaskara Correction System

An AI-powered computer vision system for recognizing and evaluating **Surya Namaskara yoga postures** using deep learning, pose estimation, and real-time feedback.

The project combines **MobileNetV2-based image classification** with **MediaPipe Pose landmark analysis** to recognize yoga poses and eventually provide posture correction, sequence tracking, accuracy scoring, and visual/voice feedback.

---

## 📌 Project Overview

Surya Namaskara is a sequence of yoga postures that requires correct body alignment and controlled movement. Incorrect posture during the sequence can reduce its effectiveness and may increase the risk of physical strain.

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

# 🏗️ System Architecture

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

- Selected images from the Yoga-82 dataset
- Custom videos recorded to increase the number and diversity of examples for selected poses

Custom video frames were extracted and resized to **224 × 224 pixels** for compatibility with the MobileNetV2 input pipeline.

---

## Dataset Statistics

### Raw Dataset

**2,233 images**

### Clean Dataset

**2,188 images**

### Removed Images

| Category | Images Removed |
|---|---:|
| Blurry images | 17 |
| No-pose images | 15 |
| Exact duplicate images | 13 |
| **Total removed** | **45** |

Therefore:

**2,233 − 45 = 2,188 clean images**

---

## Final Class Distribution

| Class | Pose | Images |
|:---:|---|---:|
| 1 | Pranamasana | 550 |
| 2 | Hasta Uttanasana | 431 |
| 3 | Padahastasana | 286 |
| 4 | Ashwa Sanchalanasana | 161 |
| 5 | Parvatasana | 199 |
| 6 | Ashtanga Namaskara | 440 |
| 7 | Bhujangasana | 121 |
| | **Total** | **2,188** |

---

# 🧹 Dataset Preprocessing

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

The preprocessing pipeline checks for:

- Blurry images
- Duplicate images
- Images where a valid human pose cannot be detected
- Invalid or unreadable image files

The original dataset is preserved during preprocessing. Suspicious samples are placed into review directories rather than directly deleting the original data.

---

# 🔍 Blur Detection

Blur detection was used to identify images with insufficient visual sharpness.

After the preprocessing and review process:

**17 blurry images were excluded from the clean dataset.**

---

# 🧍 MediaPipe Pose Validation

MediaPipe Pose was used as a validation step to determine whether sufficient human body landmark information was available in an image.

The validation logic was designed to be permissive because valid yoga images may contain:

- Partially hidden limbs
- Unusual body orientations
- Floor-based poses
- Side views
- Difficult joint visibility

After validation:

**15 images were excluded because sufficient pose information could not be detected.**

---

# ♻️ Duplicate Analysis

A perceptual-hash-based duplicate analysis was performed on the dataset.

## Potential Duplicates

**316 images** were flagged for duplicate analysis.

### pHash Distance Distribution

| pHash Distance | Number of Images | Interpretation |
|:---:|---:|---|
| 0 | 13 | Exact duplicates |
| 2 | 127 | Near duplicates |
| 4 | 176 | Near duplicates |

### Duplicate Handling Strategy

Only images with a pHash distance of **0** were treated as exact duplicates and removed.

Near-duplicate images were retained because they may represent meaningful variations in:

- Body position
- Movement
- Camera position
- Lighting
- Background
- Posture transitions

### Final Duplicate Removal

**13 exact duplicate images were removed.**

---

# 🔀 Dataset Splitting

The final clean dataset contains:

**2,188 images**

The dataset was divided into:

- Training
- Validation
- Testing

| Split | Images |
|---|---:|
| Training | 1,532 |
| Validation | 312 |
| Testing | 344 |
| **Total** | **2,188** |

---

## Group-Aware Dataset Splitting

Custom-video frames were grouped according to their source video.

All frames originating from the same video were assigned to **only one dataset split**.

This prevents visually similar consecutive frames from the same video from appearing in both training and testing datasets.

This strategy was implemented to reduce **data leakage** and provide a more reliable estimate of model generalization.

---

# 🔒 Video Leakage Verification

A dedicated verification script was used to check whether frames from the same custom video appeared in multiple dataset splits.

### Verification Output

```text
======================================================================
DATASET LEAKAGE CHECK
======================================================================

✅ NO VIDEO GROUP LEAKAGE FOUND

Video groups checked: 11
======================================================================
```

### Result

**11 custom video groups were checked.**

**No video group appeared across multiple dataset splits.**

Therefore, the final dataset split passed the implemented video-group leakage verification.

---

# 🧠 Deep Learning Model

The project uses **MobileNetV2 with transfer learning**.

MobileNetV2 is initialized using **ImageNet pretrained weights** and adapted to classify the seven Surya Namaskara poses.

---

## Model Configuration

| Parameter | Configuration |
|---|---|
| Model | MobileNetV2 |
| Pretrained weights | ImageNet |
| Input size | 224 × 224 × 3 |
| Number of classes | 7 |
| Total parameters | 2,266,951 |
| Framework | TensorFlow / Keras |
| TensorFlow version | 2.21.0 |

---

## Model Architecture

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

---

# 🏋️ Training Strategy

Training was performed using a **two-stage transfer-learning approach**.

## Stage 1 — Transfer Learning

The pretrained MobileNetV2 feature extractor was initially frozen.

The newly added classification layers were trained for the seven Surya Namaskara classes.

### Best Stage 1 Validation Accuracy

**89.42%**

---

## Stage 2 — Fine-Tuning

Selected deeper MobileNetV2 layers were subsequently unfrozen and trained using a smaller learning rate.

The fine-tuning stage was completed successfully.

However, fine-tuning did not improve upon the best Stage 1 validation accuracy.

### Experimental Conclusion

> Stage 1 transfer learning achieved the best validation accuracy of **89.42%**. Subsequent fine-tuning did not produce an improvement beyond this value.

This result is retained as an experimental observation.

---

# 📈 Model Evaluation

The trained model was evaluated using the completely held-out test dataset.

### Test Dataset

**344 images**

### Overall Test Accuracy

# **82.56%**

This represents the current baseline performance of the seven-class Surya Namaskara classifier on the held-out test dataset.

---

## Per-Class Performance

| Pose | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Pranamasana | 99.14% | 100.00% | 99.57% |
| Hasta Uttanasana | 77.78% | 87.50% | 82.35% |
| Padahastasana | 79.49% | 72.09% | 75.61% |
| Ashwa Sanchalanasana | 55.26% | 87.50% | 67.74% |
| Parvatasana | 72.97% | 90.00% | 80.60% |
| Ashtanga Namaskara | 100.00% | 56.06% | 71.84% |
| Bhujangasana | 47.83% | 61.11% | 53.66% |

---

## Aggregate Metrics

| Metric | Result |
|---|---:|
| **Test Accuracy** | **82.56%** |
| Macro Precision | 76.07% |
| Macro Recall | 79.18% |
| Macro F1-Score | 75.91% |
| Weighted Precision | 85.84% |
| Weighted Recall | 82.56% |
| Weighted F1-Score | 82.57% |

---

# 🔬 Current Model Observations

The model performs particularly well on **Pranamasana**:

- Precision: **99.14%**
- Recall: **100.00%**
- F1-score: **99.57%**

However, some classes require further investigation.

### Bhujangasana

- Precision: **47.83%**
- Recall: **61.11%**
- F1-score: **53.66%**

### Ashwa Sanchalanasana

- Precision: **55.26%**
- Recall: **87.50%**
- F1-score: **67.74%**

### Ashtanga Namaskara

- Precision: **100.00%**
- Recall: **56.06%**
- F1-score: **71.84%**

These results indicate that visually similar postures and class imbalance may require further analysis and model improvement.

---

# 🔬 Planned Error Analysis

The next model-analysis stage will include:

- Confusion matrix visualization
- Inspection of incorrectly classified test images
- Analysis of visually similar poses
- Investigation of class imbalance
- Evaluation of additional augmentation strategies
- Potential additional fine-tuning

The objective is to improve generalization while maintaining the strict no-video-leakage evaluation methodology.

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

The final system is intended to combine **image classification** with **landmark-based posture analysis** rather than relying only on image classification.

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
│   └── *.keras
│
├── reports/
│   └── dataset_report.csv
│
├── suryanamaskar_dataset/     # Local raw dataset, not committed
├── clean_dataset/             # Local cleaned dataset, not committed
├── dataset_split/             # Local train/val/test data, not committed
└── videos/                    # Local custom videos, not committed
```

Large datasets, videos, generated splits, and trained model files are excluded from Git using `.gitignore`.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Sohan-ss-29/AI_SURYANAMASKAR_MODEL.git
cd AI_SURYANAMASKAR_MODEL
```

## 2. Create the Python Environment

```bash
conda create -n yoga python=3.11
conda activate yoga
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📂 Dataset Setup

The raw datasets and custom videos are intentionally excluded from the Git repository because of their size.

The expected local directories are:

```text
suryanamaskar_dataset/
clean_dataset/
dataset_split/
videos/
```

The preprocessing scripts can be used to reproduce the cleaning and dataset preparation pipeline.

---

# 🧹 Preprocessing Commands

Run commands from the project root.

## Create the Clean Dataset

```bash
python preprocessing/create_clean_dataset.py
```

## Create the Group-Aware Dataset Split

```bash
python preprocessing/split_dataset.py
```

## Verify Video Leakage

```bash
python preprocessing/verify_split.py
```

Expected result:

```text
NO VIDEO GROUP LEAKAGE FOUND
```

---

# 🏋️ Model Training

Train the MobileNetV2 classifier using:

```bash
python training/train_mobilenetv2.py
```

The training pipeline uses:

1. Transfer learning
2. Classification-head training
3. Fine-tuning of selected MobileNetV2 layers

---

# 🧪 Model Evaluation

Evaluate the saved model using:

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

## ✅ Completed

- [x] Problem definition and planning
- [x] Dataset collection
- [x] Custom video collection
- [x] Video frame extraction
- [x] Dataset cleaning
- [x] Blur analysis
- [x] Duplicate analysis
- [x] Exact duplicate removal
- [x] MediaPipe pose validation
- [x] Clean dataset creation
- [x] Group-aware dataset splitting
- [x] Video leakage verification
- [x] MobileNetV2 implementation
- [x] Transfer learning
- [x] Fine-tuning experiment
- [x] Test-set evaluation
- [x] Baseline test accuracy established — **82.56%**

## 🔄 In Progress / Next

- [ ] Confusion matrix visualization
- [ ] Misclassification/error analysis
- [ ] Model improvement/iteration
- [ ] Real-time webcam inference
- [ ] MediaPipe posture correctness analysis
- [ ] Surya Namaskara sequence/FSM integration
- [ ] Posture accuracy scoring
- [ ] Visual correction feedback
- [ ] Voice feedback integration
- [ ] Full system integration
- [ ] Final system testing
- [ ] Final documentation
- [ ] Final presentation / PPT

---

# 📊 Overall Project Completion

## Current Estimated Completion

# **Approximately 65–70%**

This is an approximate project-progress estimate based on the current implementation status.

The machine-learning classification pipeline has reached a functional baseline:

```text
Dataset
   ↓
Cleaning
   ↓
Validation
   ↓
Leakage-Safe Splitting
   ↓
MobileNetV2 Training
   ↓
Fine-Tuning
   ↓
Held-Out Test Evaluation
   ↓
82.56% Test Accuracy
```

The remaining major work is primarily focused on integrating the trained classifier with the real-time posture-analysis system.

---

# 🗺️ Project Development Timeline

```text
Phase 1
Problem Definition & Planning
        │
        ▼
Phase 2
Dataset Collection
        │
        ▼
Phase 3
Custom Video Collection & Frame Extraction
        │
        ▼
Phase 4
Dataset Cleaning & Validation
        │
        ▼
Phase 5
Duplicate Analysis
        │
        ▼
Phase 6
Group-Aware Dataset Splitting
        │
        ▼
Phase 7
Video Leakage Verification
        │
        ▼
Phase 8
MobileNetV2 Transfer Learning
        │
        ▼
Phase 9
Fine-Tuning
        │
        ▼
Phase 10
Held-Out Test Evaluation
        │
        ▼
        ⭐ CURRENT STAGE
        │
        ▼
Phase 11
Confusion Matrix & Error Analysis
        │
        ▼
Phase 12
Model Improvement
        │
        ▼
Phase 13
Real-Time Webcam Integration
        │
        ▼
Phase 14
MediaPipe Posture Analysis
        │
        ▼
Phase 15
Sequence / FSM Integration
        │
        ▼
Phase 16
Visual & Voice Feedback
        │
        ▼
Phase 17
Final System Testing
        │
        ▼
Phase 18
Final Documentation & Presentation
```

---

# 📝 Important Technical Notes

## Model Checkpoint

The trained model was successfully saved as a Keras model checkpoint.

An initial model-loading issue occurred because the first training version stored MobileNetV2 preprocessing inside a Keras Lambda layer.

The existing trained checkpoint was successfully loaded by explicitly providing the MobileNetV2 `preprocess_input` function through Keras `custom_objects`.

For future training, preprocessing was changed to a serializable Keras `Rescaling` layer to avoid the same serialization issue.

## Current Model Status

The current trained model is considered:

**Baseline Model — v1**

It is not yet considered the final production model.

---

# 📚 Evaluation Evidence

The following artifacts and results have been generated during development:

- Dataset cleaning report
- Dataset statistics
- Duplicate analysis results
- Clean dataset statistics
- Group-aware dataset split
- Video leakage verification output
- MobileNetV2 training logs
- Saved model checkpoint
- Test-set classification report
- Confusion matrix data
- Preprocessing scripts
- Dataset split verification scripts
- Model evaluation script

These artifacts can be used as supporting evidence during project evaluations, final documentation, and presentation preparation.

---

# 🎯 Current Milestone

## Milestone Completed

A leakage-safe, cleaned seven-class Surya Namaskara dataset has been prepared and successfully used to train and evaluate a MobileNetV2 transfer-learning classifier.

### Current Baseline

**82.56% test accuracy on 344 held-out test images.**

---

# 🚀 Next Major Milestone

Integrate the trained classifier with:

1. Real-time webcam input
2. MediaPipe Pose landmark detection
3. Posture correctness analysis
4. Accuracy scoring
5. Surya Namaskara sequence tracking
6. Visual correction feedback
7. Voice feedback

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
- Implementing complete Surya Namaskara sequence recognition
- Optimizing the trained model for real-time inference
- Deploying the system as a desktop, mobile, or web application

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes**.

The system provides computer-vision-based posture analysis and feedback and should not be considered a substitute for professional medical advice, physiotherapy, or qualified yoga instruction.

---

# 👨‍💻 Author

**Sohan Suhas**

GitHub: **Sohan-ss-29**

---

⭐ If you find this project useful or interesting, consider starring the repository.
