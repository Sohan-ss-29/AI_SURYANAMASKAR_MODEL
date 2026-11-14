
# AI-Based Surya Namaskara Correction System

An academic computer-vision project that combines **MobileNetV2 pose classification** and **MediaPipe Pose landmark analysis** as the foundation for a real-time Surya Namaskara posture-assistance system.

> **Current implementation:** a seven-class pose classifier with a held-out test accuracy of **82.56%**, plus separate real-time webcam classification and bilateral landmark/joint-angle analysis scripts. Pose-specific correctness rules, sequence tracking, scoring, and complete visual/voice correction are still under development.

## Table of contents

- [Project overview](#project-overview)
- [Supported poses](#supported-poses)
- [Current implementation status](#current-implementation-status)
- [System architecture](#system-architecture)
- [Dataset and preprocessing](#dataset-and-preprocessing)
- [Model and evaluation](#model-and-evaluation)
- [Repository structure](#repository-structure)
- [Environment setup](#environment-setup)
- [Run the project](#run-the-project)
- [Reproducing preprocessing, training, and evaluation](#reproducing-preprocessing-training-and-evaluation)
- [Limitations and next steps](#limitations-and-next-steps)
- [Evidence and documentation](#evidence-and-documentation)

## Project overview

Surya Namaskara is a sequence of yoga postures performed in a specific order. This project explores how computer vision can recognize selected postures and analyze body landmarks to support future posture-correction feedback.

The project uses two complementary components:

- **MobileNetV2:** classifies an input image into one of seven trained pose classes.
- **MediaPipe Pose:** detects body landmarks and supports bilateral joint-angle calculations.

These components currently run as separate real-time demonstrations. The full end-to-end correction system has not yet been completed.

## Supported poses

| Class | Pose |
|---:|---|
| 1 | Pranamasana |
| 2 | Hasta Uttanasana |
| 3 | Padahastasana |
| 4 | Ashwa Sanchalanasana |
| 5 | Parvatasana |
| 6 | Ashtanga Namaskara |
| 7 | Bhujangasana |

## Current implementation status

| Component | Status |
|---|---|
| Dataset cleaning and quality checks | Implemented |
| Duplicate analysis and exact-duplicate exclusion | Implemented |
| Group-aware split and video-group leakage check | Implemented |
| MobileNetV2 transfer learning and fine-tuning experiment | Completed |
| Held-out test evaluation | Completed |
| Real-time seven-class webcam classification | Implemented |
| MediaPipe 33-landmark tracking demo | Implemented |
| Bilateral elbow, knee, shoulder, and hip angle display | Implemented |
| Pose-specific correctness rules | Next stage |
| Sequence/FSM tracking and temporal stability | Planned |
| Posture scoring and complete visual/voice correction | Planned |

## System architecture

### Implemented foundations

```text
Webcam / Image
      |
      v
OpenCV frame preparation
      |
      v
MobileNetV2 classifier
      |
      v
One of 7 pose labels + confidence

Separate real-time analysis demo:
Webcam -> MediaPipe Pose -> 33 body landmarks
       -> visibility-aware bilateral joint angles
```

### Intended future integration

```text
Pose classification + body landmarks
                |
                v
       Pose-specific analysis
                |
                v
    Sequence / temporal validation
                |
                v
   Correction feedback and scoring
```

The future pipeline is a development goal, not a claim that all of these features are already implemented.

## Dataset and preprocessing

The local dataset combines selected Yoga-82 images with custom-recorded video frames. The raw images and videos are **not included in this repository**; users must have authorized local copies before reproducing the data pipeline.

### Dataset counts

| Stage | Images |
|---|---:|
| Raw dataset | 2,233 |
| Blurry images excluded | 17 |
| Images excluded for insufficient pose information | 15 |
| Exact duplicates excluded | 13 |
| Final clean dataset | 2,188 |

### Final clean class distribution

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

### Preprocessing approach

The preprocessing workflow includes image readability checks, blur review, perceptual-hash duplicate analysis, MediaPipe pose validation for selected classes, creation of a clean dataset, and a group-aware train/validation/test split. Review samples are kept separately rather than deleting the raw source dataset.

The duplicate analysis flagged 316 candidates. Only the 13 images with perceptual-hash distance 0 were excluded as exact duplicates; near-duplicates were retained to preserve legitimate visual variation.

### Dataset split and leakage check

| Split | Images | Approx. share |
|---|---:|---:|
| Training | 1,532 | 70.0% |
| Validation | 312 | 14.3% |
| Test | 344 | 15.7% |
| **Total** | **2,188** | **100%** |

Frames from each identified custom video were assigned as a group to one split. The verification script checked 11 custom-video groups and reported **no video-group leakage**. This check applies to the identified custom-video groups; it does not claim that every possible source of dataset bias has been eliminated.

## Model and evaluation

### Model configuration

| Setting | Value |
|---|---|
| Architecture | MobileNetV2 |
| Pretrained weights | ImageNet |
| Input | 224 × 224 × 3 |
| Output classes | 7 |
| Total parameters | 2,266,951 |
| Framework | TensorFlow / Keras |
| TensorFlow version used for training | 2.21.0 |

The training experiment used two stages: first, training a classification head while the pretrained backbone was frozen; then, partially unfreezing the backbone for fine-tuning. The best validation accuracy was **89.42%** during Stage 1. Fine-tuning did not exceed that validation result, so the best Stage 1 checkpoint was retained.

### Held-out test results

**Test accuracy: 82.56% on 344 test images.**

| Pose | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Pranamasana | 99.14% | 100.00% | 99.57% |
| Hasta Uttanasana | 77.78% | 87.50% | 82.35% |
| Padahastasana | 79.49% | 72.09% | 75.61% |
| Ashwa Sanchalanasana | 55.26% | 87.50% | 67.74% |
| Parvatasana | 72.97% | 90.00% | 80.60% |
| Ashtanga Namaskara | 100.00% | 56.06% | 71.84% |
| Bhujangasana | 47.83% | 61.11% | 53.66% |

| Aggregate metric | Result |
|---|---:|
| Accuracy | 82.56% |
| Macro precision | 76.07% |
| Macro recall | 79.18% |
| Macro F1-score | 75.91% |
| Weighted precision | 85.84% |
| Weighted recall | 82.56% |
| Weighted F1-score | 82.57% |

These results are a baseline, not a claim of production-level posture correction. Performance varies by pose, and the seven-class classifier does not currently include a dedicated “not a supported pose” class. A high softmax score alone does not guarantee that an input is one of the supported yoga postures.

## Repository structure

```text
AI_SURYANAMASKAR_MODEL/
├── README.md
├── .gitignore
├── requirements.txt
├── Output_Documentation.md
├── duplicate_results.txt
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
├── training/
│   ├── train_mobilenetv2.py
│   ├── test_saved_model.py
│   └── evaluate_model.py
├── inference/
│   ├── realtime_classifier.py
│   ├── pose_landmarks.py
│   └── posture_utils.py
├── models/
│   └── class_names.json
├── output_screenshots/
└── reports/
    └── dataset_report.csv
```

Local datasets, custom videos, generated splits, review images, and trained model checkpoints are excluded from Git to keep the repository manageable.

## Environment setup

The development workflow uses **separate Conda environments** for TensorFlow training/inference and MediaPipe landmark analysis. Keeping them separate avoids forcing both stacks into one environment.

### 1. Clone the repository

```bash
git clone https://github.com/Sohan-ss-29/AI_SURYANAMASKAR_MODEL.git
cd AI_SURYANAMASKAR_MODEL
```

### 2. TensorFlow environment — classifier

Use Python 3.11 and install the TensorFlow-side packages:

```bash
conda create -n yoga python=3.11
conda activate yoga
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This is the environment for MobileNetV2 training, evaluation, and real-time classifier inference. Native Windows TensorFlow may run on CPU depending on the machine and TensorFlow build.

### 3. MediaPipe environment — landmark analysis and preprocessing

Open a separate terminal:

```bash
conda create -n yoga_clean python=3.11
conda activate yoga_clean
python -m pip install --upgrade pip
python -m pip install -r requirements-vision.txt
```

This environment is intended for the MediaPipe Pose demo and MediaPipe-dependent preprocessing tools. The `ImageHash` dependency is included because duplicate-analysis and cleaning scripts import `imagehash`.

### 4. Dataset and model files

The repository does not contain the raw dataset, custom videos, or the trained `.keras` checkpoint. Before running preprocessing or evaluation, place the locally available data in the directory names expected by the scripts. To run inference, obtain the trained checkpoint from your own training artifacts or retrain using the dataset and scripts described below.

Expected local data directories include:

```text
suryanamaskar_dataset/
videos/
clean_dataset/
dataset_split/
review/
```

Do not upload private recordings or large dataset/model files to Git unless you have the necessary rights and a deliberate distribution plan.

## Run the project

Run commands from the repository root unless stated otherwise.

### Real-time MobileNetV2 classifier

Activate the `yoga` environment and ensure the model checkpoint exists at:

```text
models/suryanamaskar_mobilenetv2_best.keras
```

Then run:

```bash
python inference/realtime_classifier.py
```

The webcam window displays a predicted pose label and model confidence. Press `q` to exit.

### MediaPipe landmark and joint-angle demo

Activate the `yoga_clean` environment:

```bash
python inference/pose_landmarks.py
```

The webcam window displays the detected skeleton and visibility-aware left/right elbow, knee, shoulder, and hip angles. Angles may display as `NA` when the required landmarks are not sufficiently visible. Press `q` to exit.

## Reproducing preprocessing, training, and evaluation

These commands assume you have placed the local dataset in the expected folders. They can take time and will generate local files that are ignored by Git.

### Create the clean dataset

Activate the environment containing the preprocessing dependencies:

```bash
python preprocessing/create_clean_dataset.py
```

### Create the group-aware split

```bash
python preprocessing/split_dataset.py
```

### Verify video-group leakage

```bash
python preprocessing/verify_split.py
```

The existing run reported no video-group leakage across the 11 identified custom-video groups. Re-run this check after changing the dataset or split.

### Train the model

Activate the `yoga` environment:

```bash
python training/train_mobilenetv2.py
```

### Evaluate the saved model

Ensure the trained checkpoint and test split are present, then run:

```bash
python training/evaluate_model.py
```

The documented baseline result is 82.56% accuracy on 344 held-out test images. Results may differ if the dataset, split, dependencies, or training configuration changes.

## Limitations and next steps

- Add pose-specific correctness rules based on body geometry and joint angles.
- Connect the classifier and MediaPipe analysis into a single application.
- Add temporal stability and sequence/FSM validation.
- Develop and evaluate posture scoring and corrective feedback.
- Investigate class-level errors, especially Bhujangasana and visually similar poses.
- Consider a supported-pose/unknown-input rejection strategy.
- Re-evaluate the integrated system with a clearly documented protocol and user study.

## Evidence and documentation

- [Project source code](https://github.com/Sohan-ss-29/AI_SURYANAMASKAR_MODEL)
- [Output documentation](Output_Documentation.md)
- [Training and evaluation screenshots](output_screenshots/)
- [Dataset report](reports/dataset_report.csv)

## Disclaimer

This repository is an academic prototype for computer-vision research and learning. The current classifier recognizes visual categories; it does not yet provide validated medical or safety advice. Users should practise yoga within their ability and seek qualified instruction when needed.
