# Real-Time Hand Gesture Recognition System

A real-time vision-based hand gesture recognition pipeline powered by **MediaPipe Tasks (HandLandmarker)**, **OpenCV**, and a temporal sequence **Random Forest Classifier** trained with **scikit-learn**.

The system captures hand landmarks from live webcam video, normalizes coordinates relative to the wrist, maintains a 10-frame sliding window buffer to capture both static poses and dynamic motion, and classifies gestures in real time with high accuracy (~99.55%).

---

## Architecture Overview

The system is organized into a modular four-stage pipeline:

```
[ Webcam Feed (OpenCV) ]
           │
           ▼
[ Hand Landmark Detection (MediaPipe HandLandmarker) ]
  • 21 3D landmarks (x, y, z) per detected hand
           │
           ▼
[ Feature Engineering & Normalization ]
  • Translation-invariance: (x - wrist_x, y - wrist_y, z - wrist_z)
  • Temporal sequence window: 10 consecutive frames buffer (deque)
  • Flattened feature vector: 10 frames × 21 landmarks × 3 coords = 630 features
           │
           ▼
[ Classification & Prediction (scikit-learn) ]
  • RandomForestClassifier (100 estimators, parallel execution)
  • Real-time HUD overlay on video stream
```

### Architectural Components

1. **Vision & Tracking Layer (`collectData.py`, `handOverLay.py`, `live_model.py`)**:
   - Integrates MediaPipe Tasks Vision API (`mp.tasks.vision.HandLandmarker`).
   - Uses the pretrained `hand_landmarker.task` model asset.
   - Extracts 21 3D hand landmarks at high FPS.
   - Supports handedness detection and skeletal overlay visualization.

2. **Feature Representation & Windowing**:
   - **Wrist-Relative Normalization**: Landmarks are translated relative to wrist landmark (`landmark[0]`), ensuring model predictions are invariant to the hand's location in the camera frame.
   - **Temporal Sliding Window (`WINDOW_SIZE = 10`)**: Employs a rolling deque of size 10 to capture motion over time, making it capable of distinguishing directional dynamic gestures (e.g., swipe left vs. swipe right).
   - Produces a 630-dimensional feature vector per sample.

3. **Model & Inference Layer (`train_model.py`, `live_model.py`)**:
   - Trained `RandomForestClassifier` stored in `gesture_model.pkl`.
   - Low latency real-time inference suitable for interactive applications and human-computer interaction (HCI).

---

## Capabilities & Supported Gestures

The system recognizes **10 distinct gesture classes**, including static poses and temporal dynamic gestures:

| Key | Gesture Label | Type | Description |
|:---:|:---|:---:|:---|
| `1` | `closed fist` | Static | Hand curled into a closed fist |
| `2` | `open palm` | Static | All fingers extended outward |
| `3` | `thumbs up` | Static | Thumb pointing upwards, fingers closed |
| `4` | `thumbs down` | Static | Thumb pointing downwards, fingers closed |
| `5` | `pointing left` | Static/Directional | Index finger extended leftward |
| `6` | `pointing right` | Static/Directional | Index finger extended rightward |
| `7` | `okay sign` | Static | Thumb and index forming circle, other fingers open |
| `8` | `swip left` | Dynamic | Hand moving in a leftward swipe motion |
| `9` | `swip right` | Dynamic | Hand moving in a rightward swipe motion |
| `Space` | `idle/normal` | Baseline | Relaxed/neutral hand state |

---

## Model Performance

The current model was trained on **13,478 sequence frames** with an 80/20 stratified train/test split:

- **Overall Accuracy**: **99.55%**
- **Test Set Size**: 2,696 samples

### Classification Report

```
                 precision    recall  f1-score   support

    closed fist       1.00      1.00      1.00       251
    idle/normal       1.00      0.99      0.99       360
      okay sign       1.00      1.00      1.00       256
      open palm       1.00      1.00      1.00       259
  pointing left       1.00      1.00      1.00       237
 pointing right       1.00      1.00      1.00       230
      swip left       0.99      0.98      0.99       280
     swip right       0.98      1.00      0.99       346
    thumbs down       0.99      0.99      0.99       199
      thumbs up       1.00      1.00      1.00       278

       accuracy                           1.00      2696
      macro avg       1.00      1.00      1.00      2696
   weighted avg       1.00      1.00      1.00      2696
```

---

## Repository Structure

```
.
├── collectData.py            # Interactive data collection tool for logging 10-frame windows
├── handOverLay.py            # Standalone visualizer for hand landmarks and handedness
├── live_model.py             # Real-time gesture prediction script with camera HUD
├── train_model.py            # Training pipeline: loads dataset, trains Random Forest, exports pkl
├── hand_landmarker.task      # MediaPipe HandLandmarker task bundle model asset
├── gesture_model.pkl         # Trained serialized Random Forest model (joblib)
├── gesture_dataset.csv       # Flattened 630-feature sequence dataset with labels
├── model training result.txt # Evaluation metrics and classification report log
├── requirements.txt          # System and Python package dependencies
└── README.md                 # Project documentation
```

---

## Setup & Installation

### 1. Prerequisites

Ensure system libraries for OpenCV and MediaPipe are installed (Linux / Ubuntu / Debian):

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
```

### 2. Virtual Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install opencv-python mediapipe scikit-learn pandas joblib
```

*(Or refer to `requirements.txt`)*

### 4. MediaPipe Task Model

Ensure `hand_landmarker.task` is in the root directory. If missing, download it from Google MediaPipe:
```bash
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## Usage Guide

### 1. Run Real-Time Live Inference

To launch the live webcam gesture classifier:

```bash
python3 live_model.py
```
- A window titled **"Predicted gesture"** will open.
- Blue circles will indicate tracked landmark points.
- The recognized gesture will be displayed at the top-left of the video stream.
- Press **`q`** to quit.

### 2. Inspect Hand Landmarks

To inspect raw MediaPipe landmark tracking with skeletal lines and handedness labels:

```bash
python3 handOverLay.py
```

### 3. Collect Custom Dataset Samples

To record new gesture sequences or add classes:

```bash
python3 collectData.py
```
- Ensure your hand is visible and moving/posing for the intended gesture.
- Press keys **`1`** through **`9`** or **`Space`** while performing the gesture to append a 10-frame sequence to `gesture_dataset.csv`.
- Press **`q`** to exit.

### 4. Train the Model

To retrain the Random Forest model on the updated dataset:

```bash
python3 train_model.py
```
- Evaluates the model on a test set (stratified split).
- Prints accuracy and full classification report.
- Saves the trained model to `gesture_model.pkl`.
