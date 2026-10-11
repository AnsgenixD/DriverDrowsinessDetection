# Real-Time Driver Drowsiness & Distraction Detection (Edge DMS)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![iOS 16+](https://img.shields.io/badge/iOS-16.0%2B%20%7C%20Swift%205.9-orange.svg)](https://developer.apple.com/swift/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-teal.svg)](https://ai.google.dev/edge/mediapipe)
[![Model Size](https://img.shields.io/badge/Model%20Size-%3C%201.5%20MB-brightgreen.svg)]()
[![Inference Speed](https://img.shields.io/badge/Inference-120%2B%20FPS-green.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An end-to-end, lightweight **Driver Monitoring System (DMS)** engineered for low-power edge devices (Apple Silicon Macs, Raspberry Pi 4 / Pi Zero 2W, and iOS iPhones). 

The system detects **drowsiness (microsleep)**, **yawning**, and **distraction (looking away from the road)** in real-time by combining 468-point facial mesh geometry with lightweight convolutional neural networks.

---

## Key Features

- **Multimodal Dual-Branch Architecture**: Fuses cropped visual ocular/perioral patches (112x112) with normalized 7D geometric scalar vectors ($EAR, MAR, \text{Yaw}, \text{Pitch}, \text{Roll}$).
- **Two Model Backbones**:
  - `Custom 2D CNN`: 4-layer lightweight convolutional feature extractor (**0.43 MB**).
  - `MobileNetV3-Small`: Depthwise separable convolutions with Squeeze-and-Excitation (SE) attention (**1.32 MB**).
- **Post-Training INT8 Quantization**: Quantized via PyTorch `qnnpack` / `fbgemm` for ~4x model compression and SIMD edge speedup.
- **Real-World Driving Ergonomics**:
  - **45° Off-Center Phone Mount Calibration**: 1-tap road baseline zeroing (`c` in Python, `"Zero Heading"` in iOS) so side/vent phone mounts don't trigger false alerts.
  - **Adaptive Relative Headroom**: Dynamic thresholding ($MAR_{\text{threshold}} = MAR_{\text{idle}} + 0.26$, $EAR_{\text{threshold}} = EAR_{\text{idle}} \times 0.58$) to prevent false alarms from perspective foreshortening.
  - **2.5s Mirror-Check Grace Period**: Gives drivers 75 frames (@ 30 FPS) of leeway to check side mirrors and speedometers with a silent status indicator before sounding the alarm.
- **Native iOS / Swift App**: Full SwiftUI app using AVFoundation, Apple Vision, and CoreML running on the Apple Neural Engine (ANE).
- **Zero Local Storage Overhead**: Full model training workflow engineered for free Google Colab cloud execution.

---

## System Pipeline Architecture

Every camera frame passes sequentially through five stages:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. Capture  │ ──> │ 2. Face Mesh │ ──> │  3. Geometry │ ──> │4. Classifier │ ──> │   5. Alert   │
│ Webcam / iOS │     │  468 Points  │     │ EAR/MAR/Pose │     │  Dual-Branch │     │ Screen / HUD │
│    30 FPS    │     │  MediaPipe   │     │   solvePnP   │     │ CNN / Mobile │     │ Audio/Haptic │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## Geometric Formulations

### 1. Eye Aspect Ratio (EAR)
Calculated from 6 periocular landmark coordinates per eye:
$$\text{EAR} = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2 \cdot \|p_1 - p_4\|_2}$$
- Drops sharply toward zero during eye closure and microsleep events.

### 2. Mouth Aspect Ratio (MAR)
Calculated across opposing lip contour points:
$$\text{MAR} = \frac{\|q_2 - q_8\|_2 + \|q_3 - q_7\|_2 + \|q_4 - q_6\|_2}{3 \cdot \|q_1 - q_5\|_2}$$
- Increases when mouth opening exceeds resting driving aperture.

### 3. Head Pose Estimation (solvePnP)
Maps 2D facial keypoints to a 3D generic anthropometric face model using OpenCV `cv2.solvePnP` to derive Euler angles:
- **Yaw ($\alpha$)**: Left / Right head rotation ($\pm 28^\circ$).
- **Pitch ($\beta$)**: Upward gaze / Downward instrument-cluster glance ($\pm 22^\circ$).

---

## Project Structure

```
DriverDrowsinessDetection/
├── config.py                 # Central configurations, thresholds, and landmark indices
├── main.py                   # Real-time desktop edge inference pipeline (OpenCV HUD)
├── benchmark.py              # Performance profiling (FPS, latency, RAM, response time)
├── export_coreml.py          # PyTorch -> Apple CoreML (.mlpackage) export script
├── requirements.txt          # Python dependencies
├── face_landmarker.task      # Precompiled MediaPipe 468-point face mesh model
├── core/
│   ├── alert_manager.py      # HUD visual rendering and non-blocking audio alerts
│   ├── fusion_classifier.py  # Dual-branch neural predictor + heuristic fallback
│   ├── geometry.py           # EAR, MAR, solvePnP pose, and 7D feature packing
│   ├── landmark_detector.py  # MediaPipe Face Mesh wrapper (Metal GPU / CPU)
│   └── visual_extractor.py   # 112x112 ocular & perioral composite patch extractor
├── models/
│   ├── custom_cnn.py         # 4-layer custom 2D CNN dual-branch architecture
│   ├── mobilenet_v3.py       # MobileNetV3-Small dual-branch architecture
│   └── quantization.py       # Post-training dynamic INT8 quantization utilities
├── training/
│   ├── dataset.py            # Multimodal balanced dataset generator
│   └── train.py              # Training pipeline with Cosine Annealing & AdamW
├── colab/
│   └── Driver_Drowsiness_Training_Colab.ipynb # Free cloud GPU training notebook
├── ios/
│   ├── DMSApp.swift          # SwiftUI App entry point
│   ├── ContentView.swift     # SwiftUI HUD preview, gauges, and sensitivity controls
│   ├── DMSViewModel.swift    # ObservableObject connecting AVFoundation & Vision
│   ├── DMSPredictor.swift    # Adaptive baseline tracker & relative thresholding
│   ├── CameraManager.swift   # AVFoundation front camera capture (iOS 16 & 17+)
│   ├── GeometryUtils.swift   # Swift isotropic pixel coordinate EAR & MAR math
│   ├── AlertService.swift    # System sound chimes & Taptic Engine haptics
│   ├── Info.plist            # Camera usage permission (NSCameraUsageDescription)
│   └── README_IOS.md         # Step-by-step Xcode deployment guide
└── weights/
    ├── best_dms_model.pth    # Trained PyTorch Float32 weights (~1.4 MB)
    ├── dms_quantized_int8.pth# Quantized INT8 weights (~1.3 MB)
    ├── DriverStateClassifier.mlpackage # Apple CoreML package (Custom CNN)
    └── MobileNetV3_DMS.mlpackage       # Apple CoreML package (MobileNetV3)
```

---

## Quick Start (Python / Desktop)

### 1. Installation
Clone the repository and install the lightweight requirements:
```bash
git clone https://github.com/AnsgenixD/DriverDrowsinessDetection
cd DriverDrowsinessDetection
pip install -r requirements.txt
```
## Recommended install (using `uv` with Python 3.12)

[`uv`](https://github.com/astral-sh/uv) is a fast Python package and environment manager. It creates an isolated virtual environment and installs dependencies much quicker than `pip` alone.
1. Install `uv` (if you don't have it yet)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Clone the repository and enter it

```bash
git clone https://github.com/AnsgenixD/DriverDrowsinessDetection
cd DriverDrowsinessDetection
```

3. Create a virtual environment with Python 3.12

```bash
uv venv --python 3.12
```

> `uv` will automatically download Python 3.12 if it isn't already installed on your system.

4. Activate the environment

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat
```

5. Install the dependencies

```bash
uv pip install -r requirements.txt
```

6. (Optional) Verify the environment

```bash
python --version   # should report Python 3.12.x
```

### 2. Run Real-Time Edge Detection
```bash
python3 main.py
```

**Hotkeys during live execution:**
- `q` or `ESC`: Quit application
- `c`: Calibrate forward road heading (zeros out angle for side phone mounts)
- `s`: Save snapshot photo
- `r`: Reset alert counters
- `m`: Toggle audio mute

### 3. Run with INT8 Quantized Model
```bash
python3 main.py --quantized
```

### 4. Run with MobileNetV3 Backbone
```bash
python3 main.py --model mobilenet_v3
```

### 5. Run Performance Benchmarks
```bash
python3 benchmark.py --frames 300
```

---

## Quick Start (iOS / iPhone)

1. Open **Xcode** on your Mac.
2. Create a new **iOS SwiftUI App**.
3. Drag the files from the `ios/` folder into your Xcode project.
4. Drag either `weights/MobileNetV3_DMS.mlpackage` or `weights/DriverStateClassifier.mlpackage` into Xcode.
5. In your project target's **Info** tab, add `Privacy - Camera Usage Description`.
6. Connect your iPhone and press **Run** (`Cmd + R`).

See [`ios/README_IOS.md`](ios/README_IOS.md) for the complete walkthrough.

---

## Performance Benchmarks

Measured on Apple Silicon hardware (macOS / Darwin arm64):

| Metric | Measured Value | Target Goal | Status |
| :--- | :--- | :--- | :--- |
| **Inference Speed (FPS)** | **129.87 FPS** | $\ge 15\text{ FPS}$ | **PASSED** |
| **Inference Latency** | **7.70 ms** | $< 66.7\text{ ms}$ | **PASSED** |
| **Memory Footprint (RAM)** | **10.12 MB** | $\le 80\text{ MB}$ | **PASSED** |
| **Alert Responsiveness** | **0.10 ms** | $\le 1.0\text{ second}$ | **PASSED** |
| **Model Size (INT8)** | **1.30 MB** | $< 5\text{ MB}$ | **PASSED** |
| **Low-Light Operation** | **161.06 FPS** | Real-time | **PASSED** |

---

## Training on Google Colab

To train models without consuming local computer storage:
1. Open [`colab/Driver_Drowsiness_Training_Colab.ipynb`](colab/Driver_Drowsiness_Training_Colab.ipynb) in [Google Colab](https://colab.research.google.com).
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Run all cells (`Cmd/Ctrl + F9`).
4. The notebook trains the dual-branch model, quantizes it to INT8, evaluates validation accuracy, and exports the final `< 1.5 MB` weight file directly to your downloads.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
