# Driver Monitoring System (DMS) — Setup & Storage Guide

## 1. Storage-Optimized Architecture (Google Colab vs. Local Laptop)

Because your local device has limited storage, this project is engineered with a **zero-storage-waste** edge design:

| Component | Where It Runs | Disk Space Used | Why |
|---|---|---|---|
| **Large Datasets (MRL Eye ~85k imgs, YawDD)** | **Google Colab Cloud** | **0 MB on your laptop** | Datasets are processed in the cloud using Google Colab's free virtual drive. |
| **Model Training & Quantization** | **Google Colab Cloud** | **0 MB on your laptop** | Trained on Colab's free Nvidia T4 GPU without heating up your laptop. |
| **Trained Model Weights (`.pth`)** | **Local Laptop** | **< 1.5 MB** | Only the tiny final model file is downloaded to your laptop. |
| **MediaPipe Face Mesh Task** | **Local Laptop** | **3.6 MB** | Pre-downloaded 468-point landmark detector. |
| **Real-Time Webcam DMS (`main.py`)** | **Local Laptop** | **~10 MB RAM** | Runs at 30+ FPS directly on your webcam / Raspberry Pi. |

---

## 2. What Do You Need to Download?

### A. Local Machine (Your Laptop / Mac)
**Good news:** Your Python environment already has the core libraries installed!
If you ever run this on another device (such as a Raspberry Pi or another laptop), you only need:
```bash
pip install opencv-python mediapipe torch numpy
```

### B. The Trained Model Weights
You have two instant options:
1. **Option 1 (Pre-packaged Local Model):** A champion model has already been trained and quantized into `weights/best_dms_model.pth` (454 KB) and `weights/dms_quantized_int8.pth` (416 KB). You can run live detection **immediately** without waiting for any download!
2. **Option 2 (Train your own on Google Colab):**
   - Upload [`colab/Driver_Drowsiness_Training_Colab.ipynb`](colab/Driver_Drowsiness_Training_Colab.ipynb) to [Google Colab](https://colab.research.google.com).
   - Click `Runtime` -> `Change runtime type` -> select `T4 GPU`.
   - Run all cells (`Ctrl + F9` or `Cmd + F9`).
   - The final cell automatically downloads `best_dms_model.pth` (< 1.5 MB) to your `Downloads` folder.
   - Simply move that file into your project's `weights/` directory:
     ```bash
     mv ~/Downloads/best_dms_model.pth ./weights/
     ```

---

## 3. How to Run the System

### 1. Launch Live Webcam Monitoring
```bash
python3 main.py
```
- Opens your camera feed with the complete **Real-Time Heads-Up Display (HUD)**.
- Tracks:
  - **468-point facial mesh** (yellow points for eyes, magenta for lips).
  - **EAR (Eye Aspect Ratio)**: drops towards zero on closed eyes/microsleep.
  - **MAR (Mouth Aspect Ratio)**: spikes upward on yawning.
  - **Head Pose (solvePnP)**: displays Yaw, Pitch, Roll angles in real-time.
  - **Driver States**: `FOCUSED` (Green), `DROWSY` (Red), `YAWNING` (Yellow), `LOOKING AWAY` (Orange).
  - **Non-blocking acoustic alert**: triggers after 1.0 second of sustained inattention.

**Interactive Hotkeys during live stream:**
- `q` or `ESC`: Exit
- `s`: Save snapshot photo
- `r`: Reset alert frame counters
- `m`: Mute / Unmute acoustic sound alert

### 2. Run with INT8 Quantized Model (Ultra-low Edge Latency)
```bash
python3 main.py --quantized
```

### 3. Run Section 7 Performance Benchmark Suite
To generate the formal benchmarks (FPS, latency, peak RAM via `tracemalloc`, and low-light cabin simulation) required by Table 7 of the methodology:
```bash
python3 benchmark.py --frames 300
```
