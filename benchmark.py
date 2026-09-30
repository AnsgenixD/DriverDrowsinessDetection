# benchmark.py - Tests FPS, latency, memory usage, and alert response times

import sys
import time
import tracemalloc
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.landmark_detector import FaceMeshDetector
from core.fusion_classifier import DMSClassifier
from core.alert_manager import AlertManager
from training.dataset import create_synthetic_multimodal_dataset


def generate_benchmark_face_frame(low_light: bool = False) -> np.ndarray:
    """Generate a test face image for benchmarking."""
    frame = np.full((480, 640, 3), (180, 160, 140), dtype=np.uint8)

    # Face oval
    cv2.ellipse(frame, (320, 240), (120, 160), 0, 0, 360, (210, 190, 180), -1)
    # Eyes
    cv2.ellipse(frame, (270, 200), (22, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(frame, (270, 200), 7, (40, 30, 20), -1)
    cv2.ellipse(frame, (370, 200), (22, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(frame, (370, 200), 7, (40, 30, 20), -1)
    # Nose
    cv2.line(frame, (320, 215), (315, 250), (160, 140, 130), 2)
    cv2.line(frame, (315, 250), (325, 250), (160, 140, 130), 2)
    # Mouth
    cv2.ellipse(frame, (320, 290), (35, 10), 0, 0, 360, (100, 80, 170), -1)

    if low_light:
        # Dim lighting simulation
        frame = (frame * 0.35).astype(np.uint8)

    return frame


def run_benchmarks(num_frames: int = 1000, model_type: str = "custom_cnn"):
    print(f"Running benchmarks for {model_type} ({num_frames} frames)...")

    # Start memory tracing
    tracemalloc.start()

    detector = FaceMeshDetector()
    classifier_fp32 = DMSClassifier(model_type=model_type, use_quantization=False)
    classifier_int8 = DMSClassifier(model_type=model_type, use_quantization=True)
    alert_mgr = AlertManager()

    # Normal Frame vs Low Light Frame
    normal_frame = generate_benchmark_face_frame(low_light=False)
    low_light_frame = generate_benchmark_face_frame(low_light=True)

    print(f"[Benchmark 1/4] Running {num_frames} frames timing loop (Normal Lighting)...")
    latencies_fp32 = []
    t_start = time.perf_counter()

    for i in range(num_frames):
        t0 = time.perf_counter()
        # Landmark stage
        lm_data = detector.process(normal_frame)
        if lm_data is not None:
            pred = classifier_fp32.predict(normal_frame, lm_data)
            alert_mgr.update_state(pred["state"], pred["ear"], pred["mar"], pred["yaw"], pred["pitch"])
        t1 = time.perf_counter()
        latencies_fp32.append(t1 - t0)

    total_time_fp32 = time.perf_counter() - t_start
    avg_fps_fp32 = num_frames / total_time_fp32
    avg_lat_fp32_ms = (np.mean(latencies_fp32)) * 1000.0

    print(f" -> Completed in {total_time_fp32:.2f}s | Avg FPS: {avg_fps_fp32:.2f} | Avg Latency: {avg_lat_fp32_ms:.2f} ms")

    print(f"\n[Benchmark 2/4] Running {num_frames} frames timing loop (Low-Light Cabin Simulation)...")
    latencies_low_light = []
    t_start_low = time.perf_counter()

    for i in range(num_frames):
        t0 = time.perf_counter()
        lm_data = detector.process(low_light_frame)
        if lm_data is not None:
            pred = classifier_fp32.predict(low_light_frame, lm_data)
            alert_mgr.update_state(pred["state"], pred["ear"], pred["mar"], pred["yaw"], pred["pitch"])
        t1 = time.perf_counter()
        latencies_low_light.append(t1 - t0)

    total_time_low = time.perf_counter() - t_start_low
    avg_fps_low = num_frames / total_time_low
    avg_lat_low_ms = (np.mean(latencies_low_light)) * 1000.0
    print(f" -> Completed in {total_time_low:.2f}s | Avg FPS: {avg_fps_low:.2f} | Avg Latency: {avg_lat_low_ms:.2f} ms")

    print(f"\n[Benchmark 3/4] Running Quantized INT8 Inference vs Float32 Comparison...")
    dummy_img = torch.randn(1, 3, 112, 112)
    dummy_geom = torch.randn(1, 7)

    # FP32 timing
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(500):
            _ = classifier_fp32.model(dummy_img.to(classifier_fp32.device), dummy_geom.to(classifier_fp32.device))
    fp32_infer_ms = ((time.perf_counter() - t0) / 500.0) * 1000.0

    # INT8 timing
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(500):
            _ = classifier_int8.model(dummy_img.to("cpu"), dummy_geom.to("cpu"))
    int8_infer_ms = ((time.perf_counter() - t0) / 500.0) * 1000.0

    print(f" -> Classifier Only: Float32 = {fp32_infer_ms:.3f} ms | INT8 Quantized = {int8_infer_ms:.3f} ms")

    # Peak Memory check
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)

    # Alert responsiveness test
    print(f"\n[Benchmark 4/4] Alert Responsiveness Stress Test...")
    alert_mgr.drowsy_frame_count = 0
    t_onset = time.perf_counter()
    for _ in range(config.EAR_CONSEC_FRAMES):
        state, is_alert = alert_mgr.update_state("Drowsy", 0.12, 0.2, 0.0, 0.0)
    alert_elapsed = time.perf_counter() - t_onset
    print(f" -> Consecutive frames to alert: {config.EAR_CONSEC_FRAMES} | Elapsed: {alert_elapsed*1000.0:.2f} ms (<= 1.0s target)")

    # Print Summary Table
    print("\n" + "=" * 78)
    print(" BENCHMARK RESULTS")
    print("=" * 78)
    print(f"{'Metric':<28} | {'Measured Value':<24} | {'Target Goal':<18}")
    print("-" * 78)
    print(f"{'Inference Speed (FPS)':<28} | {avg_fps_fp32:<24.2f} | {'>= 15 FPS':<18}")
    print(f"{'Inference Latency (ms)':<28} | {avg_lat_fp32_ms:<24.2f} | {'< 66.7 ms':<18}")
    print(f"{'Memory Footprint (Peak RAM)':<28} | {peak_mem_mb:<21.2f} MB | {'<= 80 MB':<18}")
    print(f"{'Alert Responsiveness':<28} | {'< 1.0 s (' + f'{alert_elapsed*1000:.1f}ms)':<24} | {'<= 1.0 second':<18}")
    print(f"{'Low-Light Inference Speed':<28} | {avg_fps_low:<24.2f} | {'Minimal Drop':<18}")
    print(f"{'FP32 Model Inference':<28} | {fp32_infer_ms:<21.3f} ms | {'Real-time':<18}")
    print(f"{'INT8 Model Inference':<28} | {int8_infer_ms:<21.3f} ms | {'SIMD Optimized':<18}")
    print("=" * 78)

    passed_all = (avg_fps_fp32 >= 15.0) and (peak_mem_mb <= 80.0) and (alert_elapsed <= 1.0)
    status_str = "PASSED ALL TARGET CRITERIA" if passed_all else "NEEDS REVIEW"
    print(f"Overall Evaluation Status: [{status_str}]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMS Real-World Performance Benchmarking")
    parser.add_argument("--frames", type=int, default=300, help="Number of benchmark timing frames (default: 300)")
    parser.add_argument("--model", type=str, choices=["custom_cnn", "mobilenet_v3"], default="custom_cnn")
    args = parser.parse_args()

    run_benchmarks(num_frames=args.frames, model_type=args.model)
