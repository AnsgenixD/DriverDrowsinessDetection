# main.py - Real-Time Driver Drowsiness & Distraction Detection

import os
import sys
import time
import argparse
from pathlib import Path
import cv2

# Add root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.landmark_detector import FaceMeshDetector
from core.fusion_classifier import DMSClassifier
from core.alert_manager import AlertManager


def run_pipeline(
    camera_idx: int = config.CAMERA_INDEX,
    video_source: str | None = None,
    model_type: str = "custom_cnn",
    weights_path: str | None = None,
    use_quantization: bool = False,
    mute_audio: bool = False,
):
    print("Starting Driver Drowsiness and Distraction Detection...")

    # Load face mesh detector
    print("Loading MediaPipe Face Mesh...")
    detector = FaceMeshDetector()

    # Load classification model
    print(f"Loading {model_type} classifier...")
    classifier = DMSClassifier(
        weights_path=weights_path,
        model_type=model_type,
        use_quantization=use_quantization,
    )

    # Initialize alert manager
    alert_manager = AlertManager()
    if mute_audio:
        alert_manager._play_sound = lambda state: None  # Mute audio

    # Open Camera or Video Source
    source = video_source if video_source is not None else camera_idx
    print(f"[Pipeline] Opening video input source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[Error] Could not open video source {source}. If using camera, verify permissions.")
        return

    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # Performance tracking variables
    prev_time = time.time()
    fps_history = []
    fps = 0.0

    print("\n--- Running Live Inference Loop ---")
    print("Controls:")
    print("  'q' or ESC : Exit application")
    print("  's'        : Save snapshot")
    print("  'r'        : Reset alert counters")
    print("  'm'        : Toggle audio mute\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if video_source is not None:
                    # Loop video if input is a file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[Pipeline] Camera stream disconnected.")
                    break

            # Mirror frame horizontally for natural webcam interaction
            if video_source is None:
                frame = cv2.flip(frame, 1)

            current_time = time.time()
            fps_val = 1.0 / max(1e-5, (current_time - prev_time))
            prev_time = current_time

            fps_history.append(fps_val)
            if len(fps_history) > 15:
                fps_history.pop(0)
            fps = sum(fps_history) / len(fps_history)

            # Detect face and facial landmarks
            landmark_data = detector.process(frame)

            if landmark_data is not None:
                # Run prediction (geometry + neural network)
                pred_result = classifier.predict(frame, landmark_data)

                # Update alert status
                filtered_state, is_alert = alert_manager.update_state(
                    predicted_state=pred_result["state"],
                    ear=pred_result["ear"],
                    mar=pred_result["mar"],
                    yaw=pred_result["yaw"],
                    pitch=pred_result["pitch"],
                )

                # Draw HUD on frame
                display_frame = alert_manager.draw_hud(
                    frame=frame,
                    state=filtered_state,
                    confidence=pred_result["confidence"],
                    ear=pred_result["ear"],
                    mar=pred_result["mar"],
                    yaw=pred_result["yaw"],
                    pitch=pred_result["pitch"],
                    fps=fps,
                    face_bbox=pred_result["face_bbox"],
                    landmarks=pred_result["all_points"],
                    composite_patch=pred_result["composite_patch"],
                )
            else:
                # No face detected
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    "SEARCHING FOR DRIVER...",
                    (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                )

            # Display output
            cv2.imshow("Driver Monitoring System", display_frame)

            # Handle keystrokes
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:  # ESC or 'q'
                break
            elif key == ord("s"):
                snapshot_file = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snapshot_file, display_frame)
                print(f"[Snapshot] Saved {snapshot_file}")
            elif key == ord("c"):
                if landmark_data is not None and "pred_result" in locals():
                    classifier.set_baseline_pose(pred_result["yaw"], pred_result["pitch"])
            elif key == ord("r"):
                alert_manager.drowsy_frame_count = 0
                alert_manager.yawn_frame_count = 0
                alert_manager.lookaway_frame_count = 0
                print("[Reset] Alert counters cleared.")
            elif key == ord("m"):
                mute_audio = not mute_audio
                if mute_audio:
                    alert_manager._play_sound = lambda state: None
                    print("[Audio] Alerts Muted.")
                else:
                    alert_manager = AlertManager()
                    print("[Audio] Alerts Unmuted.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Driver Monitoring System stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Drowsiness & Distraction Detection (Real-Time DMS)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file instead of live camera")
    parser.add_argument("--model", type=str, choices=["custom_cnn", "mobilenet_v3"], default="custom_cnn")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights file (.pth)")
    parser.add_argument("--quantized", action="store_true", help="Use INT8 post-training quantized model")
    parser.add_argument("--no-audio", action="store_true", help="Disable acoustic alerts")
    args = parser.parse_args()

    run_pipeline(
        camera_idx=args.camera,
        video_source=args.video,
        model_type=args.model,
        weights_path=args.weights,
        use_quantization=args.quantized,
        mute_audio=args.no_audio,
    )
