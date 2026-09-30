# Dual-branch fusion classifier combining facial crops with geometric numbers
import os
import math
import torch
import torch.nn.functional as F
import numpy as np

import config
from models.custom_cnn import Custom2DCNN_Fusion
from models.mobilenet_v3 import MobileNetV3Small_Fusion
from models.quantization import quantize_model_dynamic
from core.geometry import (
    calculate_ear,
    calculate_mar,
    estimate_head_pose,
    extract_geometric_vector,
)
from core.visual_extractor import (
    extract_composite_visual_patch,
    prepare_visual_tensor,
)


class DMSClassifier:
    def __init__(self, weights_path=None, model_type="custom_cnn", use_quantization=False):
        self.model_type = model_type
        self.use_quantization = use_quantization
        self.device = torch.device("cpu" if use_quantization else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self.model = None
        self.is_neural_ready = False

        # Baseline angles to compensate for off-center phone mount
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0

        self._load_model(weights_path)

    def set_baseline_pose(self, yaw, pitch):
        """Zeros out current camera angle as the forward road direction."""
        self.baseline_yaw = yaw
        self.baseline_pitch = pitch
        print(f"[DMS] Baseline road pose set: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°")

    def _load_model(self, weights_path):
        """Initializes model architecture and loads weights if available."""
        if self.model_type == "mobilenet_v3":
            base_model = MobileNetV3Small_Fusion()
        else:
            base_model = Custom2DCNN_Fusion()

        target_path = weights_path or str(config.DEFAULT_WEIGHTS_PATH)
        if os.path.exists(target_path):
            try:
                base_model.load_state_dict(torch.load(target_path, map_location="cpu"))
                self.is_neural_ready = True
                print(f"[DMS] Loaded model weights from {target_path}")
            except Exception as e:
                print(f"[DMS] Could not load weights: {e}")
                self.is_neural_ready = False
        else:
            print("[DMS] No weights file found. Running in rule-based geometric mode.")

        if self.use_quantization:
            self.model = quantize_model_dynamic(base_model)
        else:
            self.model = base_model.to(self.device)

        self.model.eval()

    def rule_based_classify(self, ear, mar, yaw, pitch):
        """
        Classifies driver state using geometric thresholds.
        Evaluates head distraction relative to the calibrated road baseline.
        """
        delta_yaw = yaw - self.baseline_yaw
        delta_pitch = pitch - self.baseline_pitch

        # Foreshortening compensation for side-angle view
        cos_yaw = max(0.35, math.cos(math.radians(abs(delta_yaw))))
        comp_mar = mar * cos_yaw

        # Foreshortening compensation for slight downward gaze
        comp_ear = ear
        if delta_pitch < 0 and delta_pitch >= -config.PITCH_DOWN_THRESHOLD:
            cos_pitch = max(0.4, math.cos(math.radians(abs(delta_pitch))))
            comp_ear = ear / cos_pitch

        # 1. Check if looking away from the road (Priority 1)
        if abs(delta_yaw) > config.YAW_THRESHOLD or delta_pitch > config.PITCH_UP_THRESHOLD or delta_pitch < -config.PITCH_DOWN_THRESHOLD:
            conf = min(1.0, max(abs(delta_yaw) / 45.0, abs(delta_pitch) / 35.0))
            return "Looking Away", float(conf)

        # 2. Check if mouth is wide open (Yawning)
        if comp_mar > config.MAR_THRESHOLD:
            conf = min(1.0, comp_mar / 0.85)
            return "Yawning", float(conf)

        # 3. Check if eyes are closed (Drowsiness / Microsleep)
        if comp_ear < config.EAR_THRESHOLD:
            conf = min(1.0, (config.EAR_THRESHOLD - comp_ear) / config.EAR_THRESHOLD + 0.6)
            return "Drowsy", float(conf)

        # 4. Otherwise driver is focused on the road
        return "Focused", 0.95

    def predict(self, frame, landmark_data):
        """Runs one full detection step on a single frame."""
        h, w = frame.shape[:2]

        left_eye_pts = landmark_data["left_eye"]
        right_eye_pts = landmark_data["right_eye"]
        mouth_pts = landmark_data["mouth"]
        pose_pts = landmark_data["head_pose_pts"]

        # Calculate geometric metrics
        ear_l = calculate_ear(left_eye_pts)
        ear_r = calculate_ear(right_eye_pts)
        ear_avg = (ear_l + ear_r) / 2.0
        mar = calculate_mar(mouth_pts)

        pitch, yaw, roll, _, _ = estimate_head_pose(pose_pts, w, h)
        geom_vector = extract_geometric_vector(ear_l, ear_r, mar, yaw, pitch, roll)

        # Extract eye and mouth crops
        composite_patch, _, _, _ = extract_composite_visual_patch(
            frame, left_eye_pts, right_eye_pts, mouth_pts
        )

        # Classify via neural network or rule-based fallback
        if self.is_neural_ready:
            with torch.no_grad():
                img_t = prepare_visual_tensor(composite_patch, self.device)
                geom_t = torch.from_numpy(geom_vector).unsqueeze(0).to(self.device)
                logits = self.model(img_t, geom_t)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                state = config.CLASSES[pred_idx]
                confidence = float(probs[pred_idx])
        else:
            state, confidence = self.rule_based_classify(ear_avg, mar, yaw, pitch)
            probs = [0.0] * 4
            probs[config.CLASS_TO_IDX[state]] = confidence

        return {
            "state": state,
            "confidence": confidence,
            "probabilities": probs,
            "ear": ear_avg,
            "ear_l": ear_l,
            "ear_r": ear_r,
            "mar": mar,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "geom_vector": geom_vector,
            "composite_patch": composite_patch,
            "face_bbox": landmark_data["face_bbox"],
            "all_points": landmark_data["all_points"],
        }
