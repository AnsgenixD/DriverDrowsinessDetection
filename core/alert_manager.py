# alert_manager.py - Handles drawing HUD alerts on screen and playing audio warnings

import os
import platform
import subprocess
import threading
import time
import cv2
import numpy as np
import config


class AlertManager:
    def __init__(self):
        self.last_audio_alert_time = 0.0
        self.current_state = "Focused"
        self.state_start_time = time.time()
        self.alert_triggered = False

        # Consecutive frame counters for temporal filtering
        self.drowsy_frame_count = 0
        self.yawn_frame_count = 0
        self.lookaway_frame_count = 0

        self.system = platform.system()

    def update_state(self, predicted_state: str, ear: float, mar: float, yaw: float, pitch: float) -> tuple[str, bool]:
        """
        Smooths predictions over time to avoid flickering alerts.
        Returns: (state_to_display, is_alert_active)
        """
        now = time.time()

        # Update frame counters based on predicted state
        if predicted_state == "Drowsy":
            self.drowsy_frame_count += 1
            self.yawn_frame_count = max(0, self.yawn_frame_count - 1)
            self.lookaway_frame_count = max(0, self.lookaway_frame_count - 1)
        elif predicted_state == "Yawning":
            self.yawn_frame_count += 1
            self.drowsy_frame_count = max(0, self.drowsy_frame_count - 1)
            self.lookaway_frame_count = max(0, self.lookaway_frame_count - 1)
        elif predicted_state == "Looking Away":
            self.lookaway_frame_count += 1
            self.drowsy_frame_count = max(0, self.drowsy_frame_count - 1)
            self.yawn_frame_count = max(0, self.yawn_frame_count - 1)
        else:
            # Focused / Normal
            self.drowsy_frame_count = max(0, self.drowsy_frame_count - 2)
            self.yawn_frame_count = max(0, self.yawn_frame_count - 2)
            self.lookaway_frame_count = max(0, self.lookaway_frame_count - 2)

        # Evaluate state persistence against thresholds
        active_alert = False
        display_state = "Focused"

        if self.drowsy_frame_count >= config.EAR_CONSEC_FRAMES:
            display_state = "Drowsy"
            active_alert = True
        elif self.yawn_frame_count >= config.MAR_CONSEC_FRAMES:
            display_state = "Yawning"
            active_alert = True
        elif self.lookaway_frame_count >= config.POSE_CONSEC_FRAMES:
            display_state = "Looking Away"
            active_alert = True

        # Trigger acoustic alarm if in alert state and cooldown has elapsed
        if active_alert:
            if now - self.last_audio_alert_time >= config.ALERT_COOLDOWN_SEC:
                self._play_sound(display_state)
                self.last_audio_alert_time = now

        self.current_state = display_state
        self.alert_triggered = active_alert
        return display_state, active_alert

    def _play_sound(self, state: str):
        """Asynchronously play platform-native acoustic alert without blocking video loop."""
        def _sound_worker():
            try:
                if self.system == "Darwin":
                    # macOS system alert sounds
                    sound_file = "/System/Library/Sounds/Sosumi.aiff" if state == "Drowsy" else "/System/Library/Sounds/Ping.aiff"
                    if os.path.exists(sound_file):
                        subprocess.run(["afplay", sound_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        print("\a", end="", flush=True)
                elif self.system == "Linux":
                    # Linux / Raspberry Pi beep or aplay
                    subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    # Windows
                    import winsound
                    winsound.Beep(1200, 300)
            except Exception:
                # Terminal bell fallback
                print("\a", end="", flush=True)

        threading.Thread(target=_sound_worker, daemon=True).start()

    def draw_hud(
        self,
        frame: np.ndarray,
        state: str,
        confidence: float,
        ear: float,
        mar: float,
        yaw: float,
        pitch: float,
        fps: float,
        face_bbox: tuple[int, int, int, int] | None = None,
        landmarks: np.ndarray | None = None,
        composite_patch: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draws the status bar, metrics, and face tracking boxes on the frame."""
        h, w = frame.shape[:2]
        output = frame.copy()
        color = config.COLORS.get(state, config.COLORS["Neutral"])

        # 1. Draw face bounding box
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                f"Driver: {state}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # 2. Draw eye and mouth landmarks
        if landmarks is not None:
            for idx in config.LEFT_EYE_LANDMARKS + config.RIGHT_EYE_LANDMARKS:
                pt = tuple(landmarks[idx].astype(int))
                cv2.circle(output, pt, 2, (0, 255, 255), -1)
            for idx in config.MOUTH_LANDMARKS:
                pt = tuple(landmarks[idx].astype(int))
                cv2.circle(output, pt, 2, (255, 0, 255), -1)

        # 3. Top status banner
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)

        status_text = f"STATE: {state.upper()}"
        cv2.putText(output, status_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
        cv2.putText(
            output,
            f"Conf: {confidence*100:.1f}% | FPS: {fps:.1f}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )

        # Flashing alert text when dangerous state is detected
        if self.alert_triggered and int(time.time() * 4) % 2 == 0:
            alert_msg = f"*** ALERT: {state.upper()} ***"
            cv2.putText(output, alert_msg, (w // 2 - 140, 48), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

        # 4. Metrics side card
        card_w, card_h = 230, 160
        card_x = w - card_w - 15
        card_y = 95

        card_overlay = output.copy()
        cv2.rectangle(card_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 25, 25), -1)
        cv2.addWeighted(card_overlay, 0.65, output, 0.35, 0, output)
        cv2.rectangle(output, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1)

        cv2.putText(output, "DRIVER TELEMETRY", (card_x + 12, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

        # Helper to draw progress bars
        def draw_bar(y_pos, label, value, threshold, max_val, warn_above=False):
            cv2.putText(output, f"{label}: {value:.2f}", (card_x + 12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            bar_x = card_x + 95
            bar_w = 115
            bar_h = 10
            cv2.rectangle(output, (bar_x, y_pos - 10), (bar_x + bar_w, y_pos - 10 + bar_h), (50, 50, 50), -1)
            
            fill_pct = min(1.0, max(0.0, value / max_val))
            fill_w = int(fill_pct * bar_w)
            
            # Warning color selection
            is_warning = (value > threshold) if warn_above else (value < threshold)
            bar_color = (0, 0, 255) if is_warning else (0, 220, 0)
            
            cv2.rectangle(output, (bar_x, y_pos - 10), (bar_x + fill_w, y_pos - 10 + bar_h), bar_color, -1)
            # Threshold marker
            thresh_x = int(bar_x + min(1.0, threshold / max_val) * bar_w)
            cv2.line(output, (thresh_x, y_pos - 12), (thresh_x, y_pos + 2), (0, 255, 255), 2)

        # Draw EAR bar (warn below 0.22)
        draw_bar(card_y + 50, "EAR", ear, config.EAR_THRESHOLD, 0.45, warn_above=False)
        # Draw MAR bar (warn above 0.60)
        draw_bar(card_y + 80, "MAR", mar, config.MAR_THRESHOLD, 0.90, warn_above=True)

        # Head Pose text
        pose_color = (0, 0, 255) if (abs(yaw) > config.YAW_THRESHOLD or abs(pitch) > config.PITCH_UP_THRESHOLD) else (200, 200, 200)
        cv2.putText(
            output,
            f"Yaw: {yaw:+.1f} | Pitch: {pitch:+.1f}",
            (card_x + 12, card_y + 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            pose_color,
            1,
        )

        # Active filter counts
        cv2.putText(
            output,
            f"Frames: D:{self.drowsy_frame_count} Y:{self.yawn_frame_count} P:{self.lookaway_frame_count}",
            (card_x + 12, card_y + 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
        )

        # 5. Picture-in-Picture (PiP) Inset showing composite crop in bottom left
        if composite_patch is not None:
            pip_size = (90, 90)
            pip_resized = cv2.resize(composite_patch, pip_size)
            px1, py1 = 15, h - pip_size[1] - 15
            px2, py2 = px1 + pip_size[0], py1 + pip_size[1]
            output[py1:py2, px1:px2] = pip_resized
            cv2.rectangle(output, (px1, py1), (px2, py2), (100, 100, 100), 1)
            cv2.putText(output, "AI Crop", (px1 + 2, py1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return output
