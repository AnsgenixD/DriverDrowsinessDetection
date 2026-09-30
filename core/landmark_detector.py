# landmark_detector.py - MediaPipe Face Mesh detector for 468 facial landmarks

import os
import platform
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import config


class FaceMeshDetector:
    def __init__(self, task_model_path: str | None = None):
        self.task_model_path = task_model_path or str(config.TASK_MODEL_PATH)
        self.use_tasks_api = hasattr(mp, "tasks")
        self.detector = None
        self.is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        self._init_detector()

    def _ensure_task_model(self):
        """Ensure face_landmarker.task file exists locally; download if needed."""
        if not os.path.exists(self.task_model_path):
            print(f"[MediaPipe] Downloading face_landmarker.task from {config.TASK_MODEL_URL}...")
            urllib.request.urlretrieve(config.TASK_MODEL_URL, self.task_model_path)
            print(f"[MediaPipe] Download complete: {os.path.getsize(self.task_model_path)} bytes.")

    def _init_detector(self):
        """Initialize appropriate MediaPipe FaceMesh engine."""
        if self.use_tasks_api:
            self._ensure_task_model()
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python

            # On Apple Silicon macOS, Metal GPU delegate works stably with RGBA buffers
            delegate = python.BaseOptions.Delegate.GPU if self.is_apple_silicon else python.BaseOptions.Delegate.CPU
            
            try:
                base_options = python.BaseOptions(
                    model_asset_path=self.task_model_path,
                    delegate=delegate,
                )
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
                print(f"[MediaPipe] Task Landmarker initialized successfully (Delegate: {delegate.name}).")
            except Exception as e:
                print(f"[MediaPipe] Warning: GPU delegate initialization failed ({e}), falling back to CPU...")
                base_options = python.BaseOptions(
                    model_asset_path=self.task_model_path,
                    delegate=python.BaseOptions.Delegate.CPU,
                )
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
        elif hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.detector = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("[MediaPipe] Legacy FaceMesh solutions API initialized successfully.")
        else:
            raise RuntimeError("Unable to initialize MediaPipe FaceMesh detector.")

    def process(self, frame_bgr: np.ndarray) -> dict | None:
        """
        Process a BGR frame and extract facial landmarks and keypoints.
        Returns a dict containing pixel coordinates and normalized landmarks, or None if no face detected.
        """
        h, w = frame_bgr.shape[:2]

        if self.use_tasks_api:
            # Prepare image format
            if self.is_apple_silicon:
                frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            detection_result = self.detector.detect(mp_image)
            if not detection_result.face_landmarks:
                return None

            landmarks = detection_result.face_landmarks[0]
            # Convert normalized coordinates to pixel numpy array (468, 2)
            pts_2d = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.detector.process(frame_rgb)
            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            pts_2d = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

        # Extract landmark groups
        left_eye_pts = [pts_2d[idx] for idx in config.LEFT_EYE_LANDMARKS]
        right_eye_pts = [pts_2d[idx] for idx in config.RIGHT_EYE_LANDMARKS]
        mouth_pts = [pts_2d[idx] for idx in config.MOUTH_LANDMARKS]
        head_pose_pts = [pts_2d[idx] for idx in config.HEAD_POSE_LANDMARKS]

        # Face bounding box
        x_min = max(0, int(np.min(pts_2d[:, 0])))
        y_min = max(0, int(np.min(pts_2d[:, 1])))
        x_max = min(w, int(np.max(pts_2d[:, 0])))
        y_max = min(h, int(np.max(pts_2d[:, 1])))

        return {
            "all_points": pts_2d,
            "left_eye": left_eye_pts,
            "right_eye": right_eye_pts,
            "mouth": mouth_pts,
            "head_pose_pts": head_pose_pts,
            "face_bbox": (x_min, y_min, x_max, y_max),
        }

    def close(self):
        if hasattr(self.detector, "close"):
            self.detector.close()
