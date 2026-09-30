# Configuration file for Driver Drowsiness and Distraction Detection
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "weights"
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_WEIGHTS_PATH = MODEL_DIR / "best_dms_model.pth"
QUANTIZED_WEIGHTS_PATH = MODEL_DIR / "dms_quantized_int8.pth"
TASK_MODEL_PATH = BASE_DIR / "face_landmarker.task"
TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Driver states
CLASSES = ["Focused", "Drowsy", "Yawning", "Looking Away"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}

# BGR colors for OpenCV UI
COLORS = {
    "Focused": (0, 200, 0),        # Green
    "Drowsy": (0, 0, 255),         # Red
    "Yawning": (0, 215, 255),      # Yellow/Gold
    "Looking Away": (255, 140, 0), # Orange
    "Neutral": (200, 200, 200),    # Gray
}

# --- Thresholds ---
# Eye Aspect Ratio (EAR)
EAR_THRESHOLD = 0.20             # Below this means eyes are closed
EAR_CONSEC_FRAMES = 15           # ~0.5 - 1.0 second before drowsiness alert

# Mouth Aspect Ratio (MAR)
MAR_THRESHOLD = 0.60             # Above this means mouth is wide open (yawn)
MAR_CONSEC_FRAMES = 20           # Sustained frames before yawn alert

# Head Pose angles (in degrees)
YAW_THRESHOLD = 28.0             # Turning head left/right
PITCH_UP_THRESHOLD = 24.0        # Tilting head up
PITCH_DOWN_THRESHOLD = 22.0      # Looking down (phone/speedometer)
POSE_CONSEC_FRAMES = 75          # ~2.5s leeway for mirror checks before alerting

# Alert cooldown to prevent annoying continuous beeps
ALERT_COOLDOWN_SEC = 1.0

# --- MediaPipe Landmark Indices ---
# Left and Right eyes (6 points each)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Mouth contour (8 points)
MOUTH_LANDMARKS = [61, 39, 0, 269, 291, 405, 17, 181]

# 2D points used for solvePnP head pose
HEAD_POSE_LANDMARKS = [
    1,    # Nose tip
    152,  # Chin
    33,   # Left eye corner
    263,  # Right eye corner
    61,   # Left mouth corner
    291   # Right mouth corner
]

# 3D generic face model points (in mm)
FACE_3D_MODEL = [
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, 110.0, -20.0),      # Chin
    (-75.0, -75.0, -50.0),    # Left eye corner
    (75.0, -75.0, -50.0),     # Right eye corner
    (-40.0, 50.0, -30.0),     # Left mouth corner
    (40.0, 50.0, -30.0)       # Right mouth corner
]

# Model input parameters
IMAGE_INPUT_SIZE = (112, 112)
GEOM_FEATURE_DIM = 7          # [ear_l, ear_r, ear_avg, mar, yaw, pitch, roll]
NUM_CLASSES = 4
