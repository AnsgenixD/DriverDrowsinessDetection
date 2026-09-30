# Geometric calculations: EAR, MAR, and Head Pose estimation
import math
import cv2
import numpy as np
import config

FACE_3D_MODEL_NP = np.array(config.FACE_3D_MODEL, dtype=np.float64)


def dist(p1, p2):
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def calculate_ear(eye_points):
    """
    Eye Aspect Ratio (EAR) = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Drops close to 0 when the eye is closed.
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    v1 = dist(p2, p6)
    v2 = dist(p3, p5)
    h = dist(p1, p4)

    if h < 1e-5:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def calculate_mar(mouth_points):
    """
    Mouth Aspect Ratio (MAR) = (|q2-q8| + |q3-q7| + |q4-q6|) / (3 * |q1-q5|)
    Increases sharply when yawning.
    """
    q1, q2, q3, q4, q5, q6, q7, q8 = mouth_points
    v1 = dist(q2, q8)
    v2 = dist(q3, q7)
    v3 = dist(q4, q6)
    h = dist(q1, q5)

    if h < 1e-5:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h)


def estimate_head_pose(points_2d, frame_w, frame_h):
    """
    Estimates Pitch, Yaw, Roll using OpenCV solvePnP.
    Returns: pitch, yaw, roll (in degrees)
    """
    image_points = np.array(points_2d, dtype=np.float64)

    # Approximate camera matrix
    focal_length = frame_w
    center = (frame_w / 2.0, frame_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL_NP,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0, None, None

    rmat, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles from rotation matrix
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll), rvec, tvec


def extract_geometric_vector(ear_l, ear_r, mar, yaw, pitch, roll):
    """Packs geometric features into a 7D normalized vector for the model."""
    ear_avg = (ear_l + ear_r) / 2.0
    return np.array([
        ear_l,
        ear_r,
        ear_avg,
        mar,
        yaw / 90.0,
        pitch / 90.0,
        roll / 90.0
    ], dtype=np.float32)
