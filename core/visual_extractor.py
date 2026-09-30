# visual_extractor.py - Crops eye and mouth regions from video frames

import cv2
import numpy as np
import torch
import config


def get_region_crop(
    frame: np.ndarray,
    landmarks: list[np.ndarray],
    target_size: tuple[int, int],
    margin_ratio: float = 0.25,
) -> np.ndarray:
    """
    Extract a cropped bounding box around specified landmark points with safety margin.
    """
    h, w = frame.shape[:2]
    pts = np.array(landmarks)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)

    # Compute margins
    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_x = box_w * margin_ratio
    pad_y = box_h * margin_ratio

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))

    if x2 <= x1 or y2 <= y1:
        # Fallback to zero image if invalid bounding box
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)


def extract_composite_visual_patch(
    frame: np.ndarray,
    left_eye_pts: list[np.ndarray],
    right_eye_pts: list[np.ndarray],
    mouth_pts: list[np.ndarray],
    target_dim: tuple[int, int] = config.IMAGE_INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts ocular and perioral regions and creates a standardized composite visual image:
    - Top half: Both eyes panorama (or eyes stitched)
    - Bottom half: Mouth region
    Returns:
    (composite_patch, left_eye_crop, right_eye_crop, mouth_crop)
    """
    target_h, target_w = target_dim
    half_h = target_h // 2
    half_w = target_w // 2

    # Left eye crop (half_h x half_w)
    left_crop = get_region_crop(frame, left_eye_pts, (half_w, half_h), margin_ratio=0.35)
    # Right eye crop (half_h x half_w)
    right_crop = get_region_crop(frame, right_eye_pts, (half_w, half_h), margin_ratio=0.35)
    # Both eyes stitched horizontally into top half (half_h x target_w)
    eyes_panorama = np.hstack([left_crop, right_crop])

    # Mouth crop (half_h x target_w)
    mouth_crop = get_region_crop(frame, mouth_pts, (target_w, half_h), margin_ratio=0.30)

    # Stack eyes on top, mouth on bottom
    composite = np.vstack([eyes_panorama, mouth_crop])
    return composite, left_crop, right_crop, mouth_crop


def prepare_visual_tensor(composite_patch: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert (H, W, 3) BGR image to normalized PyTorch tensor (1, 3, H, W) on specified device.
    """
    # Convert BGR to RGB, normalize to [0, 1]
    rgb = cv2.cvtColor(composite_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Standard normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std

    # Transpose from (H, W, C) to (C, H, W)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor
