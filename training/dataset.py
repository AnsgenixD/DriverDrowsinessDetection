# dataset.py - PyTorch Dataset for loading face patches and geometric features

import os
import sys
import random
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import config


class DMSDataset(Dataset):
    """
    Multimodal DMS Dataset returning:
    (image_tensor, geom_vector_tensor, label_idx)
    """
    def __init__(
        self,
        samples: list[dict],
        is_training: bool = True,
        augment_low_light: bool = True,
    ):
        self.samples = samples
        self.is_training = is_training
        self.augment_low_light = augment_low_light

    def __len__(self):
        return len(self.samples)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Apply cabin lighting variations, brightness jitter, and noise."""
        img = image.astype(np.float32)

        # 1. Random Brightness Jitter (-30 to +30)
        brightness = random.uniform(-25.0, 25.0)
        img = np.clip(img + brightness, 0, 255)

        # 2. Random Contrast (0.7 to 1.3)
        contrast = random.uniform(0.75, 1.25)
        img = np.clip((img - 127.5) * contrast + 127.5, 0, 255)

        # 3. Low-Light Cabin Simulation (random dimming to simulate night driving)
        if self.augment_low_light and random.random() < 0.35:
            dim_factor = random.uniform(0.3, 0.6)
            img = img * dim_factor

        # 4. Slight Gaussian noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 4.0, img.shape)
            img = np.clip(img + noise, 0, 255)

        return img.astype(np.uint8)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]

        # 1. Load or retrieve image
        if isinstance(sample["image"], str):
            image = cv2.imread(sample["image"])
            if image is None:
                image = np.zeros((*config.IMAGE_INPUT_SIZE, 3), dtype=np.uint8)
            else:
                image = cv2.resize(image, config.IMAGE_INPUT_SIZE)
        else:
            image = sample["image"].copy()

        # Apply augmentation if in training mode
        if self.is_training:
            image = self._augment(image)

        # Normalize image to PyTorch tensor [3, H, W]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1))

        # 2. Geometric vector
        geom = np.array(sample["geom_vector"], dtype=np.float32)
        if self.is_training:
            # Add small sensory jitter to geometric features to prevent overfitting
            noise = np.random.normal(0, 0.015, geom.shape).astype(np.float32)
            geom = np.clip(geom + noise, -1.0, 1.0)
        geom_tensor = torch.from_numpy(geom)

        # 3. Target class index
        label = int(sample["label"])
        return img_tensor, geom_tensor, label


def create_synthetic_multimodal_dataset(
    num_samples_per_class: int = 400,
    val_ratio: float = 0.2,
) -> tuple[DMSDataset, DMSDataset]:
    """
    Generates realistic, physically-grounded multimodal samples across the 4 driver states:
    - Focused: EAR ~ 0.32-0.42, MAR ~ 0.10-0.25, Pose ~ [-8, 8] deg
    - Drowsy: EAR ~ 0.05-0.18 (closed eyes), MAR ~ 0.10-0.25, Pose ~ [-10, 10] deg
    - Yawning: EAR ~ 0.20-0.35, MAR ~ 0.65-0.90 (wide mouth), Pose ~ [-12, 12] deg
    - Looking Away: EAR ~ 0.25-0.40, MAR ~ 0.12-0.28, Yaw ~ [28, 65] deg or Pitch ~ [25, 45] deg
    Allows instantaneous training and evaluation without requiring multi-gigabyte downloads.
    """
    train_samples = []
    val_samples = []

    for label_idx, class_name in enumerate(config.CLASSES):
        for i in range(num_samples_per_class):
            # Create synthetic composite crop with ocular and perioral cues
            img = np.zeros((*config.IMAGE_INPUT_SIZE, 3), dtype=np.uint8)
            # Base skin tone background
            base_color = (random.randint(110, 160), random.randint(130, 180), random.randint(170, 210))
            img[:] = base_color

            if class_name == "Focused":
                ear = random.uniform(0.30, 0.42)
                mar = random.uniform(0.12, 0.25)
                yaw = random.uniform(-8.0, 8.0)
                pitch = random.uniform(-6.0, 6.0)
                roll = random.uniform(-4.0, 4.0)

                # Draw open eyes (ellipses)
                cv2.ellipse(img, (30, 28), (14, 8), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (30, 28), 5, (50, 40, 30), -1)
                cv2.ellipse(img, (82, 28), (14, 8), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (82, 28), 5, (50, 40, 30), -1)
                # Draw closed / relaxed mouth
                cv2.ellipse(img, (56, 84), (20, 4), 0, 0, 360, (80, 70, 160), -1)

            elif class_name == "Drowsy":
                ear = random.uniform(0.06, 0.18) # Dropping close to zero (Eq. 1)
                mar = random.uniform(0.10, 0.25)
                yaw = random.uniform(-10.0, 10.0)
                pitch = random.uniform(-12.0, 5.0)
                roll = random.uniform(-6.0, 6.0)

                # Draw closed eyes (thin horizontal slits / eyelid lines)
                cv2.line(img, (16, 28), (44, 28), (70, 60, 50), 3)
                cv2.line(img, (68, 28), (96, 28), (70, 60, 50), 3)
                # Normal mouth
                cv2.ellipse(img, (56, 84), (20, 4), 0, 0, 360, (80, 70, 160), -1)

            elif class_name == "Yawning":
                ear = random.uniform(0.18, 0.32) # Squinting / semi-open eyes during yawn
                mar = random.uniform(0.65, 0.90) # Sharp increase across 8 lip points (Eq. 2)
                yaw = random.uniform(-12.0, 12.0)
                pitch = random.uniform(-10.0, 15.0)
                roll = random.uniform(-6.0, 6.0)

                # Eyes squinting
                cv2.ellipse(img, (30, 28), (14, 4), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (30, 28), 3, (50, 40, 30), -1)
                cv2.ellipse(img, (82, 28), (14, 4), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (82, 28), 3, (50, 40, 30), -1)
                # Wide open mouth / dark oral cavity
                cv2.ellipse(img, (56, 84), (24, 20), 0, 0, 360, (20, 15, 30), -1)
                cv2.ellipse(img, (56, 84), (26, 22), 0, 0, 360, (80, 60, 160), 2)

            else:  # Looking Away
                ear = random.uniform(0.24, 0.38)
                mar = random.uniform(0.12, 0.28)
                side = random.choice([-1, 1])
                yaw = side * random.uniform(28.0, 55.0) # > 25 degrees
                pitch = random.uniform(-20.0, 25.0)
                roll = random.uniform(-10.0, 10.0)

                # Eyes turned to side
                pupil_shift = int(side * 6)
                cv2.ellipse(img, (30, 28), (14, 7), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (30 + pupil_shift, 28), 4, (50, 40, 30), -1)
                cv2.ellipse(img, (82, 28), (14, 7), 0, 0, 360, (255, 255, 255), -1)
                cv2.circle(img, (82 + pupil_shift, 28), 4, (50, 40, 30), -1)
                cv2.ellipse(img, (56, 84), (20, 4), 0, 0, 360, (80, 70, 160), -1)

            geom_vector = [
                ear,
                ear + random.uniform(-0.02, 0.02),
                ear,
                mar,
                yaw / 90.0,
                pitch / 90.0,
                roll / 90.0,
            ]

            sample = {
                "image": img,
                "geom_vector": geom_vector,
                "label": label_idx,
            }

            if i < int(num_samples_per_class * (1.0 - val_ratio)):
                train_samples.append(sample)
            else:
                val_samples.append(sample)

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    train_ds = DMSDataset(train_samples, is_training=True)
    val_ds = DMSDataset(val_samples, is_training=False)
    return train_ds, val_ds
