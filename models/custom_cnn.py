# custom_cnn.py - Lightweight dual-branch neural network combining face crops and geometric values

import torch
import torch.nn as nn
import config


class VisualCNNBranch(nn.Module):
    """Compact 4-layer 2D Convolutional feature extractor."""
    def __init__(self, in_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        self.conv_net = nn.Sequential(
            # Block 1: 112x112 -> 56x56
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 56x56 -> 28x28
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_net(x)


class GeometricBranch(nn.Module):
    """Dense network to embed the 7D geometric feature vector."""
    def __init__(self, in_features: int = config.GEOM_FEATURE_DIM, out_features: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Custom2DCNN_Fusion(nn.Module):
    """
    Dual-branch DMS Classifier:
    Concatenates visual embeddings with explicit geometric scalar signals.
    """
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        geom_dim: int = config.GEOM_FEATURE_DIM,
    ):
        super().__init__()
        self.visual_branch = VisualCNNBranch(in_channels=3, feature_dim=64)
        self.geom_branch = GeometricBranch(in_features=geom_dim, out_features=32)

        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(48, num_classes),
        )

    def forward(self, img: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        v_feat = self.visual_branch(img)
        g_feat = self.geom_branch(geom)
        fused = torch.cat([v_feat, g_feat], dim=1)
        return self.classifier(fused)
