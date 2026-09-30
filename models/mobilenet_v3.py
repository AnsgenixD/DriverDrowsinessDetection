# mobilenet_v3.py - MobileNetV3-Small architecture for feature extraction

import torch
import torch.nn as nn
import config


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, in_channels: int, reduced_dim: int):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Bottleneck block with optional SE and Hardswish."""
    def __init__(
        self,
        in_ch: int,
        exp_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        act = nn.Hardswish if use_hs else nn.ReLU

        layers = []
        # Expansion
        if exp_ch != in_ch:
            layers.extend([
                nn.Conv2d(in_ch, exp_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(exp_ch),
                act(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(
                exp_ch,
                exp_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=exp_ch,
                bias=False,
            ),
            nn.BatchNorm2d(exp_ch),
            act(inplace=True),
        ])

        # Squeeze-and-Excitation Attention
        if use_se:
            layers.append(SqueezeExcitation(exp_ch, max(1, exp_ch // 4)))

        # Projection (Linear bottleneck)
        layers.extend([
            nn.Conv2d(exp_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV3SmallBackbone(nn.Module):
    """MobileNetV3-Small feature extraction trunk."""
    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        )

        # Specifications: (in_ch, exp_ch, out_ch, k, s, se, hs)
        configs = [
            (16, 16, 16, 3, 2, True, False),
            (16, 72, 24, 3, 2, False, False),
            (24, 88, 24, 3, 1, False, False),
            (24, 96, 40, 5, 2, True, True),
            (40, 240, 40, 5, 1, True, True),
            (40, 240, 40, 5, 1, True, True),
            (40, 120, 48, 5, 1, True, True),
            (48, 144, 48, 5, 1, True, True),
            (48, 288, 96, 5, 2, True, True),
        ]

        blocks = []
        for in_ch, exp_ch, out_ch, k, s, se, hs in configs:
            blocks.append(InvertedResidual(in_ch, exp_ch, out_ch, k, s, se, hs))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


class MobileNetV3Small_Fusion(nn.Module):
    """
    Fused MobileNetV3-Small DMS Classifier combining visual embeddings and 7D geometric signals.
    """
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        geom_dim: int = config.GEOM_FEATURE_DIM,
    ):
        super().__init__()
        self.visual_branch = MobileNetV3SmallBackbone(in_channels=3, feature_dim=128)
        self.geom_branch = nn.Sequential(
            nn.Linear(geom_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, img: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        v_feat = self.visual_branch(img)
        g_feat = self.geom_branch(geom)
        fused = torch.cat([v_feat, g_feat], dim=1)
        return self.classifier(fused)
