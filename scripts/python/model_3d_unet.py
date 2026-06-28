"""Minimal geometry-conditioned 3D U-Net baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GeometryConditionedUNet3D(nn.Module):
    """Predict one control-point dose volume from CT and beam condition vector."""

    def __init__(self, in_channels: int = 1, condition_dim: int = 167, base_channels: int = 8) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.bottleneck = ConvBlock(c2, c3)

        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, c3 * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(c3 * 2, c3 * 2),
        )

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)
        self.out = nn.Conv3d(c1, 1, kernel_size=1)

    def forward(self, ct: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(ct)
        e2 = self.enc2(F.max_pool3d(e1, kernel_size=2))
        b = self.bottleneck(F.max_pool3d(e2, kernel_size=2))

        gamma_beta = self.condition_mlp(condition)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        b = b * (1.0 + gamma) + beta

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return F.relu(self.out(d1))
