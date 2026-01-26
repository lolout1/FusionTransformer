"""Dual-stream state-space model for IMU."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock(nn.Module):

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        se = x.mean(dim=1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se.unsqueeze(1)


class StateSpaceBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_conv: int = 4,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_inner = d_model * expand

        self.conv = nn.Conv1d(
            d_model, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=1
        )

        self.gru = nn.GRU(
            d_inner, d_state,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        self.out_proj = nn.Linear(d_state * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :residual.shape[1]]
        x = F.silu(x)
        x = x.transpose(1, 2)

        x, _ = self.gru(x)

        x = self.out_proj(x)
        x = self.dropout(x)

        return self.norm(x + residual)


class DualStreamMamba(nn.Module):

    def __init__(
        self,
        acc_frames: int = 128,
        acc_coords: int = 4,
        gyro_coords: int = 3,
        imu_frames: int = 128,
        imu_channels: int = 7,
        num_classes: int = 1,
        d_model: int = 48,
        embed_dim: int = 48,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 2,
        acc_ratio: float = 0.75,
        dropout: float = 0.3,
        use_se: bool = True,
        se_reduction: int = 4,
        num_heads: int = 4,
        num_layers_compat: int = 2,
        activation: str = 'relu',
        norm_first: bool = True,
        **kwargs
    ):
        super().__init__()

        d_model = embed_dim if embed_dim != 48 else d_model
        acc_frames = imu_frames if imu_frames != 128 else acc_frames

        acc_dim = int(d_model * acc_ratio)
        gyro_dim = d_model - acc_dim

        self.acc_coords = acc_coords
        self.gyro_coords = gyro_coords
        self.d_model = d_model
        self.acc_dim = acc_dim
        self.gyro_dim = gyro_dim

        self.acc_proj = nn.Sequential(
            nn.Conv1d(acc_coords, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.SiLU()
        )

        self.gyro_proj = nn.Sequential(
            nn.Conv1d(gyro_coords, gyro_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(gyro_dim),
            nn.SiLU()
        )

        self.acc_ssm = nn.ModuleList([
            StateSpaceBlock(acc_dim, d_conv, d_state, expand, dropout)
            for _ in range(num_layers)
        ])

        self.gyro_ssm = nn.ModuleList([
            StateSpaceBlock(gyro_dim, d_conv, d_state, expand, dropout)
            for _ in range(num_layers)
        ])

        self.use_se = use_se
        if use_se:
            self.acc_se = SEBlock(acc_dim, se_reduction)
            self.gyro_se = SEBlock(gyro_dim, se_reduction)

        self.fusion_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        x = acc_data
        if isinstance(x, dict):
            x = x['accelerometer']

        acc = x[..., :self.acc_coords]
        gyro = x[..., self.acc_coords:self.acc_coords + self.gyro_coords]

        acc = self.acc_proj(acc.transpose(1, 2)).transpose(1, 2)
        gyro = self.gyro_proj(gyro.transpose(1, 2)).transpose(1, 2)

        for ssm in self.acc_ssm:
            acc = ssm(acc)

        for ssm in self.gyro_ssm:
            gyro = ssm(gyro)

        if self.use_se:
            acc = self.acc_se(acc)
            gyro = self.gyro_se(gyro)

        acc = acc.mean(dim=1)
        gyro = gyro.mean(dim=1)

        fused = torch.cat([acc, gyro], dim=-1)
        fused = self.fusion_norm(fused)

        out = self.classifier(fused)

        return out, None
