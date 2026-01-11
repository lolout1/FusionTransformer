"""Flexible single/dual stream transformers for ablation studies."""

import torch
import torch.nn as nn
import math


class FlexibleSingleStream(nn.Module):
    """Single-stream transformer with configurable input channels."""

    def __init__(self, imu_frames=128, imu_channels=7, num_heads=4, num_layers=2,
                 embed_dim=64, dropout=0.5, activation='gelu', norm_first=True,
                 se_reduction=4, use_se=True, use_tap=True, **kwargs):
        super().__init__()
        self.imu_channels = imu_channels
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Conv1d(imu_channels, embed_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        self.input_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2,
            dropout=dropout, activation=activation, norm_first=norm_first, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.se = SEBlock(embed_dim, se_reduction) if use_se else nn.Identity()
        self.pool = TemporalAttentionPool(embed_dim, dropout) if use_tap else GlobalAvgPool()

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, acc=None, imu=None, **kwargs):
        x = imu if imu is not None else acc
        if x is None:
            raise ValueError("Input required")
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x[:, :self.imu_channels, :]

        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.se(x)
        x = self.pool(x)
        return self.classifier(x)


class FlexibleDualStream(nn.Module):
    """Dual-stream transformer with configurable channel split."""

    def __init__(self, imu_frames=128, imu_channels=7, acc_channels=4, num_heads=4,
                 num_layers=2, embed_dim=64, dropout=0.5, activation='gelu',
                 norm_first=True, se_reduction=4, use_se=True, use_tap=True,
                 acc_ratio=0.5, **kwargs):
        super().__init__()
        self.acc_channels = acc_channels
        self.aux_channels = imu_channels - acc_channels

        acc_dim = int(embed_dim * acc_ratio)
        aux_dim = embed_dim - acc_dim

        self.acc_proj = nn.Sequential(
            nn.Conv1d(acc_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim), nn.SiLU(), nn.Dropout(dropout * 0.2)
        )
        self.aux_proj = nn.Sequential(
            nn.Conv1d(self.aux_channels, aux_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(aux_dim), nn.SiLU(), nn.Dropout(dropout * 0.2)
        )

        self.input_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2,
            dropout=dropout, activation=activation, norm_first=norm_first, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.se = SEBlock(embed_dim, se_reduction) if use_se else nn.Identity()
        self.pool = TemporalAttentionPool(embed_dim, dropout) if use_tap else GlobalAvgPool()

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, acc=None, imu=None, **kwargs):
        x = imu if imu is not None else acc
        if x is None:
            raise ValueError("Input required")
        if x.dim() == 4:
            x = x.squeeze(1)

        acc_x = x[:, :self.acc_channels, :]
        aux_x = x[:, self.acc_channels:self.acc_channels + self.aux_channels, :]

        acc_feat = self.acc_proj(acc_x)
        aux_feat = self.aux_proj(aux_x)
        x = torch.cat([acc_feat, aux_feat], dim=1)

        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.se(x)
        x = self.pool(x)
        return self.classifier(x)


class SEBlock(nn.Module):
    """Squeeze-Excitation block."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.shape
        w = x.mean(dim=2)
        w = self.fc(w).unsqueeze(2)
        return x * w


class TemporalAttentionPool(nn.Module):
    """Attention-based temporal pooling."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=2)
