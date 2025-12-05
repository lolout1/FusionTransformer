"""
IMU Transformer with Kalman-derived features.

Architecture for fall detection using Kalman-filtered orientation features:
    - Input: 7 channels [smv, ax, ay, az, roll, pitch, yaw]
    - Dual-stream projection: acc (4ch) + orientation (3ch)
    - Asymmetric capacity: acc_ratio=0.65 (acc gets 65%, ori gets 35%)
    - TransformerEncoder with SE module
    - Temporal Attention Pooling
"""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


class SqueezeExcitation(nn.Module):
    """Channel attention module."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C)"""
        scale = x.mean(dim=1)
        scale = self.fc(scale).unsqueeze(1)
        return x * scale


class TemporalAttentionPooling(nn.Module):
    """Learnable temporal pooling for transient event detection."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """x: (B, T, C) -> (B, C), (B, T)"""
        scores = self.attention(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.einsum('bt,btc->bc', weights, x)
        return context, weights


class TransformerEncoderWithNorm(nn.TransformerEncoder):
    """Transformer encoder with final layer normalization."""

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class KalmanTransformer(nn.Module):
    """
    Transformer for Kalman-filtered IMU features.

    Input channels:
        7ch:  [smv, ax, ay, az, roll, pitch, yaw]
        10ch: [smv, ax, ay, az, roll, pitch, yaw, sigma_r, sigma_p, sigma_y]
        11ch: + innovation_magnitude
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 7,
                 acc_frames: int = 128,
                 acc_coords: int = 7,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 num_classes: int = 1,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 embed_dim: int = 64,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.65,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Channel split for Kalman features
        # 7ch: [smv, ax, ay, az] (4) + [roll, pitch, yaw] (3)
        # 10ch: [smv, ax, ay, az] (4) + [roll, pitch, yaw, sigma_r, sigma_p, sigma_y] (6)
        # 11ch: [smv, ax, ay, az] (4) + [roll, pitch, yaw, sigma_r, sigma_p, sigma_y, innov] (7)
        self.acc_channels = 4  # smv, ax, ay, az
        self.ori_channels = self.imu_channels - 4  # orientation + optional uncertainty/innovation

        # Asymmetric embedding allocation
        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim

        # Accelerometer projection
        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        # Orientation projection (lower capacity, orientation is cleaner than raw gyro)
        self.ori_proj = nn.Sequential(
            nn.Conv1d(self.ori_channels, ori_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(ori_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3)
        )

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )

        self.encoder = TransformerEncoderWithNorm(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # SE module for channel attention
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)

        # Temporal attention pooling
        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize output layer."""
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        """
        Forward pass.

        Args:
            acc_data: (B, T, C) Kalman features
                7ch: [smv, ax, ay, az, roll, pitch, yaw]
            skl_data: Unused, for API compatibility

        Returns:
            logits: (B, num_classes)
            features: (B, T, embed_dim) for distillation
        """
        # Split accelerometer and orientation channels
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        # Rearrange for Conv1d: (B, T, C) -> (B, C, T)
        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        # Project each stream
        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        # Concatenate and normalize: (B, embed_dim, T) -> (B, T, embed_dim)
        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        # Transformer: (B, T, C) -> (T, B, C)
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)

        # Back to batch-first: (T, B, C) -> (B, T, C)
        x = rearrange(x, 't b c -> b t c')

        # SE module
        x = self.se(x)
        features = x  # Save for distillation

        # Temporal pooling
        x, attn_weights = self.temporal_pool(x)

        # Classification
        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


if __name__ == "__main__":
    print("=" * 60)
    print("KalmanTransformer Model Test")
    print("=" * 60)

    # Test 7ch (standard Kalman)
    print("\nTest 1: 7 channels [smv, ax, ay, az, roll, pitch, yaw]")
    print("-" * 50)
    model_7ch = KalmanTransformer(
        imu_frames=128,
        imu_channels=7,
        embed_dim=64,
        num_classes=1,
        num_heads=4,
        num_layers=2,
        dropout=0.5,
        acc_ratio=0.65
    )
    x_7ch = torch.randn(16, 128, 7)
    logits, features = model_7ch(x_7ch)

    total_params = sum(p.numel() for p in model_7ch.parameters())
    print(f"Input shape: {x_7ch.shape}")
    print(f"Acc channels: {model_7ch.acc_channels}, Ori channels: {model_7ch.ori_channels}")
    print(f"Output logits: {logits.shape}")
    print(f"Features: {features.shape}")
    print(f"Parameters: {total_params:,}")

    # Test 10ch (with uncertainty)
    print("\nTest 2: 10 channels (+ uncertainty)")
    print("-" * 50)
    model_10ch = KalmanTransformer(
        imu_frames=128,
        imu_channels=10,
        embed_dim=64,
        num_classes=1
    )
    x_10ch = torch.randn(16, 128, 10)
    logits, features = model_10ch(x_10ch)

    total_params = sum(p.numel() for p in model_10ch.parameters())
    print(f"Input shape: {x_10ch.shape}")
    print(f"Acc channels: {model_10ch.acc_channels}, Ori channels: {model_10ch.ori_channels}")
    print(f"Output logits: {logits.shape}")
    print(f"Parameters: {total_params:,}")

    # Test gradient flow
    print("\nTest 3: Gradient flow check")
    print("-" * 50)
    model_7ch.train()
    x = torch.randn(8, 128, 7, requires_grad=True)
    logits, _ = model_7ch(x)
    loss = logits.sum()
    loss.backward()
    print(f"Input gradient norm: {x.grad.norm():.4f}")
    print("Gradient flow: OK")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
