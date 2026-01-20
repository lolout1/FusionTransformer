"""
AccGyroKalman: Dual-stream model with Acc+Orientation and Raw Gyro streams.

Architecture:
    Stream 1: [smv, ax, ay, az, roll, pitch] (6ch) - Accelerometer + Kalman orientation
    Stream 2: [gx, gy, gz] (3ch) - Raw gyroscope

Rationale:
    - Preserves raw gyro high-frequency angular velocity information
    - Roll/pitch are reliably computed from accelerometer gravity reference
    - Yaw is excluded (noisier, drift-prone from gyro integration)
    - Different encoders for different signal characteristics

Input: 9ch [smv, ax, ay, az, roll, pitch, gx, gy, gz]
    - Requires: kalman_exclude_yaw=True, kalman_include_raw_gyro=True

Usage:
    model = AccGyroKalmanTransformer(imu_frames=128, imu_channels=9, embed_dim=48)
"""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
from typing import Literal, Optional


# =============================================================================
# Shared Components
# =============================================================================

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


class Conv1DEncoder(nn.Module):
    """Conv1D-based temporal encoder for high-frequency signals."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C_in) -> (B, T, C_out)"""
        x = rearrange(x, 'b t c -> b c t')
        x = self.encoder(x)
        x = rearrange(x, 'b c t -> b t c')
        return x


class LinearEncoder(nn.Module):
    """Linear per-timestep encoder for smooth signals."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C_in) -> (B, T, C_out)"""
        return self.encoder(x)


# =============================================================================
# AccGyroKalman Model
# =============================================================================

class AccGyroKalmanTransformer(nn.Module):
    """
    Dual-stream Transformer with Acc+Orientation and Raw Gyro streams.

    Input: 9ch [smv, ax, ay, az, roll, pitch, gx, gy, gz]

    Stream 1 (Acc+Ori): [smv, ax, ay, az, roll, pitch] (6ch)
        - Conv1D encoder to capture local temporal patterns in acceleration
        - Includes Kalman-filtered orientation (roll, pitch)

    Stream 2 (Gyro): [gx, gy, gz] (3ch)
        - Conv1D encoder to preserve high-frequency angular velocity
        - Raw gyro captures rapid rotational dynamics during falls

    Architecture:
        Acc+Ori (6ch) -> Conv1D -> acc_ori_dim
        Gyro (3ch)    -> Conv1D -> gyro_dim
        Concat -> LayerNorm -> Transformer -> SE -> TAP -> Classifier
    """

    def __init__(
        self,
        # Input configuration
        imu_frames: int = 128,
        imu_channels: int = 9,
        # Stream configuration
        acc_ori_channels: int = 6,  # [smv, ax, ay, az, roll, pitch]
        gyro_channels: int = 3,     # [gx, gy, gz]
        # Encoder configuration
        acc_ori_encoder: Literal['conv1d', 'linear'] = 'conv1d',
        gyro_encoder: Literal['conv1d', 'linear'] = 'conv1d',
        acc_ori_kernel_size: int = 8,
        gyro_kernel_size: int = 5,  # Smaller for high-freq gyro
        # Model configuration
        num_classes: int = 1,
        num_heads: int = 4,
        num_layers: int = 2,
        embed_dim: int = 48,
        dropout: float = 0.5,
        activation: str = 'relu',
        norm_first: bool = True,
        se_reduction: int = 4,
        acc_ori_ratio: float = 0.70,  # 70% for acc+ori, 30% for gyro
        # Legacy compatibility
        acc_frames: int = None,
        acc_coords: int = None,
        mocap_frames: int = None,
        num_joints: int = None,
        **kwargs
    ):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Channel split
        self.acc_ori_channels = acc_ori_channels
        self.gyro_channels = gyro_channels

        # Validate input channels
        expected = acc_ori_channels + gyro_channels
        if self.imu_channels != expected:
            raise ValueError(
                f"imu_channels ({self.imu_channels}) != acc_ori_channels ({acc_ori_channels}) "
                f"+ gyro_channels ({gyro_channels}) = {expected}"
            )

        # Asymmetric embedding allocation
        acc_ori_dim = int(embed_dim * acc_ori_ratio)
        gyro_dim = embed_dim - acc_ori_dim
        self.acc_ori_dim = acc_ori_dim
        self.gyro_dim = gyro_dim

        # Stream 1: Acc+Orientation encoder
        if acc_ori_encoder == 'conv1d':
            self.acc_ori_proj = Conv1DEncoder(
                in_channels=acc_ori_channels,
                out_channels=acc_ori_dim,
                kernel_size=acc_ori_kernel_size,
                dropout=dropout * 0.2
            )
        else:
            self.acc_ori_proj = LinearEncoder(
                in_channels=acc_ori_channels,
                out_channels=acc_ori_dim,
                dropout=dropout * 0.2
            )

        # Stream 2: Gyroscope encoder
        if gyro_encoder == 'conv1d':
            self.gyro_proj = Conv1DEncoder(
                in_channels=gyro_channels,
                out_channels=gyro_dim,
                kernel_size=gyro_kernel_size,
                dropout=dropout * 0.3
            )
        else:
            self.gyro_proj = LinearEncoder(
                in_channels=gyro_channels,
                out_channels=gyro_dim,
                dropout=dropout * 0.3
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

        # SE module
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)

        # Temporal attention pooling
        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
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
        """
        Forward pass.

        Args:
            acc_data: (B, T, 9) input tensor [smv, ax, ay, az, roll, pitch, gx, gy, gz]
            skl_data: Ignored (skeleton not used)

        Returns:
            logits: (B, num_classes) classification logits
            features: (B, T, embed_dim) encoded features (for visualization)
        """
        # Split into streams
        # Stream 1: Acc + Orientation [smv, ax, ay, az, roll, pitch] (first 6 channels)
        acc_ori = acc_data[:, :, :self.acc_ori_channels]

        # Stream 2: Gyroscope [gx, gy, gz] (last 3 channels)
        gyro = acc_data[:, :, self.acc_ori_channels:]

        # Encode each stream
        acc_ori_encoded = self.acc_ori_proj(acc_ori)  # (B, T, acc_ori_dim)
        gyro_encoded = self.gyro_proj(gyro)           # (B, T, gyro_dim)

        # Concatenate and normalize
        fused = torch.cat([acc_ori_encoded, gyro_encoded], dim=-1)  # (B, T, embed_dim)
        fused = self.fusion_norm(fused)

        # Transformer expects (T, B, C)
        x = rearrange(fused, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        # SE attention
        x = self.se(x)

        # Temporal pooling
        context, attention_weights = self.temporal_pool(x)

        # Classification
        context = self.dropout(context)
        logits = self.output(context)

        return logits, x


class AccGyroKalmanCNN(nn.Module):
    """
    CNN-only version of AccGyroKalman (no transformer).

    Simpler architecture that may work better on smaller datasets.

    Input: 9ch [smv, ax, ay, az, roll, pitch, gx, gy, gz]
    """

    def __init__(
        self,
        imu_frames: int = 128,
        imu_channels: int = 9,
        acc_ori_channels: int = 6,
        gyro_channels: int = 3,
        embed_dim: int = 48,
        dropout: float = 0.5,
        se_reduction: int = 4,
        acc_ori_ratio: float = 0.70,
        acc_ori_kernel_size: int = 8,
        gyro_kernel_size: int = 5,
        num_classes: int = 1,
        # Legacy compatibility
        acc_frames: int = None,
        acc_coords: int = None,
        **kwargs
    ):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.acc_ori_channels = acc_ori_channels
        self.gyro_channels = gyro_channels

        # Embedding allocation
        acc_ori_dim = int(embed_dim * acc_ori_ratio)
        gyro_dim = embed_dim - acc_ori_dim
        self.acc_ori_dim = acc_ori_dim
        self.gyro_dim = gyro_dim

        # Stream 1: Acc+Orientation
        self.acc_ori_proj = Conv1DEncoder(
            in_channels=acc_ori_channels,
            out_channels=acc_ori_dim,
            kernel_size=acc_ori_kernel_size,
            dropout=dropout * 0.2
        )

        # Stream 2: Gyroscope
        self.gyro_proj = Conv1DEncoder(
            in_channels=gyro_channels,
            out_channels=gyro_dim,
            kernel_size=gyro_kernel_size,
            dropout=dropout * 0.3
        )

        # Fusion
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        # Split streams
        acc_ori = acc_data[:, :, :self.acc_ori_channels]
        gyro = acc_data[:, :, self.acc_ori_channels:]

        # Encode
        acc_ori_encoded = self.acc_ori_proj(acc_ori)
        gyro_encoded = self.gyro_proj(gyro)

        # Fuse
        fused = torch.cat([acc_ori_encoded, gyro_encoded], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.se(fused)

        # Pool and classify
        context, _ = self.temporal_pool(fused)
        context = self.dropout(context)
        logits = self.output(context)

        return logits, fused


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == '__main__':
    print("Testing AccGyroKalman models...")

    # Test Transformer version
    model = AccGyroKalmanTransformer(imu_frames=128, imu_channels=9, embed_dim=48)
    x = torch.randn(4, 128, 9)  # [smv, ax, ay, az, roll, pitch, gx, gy, gz]
    logits, features = model(x)
    print(f"Transformer: Input {x.shape} -> Logits {logits.shape}, Features {features.shape}")
    print(f"  acc_ori_dim={model.acc_ori_dim}, gyro_dim={model.gyro_dim}")

    # Test CNN version
    model_cnn = AccGyroKalmanCNN(imu_frames=128, imu_channels=9, embed_dim=48)
    logits_cnn, features_cnn = model_cnn(x)
    print(f"CNN: Input {x.shape} -> Logits {logits_cnn.shape}, Features {features_cnn.shape}")

    # Parameter count
    n_params_trans = sum(p.numel() for p in model.parameters())
    n_params_cnn = sum(p.numel() for p in model_cnn.parameters())
    print(f"Parameters: Transformer={n_params_trans:,}, CNN={n_params_cnn:,}")

    print("All tests passed!")
