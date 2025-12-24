"""
Dual-Input IMU Transformer with SE Module and Temporal Attention Pooling.

Architecture:
    - Input: 6 or 8 channels
        - 6 channels: [ax, ay, az, gx, gy, gz] -> 3 acc, 3 gyro
        - 8 channels: [smv, ax, ay, az, gyro_mag, gx, gy, gz] -> 4 acc, 4 gyro
    - Separate Conv1d projections for acc and gyro with asymmetric capacity
    - Feature fusion with LayerNorm
    - TransformerEncoder (2 layers, 4 heads)
    - Squeeze-and-Excitation channel attention
    - Temporal Attention Pooling (learnable temporal aggregation)
    - Linear classifier

Design rationale:
    - Asymmetric capacity: acc gets 75%, gyro gets 25% of embedding dim
    - Higher dropout on gyro (0.4) vs acc (0.2) to handle noise
    - Separate BatchNorms before fusion for modality normalization
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
        """Apply channel attention. x: (B, T, C)"""
        scale = x.mean(dim=1)  # (B, C)
        scale = self.fc(scale).unsqueeze(1)  # (B, 1, C)
        return x * scale


class TemporalAttentionPooling(nn.Module):
    """Learnable temporal pooling."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted pooling. x: (B, T, C)"""
        scores = self.attention(x).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1)  # (B, T)
        context = torch.einsum('bt,btc->bc', weights, x)  # (B, C)
        return context, weights


class TransformerEncoderWithNorm(nn.TransformerEncoder):
    """Standard transformer encoder with final normalization."""

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class IMUTransformerSE(nn.Module):
    """Dual-input IMU transformer with SE and Temporal Attention Pooling."""

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 6,
                 acc_frames: int = 128,
                 acc_coords: int = 6,
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
                 acc_ratio: float = 0.75,
                 **kwargs):
        super().__init__()

        # Handle parameter aliasing for compatibility
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Determine input channel split based on total channels
        # 8 channels: [smv, ax, ay, az, gyro_mag, gx, gy, gz] -> 4 acc, 4 gyro
        # 6 channels: [ax, ay, az, gx, gy, gz] -> 3 acc, 3 gyro
        if self.imu_channels == 8:
            self.acc_in_channels = 4
            self.gyro_in_channels = 4
        elif self.imu_channels == 6:
            self.acc_in_channels = 3
            self.gyro_in_channels = 3
        else:
            # Flexible: split evenly
            self.acc_in_channels = self.imu_channels // 2
            self.gyro_in_channels = self.imu_channels - self.acc_in_channels

        # Asymmetric embedding allocation
        acc_dim = int(embed_dim * acc_ratio)
        gyro_dim = embed_dim - acc_dim

        # Separate input projections for acc and gyro
        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_in_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)  # Lower dropout for acc
        )

        self.gyro_proj = nn.Sequential(
            nn.Conv1d(self.gyro_in_channels, gyro_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(gyro_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.4)  # Higher dropout for noisy gyro
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

        # Attention modules
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize output layer with scaled normal distribution."""
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        """
        Forward pass.

        Args:
            acc_data: (B, T, 8) IMU data [smv, ax, ay, az, gyro_mag, gx, gy, gz]
            skl_data: Unused, for API compatibility

        Returns:
            logits: (B, num_classes) classification logits
            features: (B, T, embed_dim) encoder features before pooling
        """
        # Split modalities based on channel configuration
        acc = acc_data[:, :, :self.acc_in_channels]  # (B, T, acc_in)
        gyro = acc_data[:, :, self.acc_in_channels:self.acc_in_channels + self.gyro_in_channels]  # (B, T, gyro_in)

        # Rearrange for Conv1d: (B, T, C) -> (B, C, T)
        acc = rearrange(acc, 'b t c -> b c t')
        gyro = rearrange(gyro, 'b t c -> b c t')

        # Separate projections
        acc_feat = self.acc_proj(acc)  # (B, acc_dim, T)
        gyro_feat = self.gyro_proj(gyro)  # (B, gyro_dim, T)

        # Concatenate and normalize: (B, embed_dim, T) -> (B, T, embed_dim)
        x = torch.cat([acc_feat, gyro_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        # Transformer: (B, T, C) -> (T, B, C)
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)

        # SE module: (T, B, C) -> (B, T, C)
        x = rearrange(x, 't b c -> b t c')
        x = self.se(x)
        features = x  # Save for potential feature extraction

        # Temporal attention pooling: (B, T, C) -> (B, C)
        x, attn_weights = self.temporal_pool(x)

        # Classification
        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


if __name__ == "__main__":
    print("=" * 60)
    print("IMUTransformerSE Model Architecture Test")
    print("=" * 60)

    # Test with 6 channels (no SMV)
    print("\n" + "=" * 50)
    print("Test 1: 6 channels (ax, ay, az, gx, gy, gz)")
    print("=" * 50)
    model_6ch = IMUTransformerSE(imu_frames=128, imu_channels=6, embed_dim=64, num_classes=1)
    x_6ch = torch.randn(16, 128, 6)
    logits, features = model_6ch(x_6ch)

    total_params = sum(p.numel() for p in model_6ch.parameters())
    print(f"Input shape: {x_6ch.shape}")
    print(f"Acc channels: {model_6ch.acc_in_channels}, Gyro channels: {model_6ch.gyro_in_channels}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Total parameters: {total_params:,}")

    # Test with 8 channels (with SMV)
    print("\n" + "=" * 50)
    print("Test 2: 8 channels (smv, ax, ay, az, gyro_mag, gx, gy, gz)")
    print("=" * 50)
    model_8ch = IMUTransformerSE(imu_frames=128, imu_channels=8, embed_dim=64, num_classes=1)
    x_8ch = torch.randn(16, 128, 8)
    logits, features = model_8ch(x_8ch)

    total_params = sum(p.numel() for p in model_8ch.parameters())
    print(f"Input shape: {x_8ch.shape}")
    print(f"Acc channels: {model_8ch.acc_in_channels}, Gyro channels: {model_8ch.gyro_in_channels}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
