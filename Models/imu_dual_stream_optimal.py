"""Asymmetric dual-stream IMU transformer tuned for filtered signals."""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, LayerNorm
import torch.nn.functional as F
from einops import rearrange
import math


class OptimalDualStreamIMU(nn.Module):
    """Asymmetric accelerometer/gyroscope encoder with learnable fusion."""

    def __init__(self,
                 imu_frames: int = 128,
                 mocap_frames: int = 128,  # For compatibility
                 num_joints: int = 32,     # For compatibility
                 acc_frames: int = 128,    # Alias for imu_frames
                 imu_channels: int = 8,
                 acc_coords: int = 8,      # Alias for imu_channels
                 num_classes: int = 1,
                 acc_heads: int = 4,       # Stronger attention for reliable acc
                 gyro_heads: int = 2,      # Weaker attention for noisy gyro
                 acc_layers: int = 2,      # Deeper for acc
                 gyro_layers: int = 1,     # Shallow for gyro (prevent overfitting on noise)
                 acc_dim: int = 64,        # Wider for acc (like TransModel)
                 gyro_dim: int = 32,       # Narrower for gyro (less capacity for noise)
                 embed_dim: int = 64,      # Alias for acc_dim
                 num_heads: int = 4,       # Alias for acc_heads
                 num_layers: int = 2,      # Alias for acc_layers
                 dropout: float = 0.6,
                 activation: str = 'relu',
                 norm_first: bool = True,
                 acc_weight: float = 0.7,  # Weight for acc features in fusion
                 gyro_weight: float = 0.3, # Weight for gyro features in fusion
                 **kwargs):
        super().__init__()

        # Handle parameter aliases
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Use provided dimensions or defaults
        self.acc_dim = acc_dim if acc_dim != 64 else embed_dim
        self.gyro_dim = gyro_dim
        self.acc_heads = acc_heads if acc_heads != 4 else num_heads
        self.gyro_heads = gyro_heads
        self.acc_layers = acc_layers if acc_layers != 2 else num_layers
        self.gyro_layers = gyro_layers

        # Asymmetric fusion weights (acc more reliable)
        self.acc_weight = acc_weight
        self.gyro_weight = gyro_weight

        assert self.imu_channels == 8, "OptimalDualStreamIMU requires 8 channels (4 acc + 4 gyro)"

        # Accelerometer stream (high capacity)
        self.acc_input_proj = nn.Sequential(
            nn.Conv1d(4, self.acc_dim, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(self.acc_dim),
            nn.Dropout(dropout * 0.5)
        )

        self.acc_transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.acc_dim,
                nhead=self.acc_heads,
                dim_feedforward=self.acc_dim * 2,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                batch_first=False
            )
            for _ in range(self.acc_layers)
        ])

        self.acc_norm = LayerNorm(self.acc_dim)

        # Gyroscope stream (lightweight)
        self.gyro_input_proj = nn.Sequential(
            nn.Conv1d(4, self.gyro_dim, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(self.gyro_dim),
            nn.Dropout(dropout * 0.7)
        )

        self.gyro_transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.gyro_dim,
                nhead=self.gyro_heads,
                dim_feedforward=self.gyro_dim,
                dropout=dropout * 1.2,
                activation=activation,
                norm_first=norm_first,
                batch_first=False
            )
            for _ in range(self.gyro_layers)
        ])

        self.gyro_norm = LayerNorm(self.gyro_dim)

        fusion_dim = self.acc_dim + self.gyro_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, acc_data, skl_data=None, **kwargs):
        """Return logits and fused features for eight-channel IMU input."""
        acc = acc_data[:, :, :4]
        gyro = acc_data[:, :, 4:]

        acc_x = rearrange(acc, 'b t c -> b c t')
        acc_x = self.acc_input_proj(acc_x)
        acc_x = rearrange(acc_x, 'b c t -> t b c')
        for layer in self.acc_transformer_layers:
            acc_x = layer(acc_x)
        acc_feat = torch.mean(self.acc_norm(acc_x), dim=0)

        gyro_x = rearrange(gyro, 'b t c -> b c t')
        gyro_x = self.gyro_input_proj(gyro_x)
        gyro_x = rearrange(gyro_x, 'b c t -> t b c')
        for layer in self.gyro_transformer_layers:
            gyro_x = layer(gyro_x)
        gyro_feat = torch.mean(self.gyro_norm(gyro_x), dim=0)

        total_weight = self.acc_weight + self.gyro_weight
        acc_weight = self.acc_weight / total_weight
        gyro_weight = self.gyro_weight / total_weight

        features = torch.cat(
            [acc_feat * acc_weight, gyro_feat * gyro_weight],
            dim=-1,
        )
        logits = self.fusion(features)
        return logits, features


if __name__ == "__main__":
    batch_size = 16
    seq_len = 128
    imu_channels = 8

    imu_data = torch.randn(batch_size, seq_len, imu_channels)

    model = OptimalDualStreamIMU(
        imu_frames=seq_len,
        imu_channels=imu_channels,
        num_classes=1,
        acc_heads=4,
        gyro_heads=2,
        acc_layers=2,
        gyro_layers=1,
        acc_dim=64,
        gyro_dim=32,
        dropout=0.6,
        acc_weight=0.7,
        gyro_weight=0.3
    )

    logits, features = model(imu_data)

    print(f"Input shape: {imu_data.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nParameter breakdown:")
    print(f"  Acc encoder: {sum(p.numel() for p in model.acc_input_proj.parameters()):,}")
    print(f"  Gyro encoder: {sum(p.numel() for p in model.gyro_input_proj.parameters()):,}")
    print(f"  Acc transformer ({model.acc_layers} layers): {sum(p.numel() for p in model.acc_transformer_layers.parameters()):,}")
    print(f"  Gyro transformer ({model.gyro_layers} layers): {sum(p.numel() for p in model.gyro_transformer_layers.parameters()):,}")
    print(f"  Fusion layer: {sum(p.numel() for p in model.fusion.parameters()):,}")

    print("\nArchitecture Summary:")
    print(f"  Acc stream: {model.acc_layers} layers, {model.acc_heads} heads, {model.acc_dim}d (TransModel-like)")
    print(f"  Gyro stream: {model.gyro_layers} layers, {model.gyro_heads} heads, {model.gyro_dim}d (Lightweight)")
    print(f"  Fusion weights: Acc={model.acc_weight}, Gyro={model.gyro_weight}")
