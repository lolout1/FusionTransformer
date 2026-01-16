"""
Dual-Stream State-Space Model for IMU Fall Detection.

A pure PyTorch implementation inspired by Mamba/S4 architectures.
Uses Conv1d + GRU blocks as a practical state-space alternative.

Architecture:
    - Acc stream: Conv1d → StateSpaceBlock → SE → Pool
    - Gyro stream: Conv1d → StateSpaceBlock → SE → Pool
    - Fusion: Concat → FC → Sigmoid

This provides O(n) complexity like Mamba while being compatible
with standard PyTorch installations (no mamba-ssm required).

References:
    - Gu et al. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    - Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, T, C)
        se = x.mean(dim=1)  # (B, C)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se.unsqueeze(1)


class StateSpaceBlock(nn.Module):
    """
    Simplified state-space block using Conv1d + GRU.

    Captures both local (conv) and long-range (recurrent) dependencies
    with O(n) complexity, similar to Mamba but without CUDA kernels.
    """

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

        # Local convolution for short-range patterns
        self.conv = nn.Conv1d(
            d_model, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=1
        )

        # State-space via GRU (bidirectional for better context)
        self.gru = nn.GRU(
            d_inner, d_state,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        # Output projection
        self.out_proj = nn.Linear(d_state * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        residual = x

        # Conv for local patterns
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv(x)[:, :, :residual.shape[1]]  # Trim to original length
        x = F.silu(x)
        x = x.transpose(1, 2)  # (B, T, d_inner)

        # GRU for state-space modeling
        x, _ = self.gru(x)  # (B, T, d_state*2)

        # Project back
        x = self.out_proj(x)
        x = self.dropout(x)

        # Residual connection
        return self.norm(x + residual)


class DualStreamMamba(nn.Module):
    """
    Dual-Stream Mamba/S4-inspired model for IMU fall detection.

    Uses separate state-space streams for accelerometer and gyroscope,
    with asymmetric capacity allocation (acc > gyro).

    Input format: (B, T, C) where C = acc_coords + gyro_coords
        - For Kalman features: C = 7 (SMV, ax, ay, az, roll, pitch, yaw)
        - For raw features: C = 7 (SMV, ax, ay, az, gx, gy, gz)

    Args:
        acc_frames: Sequence length (default: 128)
        acc_coords: Accelerometer channels including SMV (default: 4)
        gyro_coords: Gyroscope/orientation channels (default: 3)
        num_classes: Output classes (default: 1 for binary)
        d_model: Total embedding dimension (default: 48)
        d_state: State-space hidden dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor for inner dimension (default: 2)
        num_layers: Number of state-space blocks (default: 2)
        acc_ratio: Fraction of d_model for acc stream (default: 0.75)
        dropout: Dropout rate (default: 0.3)
        use_se: Enable squeeze-excitation (default: True)
    """

    def __init__(
        self,
        acc_frames: int = 128,
        acc_coords: int = 4,
        gyro_coords: int = 3,
        imu_frames: int = 128,  # Alias for compatibility
        imu_channels: int = 7,  # Alias for compatibility
        num_classes: int = 1,
        d_model: int = 48,
        embed_dim: int = 48,  # Alias for compatibility
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 2,
        acc_ratio: float = 0.75,
        dropout: float = 0.3,
        use_se: bool = True,
        se_reduction: int = 4,
        # Compatibility parameters (ignored)
        num_heads: int = 4,
        num_layers_compat: int = 2,
        activation: str = 'relu',
        norm_first: bool = True,
        **kwargs
    ):
        super().__init__()

        # Handle parameter aliases
        d_model = embed_dim if embed_dim != 48 else d_model
        acc_frames = imu_frames if imu_frames != 128 else acc_frames

        # Asymmetric capacity allocation
        acc_dim = int(d_model * acc_ratio)
        gyro_dim = d_model - acc_dim

        self.acc_coords = acc_coords
        self.gyro_coords = gyro_coords
        self.d_model = d_model
        self.acc_dim = acc_dim
        self.gyro_dim = gyro_dim

        # Input projections
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

        # State-space blocks for each stream
        self.acc_ssm = nn.ModuleList([
            StateSpaceBlock(acc_dim, d_conv, d_state, expand, dropout)
            for _ in range(num_layers)
        ])

        self.gyro_ssm = nn.ModuleList([
            StateSpaceBlock(gyro_dim, d_conv, d_state, expand, dropout)
            for _ in range(num_layers)
        ])

        # SE blocks
        self.use_se = use_se
        if use_se:
            self.acc_se = SEBlock(acc_dim, se_reduction)
            self.gyro_se = SEBlock(gyro_dim, se_reduction)

        # Fusion and classification
        self.fusion_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

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
            acc_data: Input tensor of shape (B, T, C) where C = acc_coords + gyro_coords
                      Can also accept dict with 'accelerometer' key for compatibility
            skl_data: Ignored (skeleton not used for external datasets)
            **kwargs: Additional arguments (ignored)

        Returns:
            Tuple of (logits, features) for compatibility with main.py training loop
        """
        x = acc_data
        # Handle dict input for compatibility
        if isinstance(x, dict):
            x = x['accelerometer']

        # Split into acc and gyro streams
        acc = x[..., :self.acc_coords]  # (B, T, acc_coords)
        gyro = x[..., self.acc_coords:self.acc_coords + self.gyro_coords]  # (B, T, gyro_coords)

        # Project inputs
        acc = self.acc_proj(acc.transpose(1, 2)).transpose(1, 2)  # (B, T, acc_dim)
        gyro = self.gyro_proj(gyro.transpose(1, 2)).transpose(1, 2)  # (B, T, gyro_dim)

        # Apply state-space blocks
        for ssm in self.acc_ssm:
            acc = ssm(acc)

        for ssm in self.gyro_ssm:
            gyro = ssm(gyro)

        # SE attention
        if self.use_se:
            acc = self.acc_se(acc)
            gyro = self.gyro_se(gyro)

        # Global average pooling
        acc = acc.mean(dim=1)  # (B, acc_dim)
        gyro = gyro.mean(dim=1)  # (B, gyro_dim)

        # Fuse streams
        fused = torch.cat([acc, gyro], dim=-1)  # (B, d_model)
        fused = self.fusion_norm(fused)

        # Classify
        out = self.classifier(fused)

        return out, None  # Return tuple for main.py compatibility


class DualStreamMambaLight(nn.Module):
    """
    Lightweight version of DualStreamMamba with fewer parameters.

    Uses single-layer GRU and no SE blocks for minimal footprint.
    """

    def __init__(
        self,
        acc_frames: int = 128,
        acc_coords: int = 4,
        gyro_coords: int = 3,
        num_classes: int = 1,
        d_model: int = 32,
        d_state: int = 8,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()

        acc_dim = int(d_model * 0.75)
        gyro_dim = d_model - acc_dim

        self.acc_coords = acc_coords
        self.gyro_coords = gyro_coords

        # Simple conv projection
        self.acc_proj = nn.Conv1d(acc_coords, acc_dim, kernel_size=5, padding='same')
        self.gyro_proj = nn.Conv1d(gyro_coords, gyro_dim, kernel_size=5, padding='same')

        # Single GRU layer per stream
        self.acc_gru = nn.GRU(acc_dim, d_state, batch_first=True, bidirectional=True)
        self.gyro_gru = nn.GRU(gyro_dim, d_state, batch_first=True, bidirectional=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_state * 4, num_classes)
        )

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        x = acc_data
        if isinstance(x, dict):
            x = x['accelerometer']

        acc = x[..., :self.acc_coords]
        gyro = x[..., self.acc_coords:self.acc_coords + self.gyro_coords]

        acc = F.silu(self.acc_proj(acc.transpose(1, 2)).transpose(1, 2))
        gyro = F.silu(self.gyro_proj(gyro.transpose(1, 2)).transpose(1, 2))

        acc, _ = self.acc_gru(acc)
        gyro, _ = self.gyro_gru(gyro)

        acc = acc.mean(dim=1)
        gyro = gyro.mean(dim=1)

        fused = torch.cat([acc, gyro], dim=-1)
        return self.classifier(fused), None


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DualStreamMamba Test")
    print("=" * 60)

    # Test standard model
    model = DualStreamMamba(
        acc_coords=4,
        gyro_coords=3,
        d_model=48,
        num_layers=2,
    )

    x = torch.randn(4, 128, 7)  # (B, T, C)
    out, _ = model(x)

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nDualStreamMamba:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {params:,}")
    print(f"  Trainable: {trainable:,}")

    # Test lightweight model
    model_light = DualStreamMambaLight()
    out_light, _ = model_light(x)

    params_light = sum(p.numel() for p in model_light.parameters())

    print(f"\nDualStreamMambaLight:")
    print(f"  Output shape: {out_light.shape}")
    print(f"  Parameters: {params_light:,}")

    # Test with dict input
    x_dict = {'accelerometer': x}
    out_dict, _ = model(x_dict)
    print(f"\nDict input test: {out_dict.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
