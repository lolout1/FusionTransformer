"""
Dual-Stream Base Transformer

Separate projections for accelerometer and gyroscope, NO SE or temporal attention.
Used for ablation study to isolate the effect of dual-stream from SE/TAP.

Architecture:
    ACC [3ch] ──► Conv1d ──► acc_feat (48d)
                                            ├──► Concat ──► Transformer ──► GAP ──► Classifier
    GYRO [3ch] ──► Conv1d ──► gyro_feat (16d)

Key Design Choices:
- Asymmetric capacity allocation: 75% ACC (48d), 25% GYRO (16d)
- Higher dropout on gyro path (gyroscope is noisier)
- Global Average Pooling (NOT temporal attention - that's for ablation)
- No SE blocks (that's for ablation)

This is Model B and D in the ablation study (without SE+TAP).
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange
from typing import Optional, Tuple


class DualStreamBase(nn.Module):
    """
    Dual-stream transformer WITHOUT SE or temporal attention.

    Separates accelerometer and gyroscope into two streams with asymmetric
    capacity allocation, then fuses for transformer processing.

    Args:
        imu_frames: Sequence length (default 128)
        imu_channels: Total input channels (default 6, split as 3+3)
        acc_dim: Accelerometer embedding dimension (default 48, 75% of total)
        gyro_dim: Gyroscope embedding dimension (default 16, 25% of total)
        num_heads: Number of attention heads (default 4)
        num_layers: Number of transformer layers (default 2)
        dropout: Base dropout rate (default 0.5)
        acc_dropout_mult: Multiplier for acc dropout (default 0.2)
        gyro_dropout_mult: Multiplier for gyro dropout (default 0.4)
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 6,
                 acc_dim: int = 48,
                 gyro_dim: int = 16,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 acc_dropout_mult: float = 0.2,
                 gyro_dropout_mult: float = 0.4,
                 activation: str = 'silu',
                 norm_first: bool = True,
                 use_pos_encoding: bool = True,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames
        self.acc_dim = acc_dim
        self.gyro_dim = gyro_dim
        self.embed_dim = acc_dim + gyro_dim
        self.use_pos_encoding = use_pos_encoding

        # Activation function
        act_fn = nn.SiLU() if activation.lower() == 'silu' else nn.ReLU()

        # Accelerometer projection (3 channels -> acc_dim)
        # Lower dropout for cleaner accelerometer signal
        self.acc_proj = nn.Sequential(
            nn.Conv1d(3, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            act_fn,
            nn.Dropout(dropout * acc_dropout_mult)
        )

        # Gyroscope projection (3 channels -> gyro_dim)
        # Higher dropout for noisier gyroscope signal
        self.gyro_proj = nn.Sequential(
            nn.Conv1d(3, gyro_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(gyro_dim),
            act_fn,
            nn.Dropout(dropout * gyro_dropout_mult)
        )

        # Optional positional encoding for fused features
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, imu_frames, self.embed_dim) * 0.02
            )
        else:
            self.pos_encoding = None

        # Shared transformer encoder after fusion
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            norm_first=norm_first,
            batch_first=False
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers (using Global Average Pooling, NOT temporal attention)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.embed_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                acc_data: torch.Tensor,
                skl_data: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            acc_data: (B, T, 6) IMU data where channels are [ax,ay,az,gx,gy,gz]
            skl_data: Ignored (for API compatibility)

        Returns:
            logits: (B, 1) classification logits
            features: (B, embed_dim) features before classifier
        """
        # Split into accelerometer and gyroscope
        # acc_data: (B, T, 6) = [ax, ay, az, gx, gy, gz]
        B, T, C = acc_data.shape

        acc = acc_data[:, :, :3]   # (B, T, 3) accelerometer
        gyro = acc_data[:, :, 3:]  # (B, T, 3) gyroscope

        # Rearrange for Conv1d: (B, C, T)
        acc = rearrange(acc, 'b t c -> b c t')
        gyro = rearrange(gyro, 'b t c -> b c t')

        # Project each stream with separate projections
        acc_feat = self.acc_proj(acc)    # (B, acc_dim, T)
        gyro_feat = self.gyro_proj(gyro)  # (B, gyro_dim, T)

        # Concatenate features
        x = torch.cat([acc_feat, gyro_feat], dim=1)  # (B, embed_dim, T)

        # Rearrange for transformer
        x = rearrange(x, 'b c t -> b t c')  # (B, T, embed_dim)

        # Add positional encoding (if enabled)
        if self.use_pos_encoding and self.pos_encoding is not None:
            if T <= self.imu_frames:
                x = x + self.pos_encoding[:, :T, :]
            else:
                pos = torch.nn.functional.interpolate(
                    self.pos_encoding.permute(0, 2, 1),
                    size=T,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
                x = x + pos

        # Transformer expects (T, B, C)
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')  # Back to (B, T, C)

        # Global Average Pooling (NO temporal attention for base model)
        features = x.mean(dim=1)  # (B, embed_dim)

        # Classification
        features = self.dropout(features)
        logits = self.output(features)

        return logits, features

    def get_stream_features(self,
                           acc_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get separate stream features for analysis.

        Args:
            acc_data: (B, T, 6) IMU data

        Returns:
            acc_feat: (B, acc_dim, T) accelerometer features
            gyro_feat: (B, gyro_dim, T) gyroscope features
        """
        acc = acc_data[:, :, :3]
        gyro = acc_data[:, :, 3:]

        acc = rearrange(acc, 'b t c -> b c t')
        gyro = rearrange(gyro, 'b t c -> b c t')

        return self.acc_proj(acc), self.gyro_proj(gyro)


class DualStreamBaseSE(DualStreamBase):
    """
    Dual-stream with SE attention (but NO temporal attention pooling).

    This variant adds Squeeze-Excitation blocks after each stream projection
    but still uses Global Average Pooling for temporal aggregation.

    This allows studying the effect of SE separately from temporal attention.
    """

    def __init__(self, se_reduction: int = 4, **kwargs):
        super().__init__(**kwargs)

        # Add SE blocks for each stream
        self.acc_se = self._make_se_block(self.acc_dim, se_reduction)
        self.gyro_se = self._make_se_block(self.gyro_dim, se_reduction)

    def _make_se_block(self, channels: int, reduction: int) -> nn.Module:
        """Create SE block."""
        reduced = max(channels // reduction, 4)
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.SiLU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self,
                acc_data: torch.Tensor,
                skl_data: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with SE attention."""
        B, T, C = acc_data.shape

        acc = acc_data[:, :, :3]
        gyro = acc_data[:, :, 3:]

        acc = rearrange(acc, 'b t c -> b c t')
        gyro = rearrange(gyro, 'b t c -> b c t')

        # Project each stream
        acc_feat = self.acc_proj(acc)    # (B, acc_dim, T)
        gyro_feat = self.gyro_proj(gyro)  # (B, gyro_dim, T)

        # Apply SE attention
        acc_scale = self.acc_se(acc_feat).unsqueeze(2)   # (B, acc_dim, 1)
        gyro_scale = self.gyro_se(gyro_feat).unsqueeze(2)  # (B, gyro_dim, 1)

        acc_feat = acc_feat * acc_scale
        gyro_feat = gyro_feat * gyro_scale

        # Concatenate and process
        x = torch.cat([acc_feat, gyro_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')

        # Add positional encoding (if enabled)
        if self.use_pos_encoding and self.pos_encoding is not None:
            if T <= self.imu_frames:
                x = x + self.pos_encoding[:, :T, :]
            else:
                pos = torch.nn.functional.interpolate(
                    self.pos_encoding.permute(0, 2, 1),
                    size=T,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
                x = x + pos

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        # Global Average Pooling
        features = x.mean(dim=1)

        features = self.dropout(features)
        logits = self.output(features)

        return logits, features


# Aliases for easy import
DualBase = DualStreamBase
DualBaseSE = DualStreamBaseSE


if __name__ == '__main__':
    """Test the models."""
    print("=" * 60)
    print("DUAL STREAM BASE TRANSFORMER TESTS")
    print("=" * 60)

    # Test parameters
    batch_size = 4
    seq_len = 128
    channels = 6

    # Create dummy input
    x = torch.randn(batch_size, seq_len, channels)

    # Test 1: DualStreamBase (Model B/D)
    print("\nTest 1: DualStreamBase (no SE, no TAP)")
    model = DualStreamBase(
        imu_frames=seq_len,
        imu_channels=channels,
        acc_dim=48,
        gyro_dim=16
    )
    logits, features = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Embed dim: {model.embed_dim} (acc={model.acc_dim}, gyro={model.gyro_dim})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert logits.shape == (batch_size, 1)
    assert features.shape == (batch_size, 64)
    print("  PASSED")

    # Test 2: Stream feature extraction
    print("\nTest 2: Stream feature extraction")
    acc_feat, gyro_feat = model.get_stream_features(x)
    print(f"  Acc features shape: {acc_feat.shape}")
    print(f"  Gyro features shape: {gyro_feat.shape}")
    assert acc_feat.shape == (batch_size, 48, seq_len)
    assert gyro_feat.shape == (batch_size, 16, seq_len)
    print("  PASSED")

    # Test 3: DualStreamBaseSE
    print("\nTest 3: DualStreamBaseSE (with SE, no TAP)")
    model_se = DualStreamBaseSE(
        imu_frames=seq_len,
        imu_channels=channels,
        acc_dim=48,
        gyro_dim=16,
        se_reduction=4
    )
    logits, features = model_se(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_se.parameters()):,}")
    assert logits.shape == (batch_size, 1)
    assert features.shape == (batch_size, 64)
    print("  PASSED")

    # Test 4: Gradient flow
    print("\nTest 4: Gradient flow")
    model.zero_grad()
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  All parameters have gradients: {has_grad}")
    assert has_grad
    print("  PASSED")

    # Test 5: Asymmetric dropout verification
    print("\nTest 5: Dropout rates")
    model_test = DualStreamBase(dropout=0.5, acc_dropout_mult=0.2, gyro_dropout_mult=0.4)
    # Check that dropout layers exist with different rates
    acc_dropout = model_test.acc_proj[-1]
    gyro_dropout = model_test.gyro_proj[-1]
    print(f"  Acc dropout p={acc_dropout.p} (expected 0.1)")
    print(f"  Gyro dropout p={gyro_dropout.p} (expected 0.2)")
    assert abs(acc_dropout.p - 0.1) < 1e-6
    assert abs(gyro_dropout.p - 0.2) < 1e-6
    print("  PASSED")

    # Test 6: Variable sequence length
    print("\nTest 6: Variable sequence length")
    for seq in [64, 128, 256]:
        x_var = torch.randn(2, seq, 6)
        logits, _ = model(x_var)
        print(f"  Seq length {seq}: output shape {logits.shape}")
        assert logits.shape == (2, 1)
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
