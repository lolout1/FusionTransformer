"""
Encoder Ablation Study: Conv1D vs Linear Input Encoders.

This module provides a configurable KalmanTransformer variant for ablation studies
comparing different encoder architectures for accelerometer and orientation streams.

Experiment Design:
    1. conv1d_conv1d:  Conv1D(acc) + Conv1D(ori)  - Current baseline
    2. conv1d_linear:  Conv1D(acc) + Linear(ori)  - Hypothesis: Linear sufficient for smooth ori
    3. linear_conv1d:  Linear(acc) + Conv1D(ori)  - Control: verify acc needs Conv1D
    4. linear_linear:  Linear(acc) + Linear(ori)  - Ablation: both linear

Scientific Rationale:
    - Accelerometer: High-frequency transients, sharp impacts during falls
      → Conv1D's local temporal kernel captures these patterns effectively
    - Orientation (Kalman-filtered): Low-frequency, smooth signals
      → Linear projection may be sufficient; Conv1D might be overkill

References:
    - Zeng et al. (2022) "Are Transformers Effective for Time Series Forecasting?" AAAI
      - Simple linear models match complex architectures for smooth signals
    - Madgwick (2010) "An efficient orientation filter..."
      - Kalman-filtered orientation has different spectral properties than raw sensors

Usage:
    model = KalmanEncoderAblation(
        acc_encoder='conv1d',  # 'conv1d' or 'linear'
        ori_encoder='linear',  # 'conv1d' or 'linear'
        ...
    )
"""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Literal, Optional


# =============================================================================
# Shared Components (imported patterns from existing codebase)
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


# =============================================================================
# Modular Encoder Components
# =============================================================================

class Conv1DEncoder(nn.Module):
    """
    Conv1D-based temporal encoder.

    Captures local temporal patterns via sliding kernel.
    Best for high-frequency, transient signals (accelerometer).

    Architecture:
        Conv1d(in_ch, out_ch, kernel=8) -> BatchNorm -> SiLU -> Dropout
    """

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
        """
        Args:
            x: (B, T, C_in) input tensor
        Returns:
            (B, T, C_out) encoded tensor
        """
        # Conv1d expects (B, C, T)
        x = rearrange(x, 'b t c -> b c t')
        x = self.encoder(x)
        x = rearrange(x, 'b c t -> b t c')
        return x


class MultiKernelConv1DEncoder(nn.Module):
    """
    Multi-scale Conv1D encoder with parallel kernels.

    Captures multi-scale temporal patterns by running multiple kernel sizes
    in parallel and concatenating the outputs. Inspired by Inception modules.

    Architecture:
        Input -> [Conv1d(k=3), Conv1d(k=5), Conv1d(k=8), Conv1d(k=13)] -> Concat
              -> BatchNorm -> SiLU -> Dropout

    Benefits:
        - k=3: Sharp transients, sudden impacts
        - k=5: Short-duration patterns
        - k=8: Medium-duration patterns (default single-kernel)
        - k=13: Longer trends, anticipatory movements

    References:
        - Szegedy et al. (2015) "Going Deeper with Convolutions" (Inception)
        - Ismail Fawaz et al. (2019) "InceptionTime" for time series
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (3, 5, 8, 13),
        dropout: float = 0.1
    ):
        super().__init__()
        n_kernels = len(kernel_sizes)
        # Divide output channels among kernels (must be divisible)
        ch_per_kernel = out_channels // n_kernels
        remainder = out_channels % n_kernels

        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # Give extra channels to first kernels if not divisible
            ch = ch_per_kernel + (1 if i < remainder else 0)
            self.convs.append(
                nn.Conv1d(in_channels, ch, kernel_size=k, padding='same')
            )

        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C_in) input tensor
        Returns:
            (B, T, C_out) encoded tensor with multi-scale features
        """
        # Conv1d expects (B, C, T)
        x = rearrange(x, 'b t c -> b c t')

        # Apply each kernel and concatenate
        features = [conv(x) for conv in self.convs]
        x = torch.cat(features, dim=1)  # (B, C_out, T)

        # Norm, activation, dropout
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = rearrange(x, 'b c t -> b t c')
        return x


class LinearEncoder(nn.Module):
    """
    Linear (per-timestep) encoder.

    Projects each timestep independently without temporal convolution.
    Suitable for smooth, low-frequency signals (Kalman-filtered orientation).

    Architecture:
        Linear(in_ch, out_ch) -> LayerNorm -> SiLU -> Dropout

    Note: Uses LayerNorm (not BatchNorm) since operation is per-timestep.
    """

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
        """
        Args:
            x: (B, T, C_in) input tensor
        Returns:
            (B, T, C_out) encoded tensor (linear projection per timestep)
        """
        # Linear operates on last dimension, works directly on (B, T, C)
        return self.encoder(x)


# =============================================================================
# Factory Function
# =============================================================================

def create_encoder(
    encoder_type: Literal['conv1d', 'linear', 'multikernel'],
    in_channels: int,
    out_channels: int,
    kernel_size: int = 8,
    dropout: float = 0.1
) -> nn.Module:
    """
    Factory function to create encoder based on type.

    Args:
        encoder_type: 'conv1d', 'linear', or 'multikernel'
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for Conv1D (ignored for Linear/MultiKernel)
        dropout: Dropout rate

    Returns:
        Encoder module
    """
    if encoder_type == 'conv1d':
        return Conv1DEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
    elif encoder_type == 'linear':
        return LinearEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout
        )
    elif encoder_type == 'multikernel':
        return MultiKernelConv1DEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=(3, 5, 8, 13),
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'conv1d', 'linear', or 'multikernel'.")


# =============================================================================
# Main Ablation Model
# =============================================================================

class KalmanEncoderAblation(nn.Module):
    """
    Configurable KalmanTransformer for encoder ablation studies.

    Allows independent selection of encoder type for accelerometer and
    orientation streams, enabling controlled experiments.

    Configurations:
        acc_encoder='conv1d', ori_encoder='conv1d'   -> Baseline (current best)
        acc_encoder='conv1d', ori_encoder='linear'   -> Hybrid (hypothesis)
        acc_encoder='linear', ori_encoder='conv1d'   -> Control
        acc_encoder='linear', ori_encoder='linear'   -> Full ablation

    Input: 7ch [smv, ax, ay, az, roll, pitch, yaw]

    Architecture:
        Accelerometer (4ch) -> [acc_encoder] -> acc_dim
        Orientation (3ch)   -> [ori_encoder] -> ori_dim
        Concatenate -> LayerNorm -> Transformer -> SE -> TAP -> Classifier
    """

    def __init__(
        self,
        # Encoder configuration (NEW parameters for ablation)
        acc_encoder: Literal['conv1d', 'linear'] = 'conv1d',
        ori_encoder: Literal['conv1d', 'linear'] = 'conv1d',
        acc_kernel_size: int = 8,
        ori_kernel_size: int = 8,
        # Standard KalmanTransformer parameters
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
        **kwargs
    ):
        super().__init__()

        # Store configuration for logging/reproducibility
        self.acc_encoder_type = acc_encoder
        self.ori_encoder_type = ori_encoder
        self.config_name = f"{acc_encoder}_{ori_encoder}"

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Channel split for Kalman features
        # 7ch: [smv, ax, ay, az] (4) + [roll, pitch, yaw] (3)
        self.acc_channels = 4  # smv, ax, ay, az
        self.ori_channels = self.imu_channels - 4  # orientation channels

        # Asymmetric embedding allocation
        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim
        self.acc_dim = acc_dim
        self.ori_dim = ori_dim

        # Create encoders using factory (MODULAR)
        self.acc_proj = create_encoder(
            encoder_type=acc_encoder,
            in_channels=self.acc_channels,
            out_channels=acc_dim,
            kernel_size=acc_kernel_size,
            dropout=dropout * 0.2  # Lower dropout for acc (cleaner signal)
        )

        self.ori_proj = create_encoder(
            encoder_type=ori_encoder,
            in_channels=self.ori_channels,
            out_channels=ori_dim,
            kernel_size=ori_kernel_size,
            dropout=dropout * 0.3  # Slightly higher for ori
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

        # Project each stream (encoder handles rearrangement internally)
        acc_feat = self.acc_proj(acc)  # (B, T, acc_dim)
        ori_feat = self.ori_proj(ori)  # (B, T, ori_dim)

        # Concatenate and normalize
        x = torch.cat([acc_feat, ori_feat], dim=-1)  # (B, T, embed_dim)
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

    def get_encoder_info(self) -> dict:
        """Return encoder configuration for logging."""
        acc_params = sum(p.numel() for p in self.acc_proj.parameters())
        ori_params = sum(p.numel() for p in self.ori_proj.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'config_name': self.config_name,
            'acc_encoder': self.acc_encoder_type,
            'ori_encoder': self.ori_encoder_type,
            'acc_channels': self.acc_channels,
            'ori_channels': self.ori_channels,
            'acc_dim': self.acc_dim,
            'ori_dim': self.ori_dim,
            'acc_encoder_params': acc_params,
            'ori_encoder_params': ori_params,
            'total_params': total_params,
        }


# =============================================================================
# Convenience Classes for Direct Config Reference
# =============================================================================

class KalmanConv1dConv1d(KalmanEncoderAblation):
    """Baseline: Conv1D for both streams (current best)."""
    def __init__(self, **kwargs):
        # Remove encoder overrides - this class defines fixed encoders
        kwargs.pop('acc_encoder', None)
        kwargs.pop('ori_encoder', None)
        super().__init__(acc_encoder='conv1d', ori_encoder='conv1d', **kwargs)


class KalmanConv1dLinear(KalmanEncoderAblation):
    """Hybrid: Conv1D for acc, Linear for ori (hypothesis under test)."""
    def __init__(self, **kwargs):
        # Remove encoder overrides - this class defines fixed encoders
        kwargs.pop('acc_encoder', None)
        kwargs.pop('ori_encoder', None)
        super().__init__(acc_encoder='conv1d', ori_encoder='linear', **kwargs)


class KalmanLinearConv1d(KalmanEncoderAblation):
    """Control: Linear for acc, Conv1D for ori."""
    def __init__(self, **kwargs):
        # Remove encoder overrides - this class defines fixed encoders
        kwargs.pop('acc_encoder', None)
        kwargs.pop('ori_encoder', None)
        super().__init__(acc_encoder='linear', ori_encoder='conv1d', **kwargs)


class KalmanLinearLinear(KalmanEncoderAblation):
    """Full ablation: Linear for both streams."""
    def __init__(self, **kwargs):
        # Remove encoder overrides - this class defines fixed encoders
        kwargs.pop('acc_encoder', None)
        kwargs.pop('ori_encoder', None)
        super().__init__(acc_encoder='linear', ori_encoder='linear', **kwargs)


class KalmanMultiKernelLinear(KalmanEncoderAblation):
    """
    Multi-kernel Conv1D for acc, Linear for ori.

    Uses parallel kernels (k=3,5,8,13) to capture multi-scale temporal
    patterns in accelerometer data. Linear encoder for orientation since
    Kalman-filtered Euler angles are already temporally smooth.

    Expected improvement: +0.3-0.8% F1 over single-kernel Conv1D.
    """
    def __init__(self, **kwargs):
        # Remove encoder overrides - this class defines fixed encoders
        kwargs.pop('acc_encoder', None)
        kwargs.pop('ori_encoder', None)
        super().__init__(acc_encoder='multikernel', ori_encoder='linear', **kwargs)


# =============================================================================
# Test Script
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Encoder Ablation Study - Architecture Test")
    print("=" * 70)

    configs = [
        ("conv1d_conv1d (Baseline)", 'conv1d', 'conv1d'),
        ("conv1d_linear (Hybrid)", 'conv1d', 'linear'),
        ("multikernel_linear (Multi-scale)", 'multikernel', 'linear'),
        ("linear_conv1d (Control)", 'linear', 'conv1d'),
        ("linear_linear (Full Ablation)", 'linear', 'linear'),
    ]

    results = []

    for name, acc_enc, ori_enc in configs:
        print(f"\n{name}")
        print("-" * 50)

        model = KalmanEncoderAblation(
            acc_encoder=acc_enc,
            ori_encoder=ori_enc,
            imu_frames=128,
            imu_channels=7,
            embed_dim=64,
            num_classes=1,
            num_heads=4,
            num_layers=2,
            dropout=0.5,
            acc_ratio=0.65
        )

        x = torch.randn(8, 128, 7)
        logits, features = model(x)

        info = model.get_encoder_info()
        results.append(info)

        print(f"  Acc encoder:  {info['acc_encoder']} ({info['acc_encoder_params']:,} params)")
        print(f"  Ori encoder:  {info['ori_encoder']} ({info['ori_encoder_params']:,} params)")
        print(f"  Total params: {info['total_params']:,}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Features:     {features.shape}")

    # Gradient check
    print("\n" + "=" * 70)
    print("Gradient Flow Check")
    print("=" * 70)

    for name, acc_enc, ori_enc in configs:
        model = KalmanEncoderAblation(acc_encoder=acc_enc, ori_encoder=ori_enc)
        model.train()
        x = torch.randn(4, 128, 7, requires_grad=True)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        grad_norm = x.grad.norm().item()
        status = "OK" if grad_norm > 0 else "FAIL"
        print(f"  {name}: grad_norm={grad_norm:.4f} [{status}]")

    # Parameter comparison
    print("\n" + "=" * 70)
    print("Parameter Comparison")
    print("=" * 70)
    print(f"{'Configuration':<30} {'Acc Params':>12} {'Ori Params':>12} {'Total':>12}")
    print("-" * 70)
    for info in results:
        print(f"{info['config_name']:<30} {info['acc_encoder_params']:>12,} {info['ori_encoder_params']:>12,} {info['total_params']:>12,}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
