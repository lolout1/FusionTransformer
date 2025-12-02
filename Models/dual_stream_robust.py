"""
Dual-Stream Fall Detection Architecture with Asymmetric Capacity.

Design rationale:
    - Dual-stream prevents noisy gyro (SNR<1 for 76% subjects) from corrupting acc features
    - Asymmetric capacity: acc gets 75% (reliable), gyro gets 25% (noisy)
    - No transformer for gyro - would overfit on noise patterns
    - SE modules learn to suppress noisy channels
    - Temporal attention focuses on fall impact phase (<500ms in 4s window)
    - CrossModalGate enables dynamic modality weighting

Architecture (DualStreamRobust ~18K params):
    - ACC stream (strong): DepthwiseSeparableConv1d(4->48) -> TransformerEncoderLayer(1 layer, 4 heads)
                          -> SqueezeExcitation -> TemporalAttentionPooling
    - GYRO stream (weak): DepthwiseSeparableConv1d(4->16) -> SqueezeExcitation -> simple MLP -> mean pooling
    - Fusion: CrossModalGate (learns dynamic acc/gyro weights) -> concat -> MLP(64->48) -> Linear(48->1)

Architecture (DualStreamSimple ~8K params, ablation baseline):
    - Separate Conv1d projections for acc and gyro
    - No SE, no attention, no gating
    - Global average pooling -> concat -> classifier

Channel configurations supported:
    - 6 channels (no SMV): [ax, ay, az, gx, gy, gz] -> acc=3ch, gyro=3ch
    - 8 channels (with SMV): [smv, ax, ay, az, gyro_mag, gx, gy, gz] -> acc=4ch, gyro=4ch

Expected results:
    - DualStreamSimple: ~86-87% F1 (separation helps)
    - DualStreamRobust: ~88-90% F1 (all components working)
    - vs current single-stream acc+gyro: 85.3% F1
"""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Tuple, Optional


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution for parameter efficiency.

    Reduces parameters by ~8x compared to standard Conv1d:
    - Depthwise: groups=in_channels (spatial filtering per channel)
    - Pointwise: 1x1 conv (channel mixing)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dropout: Dropout rate after activation
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 8,
                 dropout: float = 0.3):
        super().__init__()

        # Depthwise convolution: spatial filtering per channel
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding='same',
            groups=in_channels,
            bias=False
        )

        # Pointwise convolution: channel mixing (1x1 conv)
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )

        # Normalization and activation
        self.norm = nn.BatchNorm1d(out_channels, momentum=0.1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, T) input tensor

        Returns:
            (B, out_channels, T) output tensor
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation channel attention module.

    Learns to recalibrate channel-wise feature responses by:
    1. Squeeze: Global average pooling to capture channel-wise statistics
    2. Excitation: Two FC layers with bottleneck to model channel interdependencies
    3. Scale: Multiply input by learned channel weights

    Args:
        channels: Number of input/output channels
        reduction: Reduction ratio for bottleneck (default: 4)
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 4)  # Minimum 4 for small channel counts

        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.

        Args:
            x: (B, T, C) input tensor

        Returns:
            (B, T, C) channel-recalibrated tensor
        """
        # Global average pooling: (B, T, C) -> (B, C)
        scale = x.mean(dim=1)
        # Excitation: (B, C) -> (B, C)
        scale = self.fc(scale)
        # Scale: (B, T, C) * (B, 1, C) -> (B, T, C)
        return x * scale.unsqueeze(1)


class TemporalAttentionPooling(nn.Module):
    """
    Learnable temporal pooling with attention mechanism.

    Learns which time steps are most important for classification
    (e.g., fall impact phase vs. quiet periods).

    Architecture:
        Linear -> Tanh -> Linear -> Softmax -> Weighted sum

    Args:
        embed_dim: Embedding dimension
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted temporal pooling.

        Args:
            x: (B, T, C) input tensor

        Returns:
            context: (B, C) pooled representation
            weights: (B, T) attention weights
        """
        # Compute attention scores: (B, T, C) -> (B, T, 1) -> (B, T)
        scores = self.attention(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)

        # Weighted sum: (B, T) @ (B, T, C) -> (B, C)
        context = torch.einsum('bt,btc->bc', weights, x)
        return context, weights


class CrossModalGate(nn.Module):
    """
    Cross-modal gating mechanism for dynamic modality weighting.

    Learns to dynamically weight accelerometer vs gyroscope contributions
    based on the quality/informativeness of each modality in a given sample.

    Architecture:
        Concat features -> MLP -> Softmax -> (acc_weight, gyro_weight)

    Args:
        acc_dim: Accelerometer feature dimension
        gyro_dim: Gyroscope feature dimension
        hidden_dim: Hidden layer dimension (default: 16)
    """
    def __init__(self, acc_dim: int, gyro_dim: int, hidden_dim: int = 16):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(acc_dim + gyro_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # 2 outputs: acc_weight, gyro_weight
        )

    def forward(self, acc_feat: torch.Tensor, gyro_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dynamic modality weights.

        Args:
            acc_feat: (B, acc_dim) accelerometer features
            gyro_feat: (B, gyro_dim) gyroscope features

        Returns:
            acc_weight: (B, 1) weight for accelerometer
            gyro_weight: (B, 1) weight for gyroscope
        """
        # Concatenate features
        combined = torch.cat([acc_feat, gyro_feat], dim=-1)

        # Compute gate values: (B, 2)
        gate_values = self.gate(combined)

        # Softmax to ensure weights sum to 1: (B, 2)
        weights = F.softmax(gate_values, dim=-1)

        # Split into separate weights: (B, 1), (B, 1)
        acc_weight = weights[:, 0:1]
        gyro_weight = weights[:, 1:2]

        return acc_weight, gyro_weight


class TransformerEncoderWithNorm(nn.TransformerEncoder):
    """Standard transformer encoder with final layer normalization."""

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DualStreamRobust(nn.Module):
    """
    Lightweight asymmetric dual-stream transformer for fall detection (~18K params).

    Architecture:
        ACC stream (strong - reliable modality):
            DepthwiseSeparableConv1d(acc_in->acc_dim) -> TransformerEncoderLayer
            -> SqueezeExcitation -> TemporalAttentionPooling

        GYRO stream (weak - noisy modality, no transformer):
            DepthwiseSeparableConv1d(gyro_in->gyro_dim) -> SqueezeExcitation
            -> simple MLP -> mean pooling

        Fusion:
            CrossModalGate -> concat -> MLP(64->48) -> Linear(48->num_classes)

    Args:
        imu_frames: Number of time frames (for compatibility)
        imu_channels: Total input channels (6 or 8)
        acc_frames: Alias for imu_frames
        acc_coords: Alias for imu_channels
        mocap_frames: For compatibility (unused)
        num_joints: For compatibility (unused)
        acc_dim: Accelerometer embedding dimension (default: 48)
        gyro_dim: Gyroscope embedding dimension (default: 16)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 1)
        dropout: Base dropout rate (default: 0.5)
        acc_dropout: Dropout for acc stream (default: 0.3)
        gyro_dropout: Dropout for gyro stream (default: 0.5)
        num_classes: Number of output classes (default: 1)
        use_se: Enable SE modules (default: True)
        use_temporal_attention: Enable temporal attention pooling (default: True)
        use_cross_modal_gate: Enable cross-modal gating (default: True)
        activation: Activation function (default: 'gelu')
        norm_first: Pre-norm in transformer (default: True)
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 8,
                 acc_frames: int = 128,
                 acc_coords: int = 8,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 acc_dim: int = 48,
                 gyro_dim: int = 16,
                 num_heads: int = 4,
                 num_layers: int = 1,
                 dropout: float = 0.5,
                 acc_dropout: float = 0.3,
                 gyro_dropout: float = 0.5,
                 num_classes: int = 1,
                 use_se: bool = True,
                 use_temporal_attention: bool = True,
                 use_cross_modal_gate: bool = True,
                 activation: str = 'gelu',
                 norm_first: bool = True,
                 acc_in_channels: int = None,
                 gyro_in_channels: int = None,
                 **kwargs):
        super().__init__()

        # Handle parameter aliasing for compatibility
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Determine input channel split
        # Priority: explicit params > infer from imu_channels
        if acc_in_channels is not None and gyro_in_channels is not None:
            # Explicit channel split (e.g., for gyro_magnitude_only: acc=3, gyro=1)
            self.acc_in_channels = acc_in_channels
            self.gyro_in_channels = gyro_in_channels
        elif self.imu_channels == 8:
            # 8 channels: [smv, ax, ay, az, gyro_mag, gx, gy, gz] -> 4 acc, 4 gyro
            self.acc_in_channels = 4
            self.gyro_in_channels = 4
        elif self.imu_channels == 6:
            # 6 channels: [ax, ay, az, gx, gy, gz] -> 3 acc, 3 gyro
            self.acc_in_channels = 3
            self.gyro_in_channels = 3
        elif self.imu_channels == 4:
            # 4 channels: [ax, ay, az, gyro_mag] -> 3 acc, 1 gyro (gyro_magnitude_only)
            self.acc_in_channels = 3
            self.gyro_in_channels = 1
        else:
            # Flexible: split evenly
            self.acc_in_channels = self.imu_channels // 2
            self.gyro_in_channels = self.imu_channels - self.acc_in_channels

        self.acc_dim = acc_dim
        self.gyro_dim = gyro_dim
        self.use_se = use_se
        self.use_temporal_attention = use_temporal_attention
        self.use_cross_modal_gate = use_cross_modal_gate

        # ================== ACC STREAM (Strong) ==================
        # Depthwise separable conv for parameter efficiency
        self.acc_conv = DepthwiseSeparableConv1d(
            self.acc_in_channels, acc_dim,
            kernel_size=8,
            dropout=acc_dropout
        )

        # Transformer encoder layer for temporal modeling
        encoder_layer = TransformerEncoderLayer(
            d_model=acc_dim,
            nhead=num_heads,
            dim_feedforward=acc_dim * 2,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )
        self.acc_transformer = TransformerEncoderWithNorm(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(acc_dim)
        )

        # SE module for channel attention
        if use_se:
            self.acc_se = SqueezeExcitation(acc_dim, reduction=4)

        # Temporal attention pooling
        if use_temporal_attention:
            self.acc_temporal_pool = TemporalAttentionPooling(acc_dim)

        # ================== GYRO STREAM (Weak) ==================
        # Simpler architecture - no transformer (would overfit on noise)
        self.gyro_conv = DepthwiseSeparableConv1d(
            self.gyro_in_channels, gyro_dim,
            kernel_size=8,
            dropout=gyro_dropout
        )

        # SE module for channel attention
        if use_se:
            self.gyro_se = SqueezeExcitation(gyro_dim, reduction=4)

        # Simple MLP for gyro processing (instead of transformer)
        self.gyro_mlp = nn.Sequential(
            nn.Linear(gyro_dim, gyro_dim),
            nn.GELU(),
            nn.Dropout(gyro_dropout)
        )

        # ================== FUSION ==================
        # Cross-modal gate for dynamic weighting
        if use_cross_modal_gate:
            self.cross_gate = CrossModalGate(acc_dim, gyro_dim, hidden_dim=16)

        # Fusion MLP
        fusion_dim = acc_dim + gyro_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 48),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Classifier
        self.output = nn.Linear(48, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize output layer with scaled normal distribution."""
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            acc_data: (B, T, C) IMU data where C is 6 or 8 channels
                     8ch: [smv, ax, ay, az, gyro_mag, gx, gy, gz]
                     6ch: [ax, ay, az, gx, gy, gz]
            skl_data: Unused (for API compatibility)

        Returns:
            logits: (B, num_classes) classification logits
            features: (B, embed_dim) fused features before classifier
        """
        B, T, C = acc_data.shape

        # Split modalities
        acc = acc_data[:, :, :self.acc_in_channels]  # (B, T, acc_in)
        gyro = acc_data[:, :, self.acc_in_channels:self.acc_in_channels + self.gyro_in_channels]  # (B, T, gyro_in)

        # ================== ACC STREAM ==================
        # Conv: (B, T, C) -> (B, C, T) -> (B, acc_dim, T)
        acc = rearrange(acc, 'b t c -> b c t')
        acc = self.acc_conv(acc)

        # Transformer: (B, acc_dim, T) -> (T, B, acc_dim) -> (T, B, acc_dim)
        acc = rearrange(acc, 'b c t -> t b c')
        acc = self.acc_transformer(acc)

        # Back to batch-first: (T, B, acc_dim) -> (B, T, acc_dim)
        acc = rearrange(acc, 't b c -> b t c')

        # SE attention
        if self.use_se:
            acc = self.acc_se(acc)

        # Temporal pooling
        if self.use_temporal_attention:
            acc_pooled, acc_attn = self.acc_temporal_pool(acc)  # (B, acc_dim)
        else:
            acc_pooled = acc.mean(dim=1)  # (B, acc_dim)

        # ================== GYRO STREAM ==================
        # Conv: (B, T, C) -> (B, C, T) -> (B, gyro_dim, T)
        gyro = rearrange(gyro, 'b t c -> b c t')
        gyro = self.gyro_conv(gyro)

        # Back to batch-first: (B, gyro_dim, T) -> (B, T, gyro_dim)
        gyro = rearrange(gyro, 'b c t -> b t c')

        # SE attention
        if self.use_se:
            gyro = self.gyro_se(gyro)

        # Simple MLP processing (no transformer for noisy gyro)
        gyro = self.gyro_mlp(gyro)

        # Mean pooling for gyro (simpler than attention)
        gyro_pooled = gyro.mean(dim=1)  # (B, gyro_dim)

        # ================== FUSION ==================
        # Cross-modal gating
        if self.use_cross_modal_gate:
            acc_weight, gyro_weight = self.cross_gate(acc_pooled, gyro_pooled)
            acc_weighted = acc_pooled * acc_weight
            gyro_weighted = gyro_pooled * gyro_weight
            fused = torch.cat([acc_weighted, gyro_weighted], dim=-1)
        else:
            fused = torch.cat([acc_pooled, gyro_pooled], dim=-1)

        # Fusion MLP
        features = self.fusion(fused)  # (B, 48)

        # Classification
        logits = self.output(features)  # (B, num_classes)

        return logits, features


class DualStreamSimple(nn.Module):
    """
    Simple dual-stream baseline for ablation (~8K params).

    Architecture:
        - Separate Conv1d projections for acc and gyro
        - No SE, no attention, no gating
        - Global average pooling -> concat -> classifier

    Args:
        imu_frames: Number of time frames (for compatibility)
        imu_channels: Total input channels (6 or 8)
        acc_frames: Alias for imu_frames
        acc_coords: Alias for imu_channels
        mocap_frames: For compatibility (unused)
        num_joints: For compatibility (unused)
        acc_dim: Accelerometer embedding dimension (default: 32)
        gyro_dim: Gyroscope embedding dimension (default: 16)
        dropout: Dropout rate (default: 0.5)
        num_classes: Number of output classes (default: 1)
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 8,
                 acc_frames: int = 128,
                 acc_coords: int = 8,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 acc_dim: int = 32,
                 gyro_dim: int = 16,
                 dropout: float = 0.5,
                 num_classes: int = 1,
                 **kwargs):
        super().__init__()

        # Handle parameter aliasing
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Determine input channel split
        if self.imu_channels == 8:
            self.acc_in_channels = 4
            self.gyro_in_channels = 4
        elif self.imu_channels == 6:
            self.acc_in_channels = 3
            self.gyro_in_channels = 3
        else:
            self.acc_in_channels = self.imu_channels // 2
            self.gyro_in_channels = self.imu_channels - self.acc_in_channels

        self.acc_dim = acc_dim
        self.gyro_dim = gyro_dim

        # Simple Conv1d projections (not depthwise separable)
        self.acc_conv = nn.Sequential(
            nn.Conv1d(self.acc_in_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        self.gyro_conv = nn.Sequential(
            nn.Conv1d(self.gyro_in_channels, gyro_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(gyro_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(acc_dim + gyro_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize output layer."""
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            acc_data: (B, T, C) IMU data where C is 6 or 8 channels
            skl_data: Unused (for API compatibility)

        Returns:
            logits: (B, num_classes) classification logits
            features: (B, embed_dim) concatenated features before classifier
        """
        B, T, C = acc_data.shape

        # Split modalities
        acc = acc_data[:, :, :self.acc_in_channels]
        gyro = acc_data[:, :, self.acc_in_channels:self.acc_in_channels + self.gyro_in_channels]

        # ACC stream: (B, T, C) -> (B, C, T) -> (B, acc_dim, T)
        acc = rearrange(acc, 'b t c -> b c t')
        acc = self.acc_conv(acc)

        # GYRO stream: (B, T, C) -> (B, C, T) -> (B, gyro_dim, T)
        gyro = rearrange(gyro, 'b t c -> b c t')
        gyro = self.gyro_conv(gyro)

        # Global average pooling
        acc_pooled = acc.mean(dim=-1)  # (B, acc_dim)
        gyro_pooled = gyro.mean(dim=-1)  # (B, gyro_dim)

        # Concatenate
        features = torch.cat([acc_pooled, gyro_pooled], dim=-1)  # (B, acc_dim + gyro_dim)

        # Classification
        features_dropped = self.dropout(features)
        logits = self.output(features_dropped)

        return logits, features


if __name__ == "__main__":
    print("=" * 60)
    print("Dual-Stream Model Architecture Test")
    print("=" * 60)

    batch_size = 16
    seq_len = 128

    # Test DualStreamRobust with 8 channels (with SMV)
    print("\n" + "=" * 50)
    print("Test 1: DualStreamRobust - 8 channels (with SMV)")
    print("=" * 50)

    model_8ch = DualStreamRobust(
        imu_frames=seq_len,
        imu_channels=8,
        acc_dim=48,
        gyro_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.5,
        acc_dropout=0.3,
        gyro_dropout=0.5,
        num_classes=1,
        use_se=True,
        use_temporal_attention=True,
        use_cross_modal_gate=True
    )

    imu_data_8ch = torch.randn(batch_size, seq_len, 8)
    logits, features = model_8ch(imu_data_8ch)

    total_params = sum(p.numel() for p in model_8ch.parameters())
    trainable_params = sum(p.numel() for p in model_8ch.parameters() if p.requires_grad)

    print(f"Input shape: {imu_data_8ch.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test DualStreamRobust with 6 channels (no SMV)
    print("\n" + "=" * 50)
    print("Test 2: DualStreamRobust - 6 channels (no SMV)")
    print("=" * 50)

    model_6ch = DualStreamRobust(
        imu_frames=seq_len,
        imu_channels=6,
        acc_dim=48,
        gyro_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.5,
        num_classes=1
    )

    imu_data_6ch = torch.randn(batch_size, seq_len, 6)
    logits, features = model_6ch(imu_data_6ch)

    total_params = sum(p.numel() for p in model_6ch.parameters())
    print(f"Input shape: {imu_data_6ch.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Total parameters: {total_params:,}")

    # Test DualStreamSimple with 8 channels
    print("\n" + "=" * 50)
    print("Test 3: DualStreamSimple - 8 channels (ablation baseline)")
    print("=" * 50)

    model_simple = DualStreamSimple(
        imu_frames=seq_len,
        imu_channels=8,
        acc_dim=32,
        gyro_dim=16,
        dropout=0.5,
        num_classes=1
    )

    logits, features = model_simple(imu_data_8ch)

    total_params = sum(p.numel() for p in model_simple.parameters())
    print(f"Input shape: {imu_data_8ch.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Total parameters: {total_params:,}")

    # Test DualStreamSimple with 6 channels
    print("\n" + "=" * 50)
    print("Test 4: DualStreamSimple - 6 channels (no SMV)")
    print("=" * 50)

    model_simple_6ch = DualStreamSimple(
        imu_frames=seq_len,
        imu_channels=6,
        acc_dim=32,
        gyro_dim=16,
        dropout=0.5,
        num_classes=1
    )

    logits, features = model_simple_6ch(imu_data_6ch)

    total_params = sum(p.numel() for p in model_simple_6ch.parameters())
    print(f"Input shape: {imu_data_6ch.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
