"""
Dual-Stream Kalman Transformer for Wearable Fall Detection.

This module implements a dual-stream transformer architecture that processes
Kalman-filtered IMU signals for binary fall detection. The model achieves
91.10% Test F1 on the SmartFallMM dataset using Leave-One-Subject-Out
cross-validation (19 folds).

Architecture Overview:
    Input: 7-channel Kalman-fused signal [SMV, ax, ay, az, roll, pitch, yaw]

    Stream 1 (Acceleration): [SMV, ax, ay, az] -> Conv1D -> BatchNorm -> SiLU
    Stream 2 (Orientation):  [roll, pitch, yaw] -> Conv1D -> BatchNorm -> SiLU

    Fusion: Concatenation + LayerNorm
    Encoder: 2-layer Transformer with pre-norm
    Attention: Squeeze-Excitation (channel) + Temporal Attention Pooling
    Output: Binary classification (fall/non-fall)

Key Design Choices:
    - Dual-stream separates acceleration (impact) from orientation (posture)
    - Linear Kalman filter fuses accelerometer + gyroscope -> euler angles
    - Channel-aware normalization: StandardScaler on acc only, ori kept in radians
    - Temporal Attention Pooling focuses on transient fall events
    - Focal loss handles class imbalance (falls are minority class)

Reference:
    SmartFallMM Dataset - Texas State University
    https://github.com/lolout1/FusionTransformer

Authors:
    Abheek Pradhan, Anne Ngu
    Texas State University
"""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


class SqueezeExcitation(nn.Module):
    """
    Squeeze-Excitation channel attention module.

    Learns channel-wise importance weights via global average pooling
    followed by a bottleneck MLP. Recalibrates feature maps by
    emphasizing informative channels and suppressing less useful ones.

    Reference:
        Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 4)
    """

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
        """
        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            Recalibrated tensor of shape (B, T, C)
        """
        scale = x.mean(dim=1)  # Global average pooling: (B, C)
        scale = self.fc(scale).unsqueeze(1)  # (B, 1, C)
        return x * scale


class TemporalAttentionPooling(nn.Module):
    """
    Learnable temporal attention pooling for sequence aggregation.

    Learns to weight timesteps based on their importance for classification.
    Critical for fall detection where the impact event is brief and localized
    within the sequence, while surrounding frames may be less informative.

    Unlike global average pooling, TAP can focus on the most discriminative
    temporal regions (e.g., the impact moment in a fall sequence).

    Args:
        embed_dim: Embedding dimension of input features
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            context: Aggregated representation (B, C)
            weights: Attention weights for visualization (B, T)
        """
        scores = self.attention(x).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1)  # (B, T)
        context = torch.einsum('bt,btc->bc', weights, x)  # (B, C)
        return context, weights


class TransformerEncoderWithNorm(nn.TransformerEncoder):
    """
    Transformer encoder with guaranteed final layer normalization.

    Extends PyTorch's TransformerEncoder to ensure the output is
    normalized regardless of the norm_first setting in encoder layers.
    """

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DualStreamKalmanTransformer(nn.Module):
    """
    Dual-Stream Transformer for Kalman-fused IMU Fall Detection.

    Processes 7-channel Kalman-filtered IMU data through parallel streams:
    - Acceleration stream (4ch): Captures impact dynamics [SMV, ax, ay, az]
    - Orientation stream (3ch): Captures body posture [roll, pitch, yaw]

    The streams are projected to a shared embedding space, concatenated,
    and processed by a transformer encoder with attention mechanisms.

    Architecture:
        Input (B, T, 7) -> Split -> Dual Conv1D Projections
            -> Concatenate -> LayerNorm -> Transformer Encoder
            -> Squeeze-Excitation -> Temporal Attention Pooling
            -> Dropout -> Linear -> Sigmoid

    Args:
        imu_frames: Sequence length (default: 128 frames = 4.27s at 30Hz)
        imu_channels: Total input channels (default: 7)
        num_classes: Output classes (default: 1 for binary)
        num_heads: Transformer attention heads (default: 4)
        num_layers: Transformer encoder layers (default: 2)
        embed_dim: Embedding dimension (default: 64)
        dropout: Dropout probability (default: 0.5)
        activation: Transformer activation (default: 'relu')
        norm_first: Pre-norm transformer (default: True)
        se_reduction: SE bottleneck reduction (default: 4)
        acc_ratio: Capacity ratio for acceleration stream (default: 0.5)
        use_se: Enable Squeeze-Excitation (default: True)
        use_tap: Enable Temporal Attention Pooling (default: True)
        use_pos_encoding: Enable positional encoding (default: False)
        acc_channels: Acceleration input channels (default: 4)
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
                 acc_ratio: float = 0.5,
                 use_se: bool = True,
                 use_tap: bool = True,
                 use_pos_encoding: bool = False,
                 acc_channels: int = 4,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_se = use_se
        self.use_tap = use_tap
        self.use_pos_encoding = use_pos_encoding

        # Channel allocation
        self.acc_channels = acc_channels  # 4: [SMV, ax, ay, az]
        self.ori_channels = self.imu_channels - acc_channels  # 3: [roll, pitch, yaw]

        # Embedding dimensions per stream
        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim

        # Acceleration stream projection
        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        # Orientation stream projection
        self.ori_proj = nn.Sequential(
            nn.Conv1d(self.ori_channels, ori_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(ori_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3)
        )

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Optional positional encoding
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, self.imu_frames, embed_dim) * 0.02
            )
        else:
            self.pos_encoding = None

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

        # Squeeze-Excitation attention
        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = None

        # Temporal pooling
        if use_tap:
            self.temporal_pool = TemporalAttentionPooling(embed_dim)
        else:
            self.temporal_pool = None

        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize output layer with scaled normal distribution."""
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        """
        Forward pass through dual-stream architecture.

        Args:
            acc_data: IMU tensor of shape (B, T, 7)
                      Channels: [SMV, ax, ay, az, roll, pitch, yaw]
            skl_data: Unused (for API compatibility with skeleton models)

        Returns:
            logits: Classification logits (B, 1)
            features: Intermediate features for analysis (B, T, embed_dim)
        """
        # Split into acceleration and orientation streams
        acc = acc_data[:, :, :self.acc_channels]  # (B, T, 4)
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]  # (B, T, 3)

        # Rearrange for Conv1d: (B, T, C) -> (B, C, T)
        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        # Stream projections
        acc_feat = self.acc_proj(acc)  # (B, acc_dim, T)
        ori_feat = self.ori_proj(ori)  # (B, ori_dim, T)

        # Concatenate streams and normalize
        x = torch.cat([acc_feat, ori_feat], dim=1)  # (B, embed_dim, T)
        x = rearrange(x, 'b c t -> b t c')  # (B, T, embed_dim)
        x = self.fusion_norm(x)

        # Optional positional encoding
        if self.use_pos_encoding and self.pos_encoding is not None:
            x = x + self.pos_encoding

        # Transformer encoder (expects T, B, C)
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        # Squeeze-Excitation channel attention
        if self.se is not None:
            x = self.se(x)

        features = x  # Save for visualization/analysis

        # Temporal pooling
        if self.temporal_pool is not None:
            x, attn_weights = self.temporal_pool(x)
        else:
            x = x.mean(dim=1)  # Global Average Pooling

        # Classification head
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# Alias for backward compatibility
KalmanBalancedFlexible = DualStreamKalmanTransformer


if __name__ == "__main__":
    # Quick test
    model = DualStreamKalmanTransformer()
    x = torch.randn(2, 128, 7)
    logits, features = model(x)
    print(f"Input: {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Features: {features.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
