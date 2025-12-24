"""
DualStreamCFEB: Dual-Stream Transformer with Convolutional Feature Extractor Block (CFEB).

CFEB adds a CNN layer before the Transformer to extract local temporal features,
inspired by SOTA 2025 time series classification approaches.

Architecture:
    ACC [4ch] --> CFEB (2-layer CNN with residual) --> acc_feat (48d) --+
                                                                        +--> Concat --> Transformer --> SE --> TAP --> Classifier
    ORI [3ch] --> CFEB (2-layer CNN with residual) --> ori_feat (16d) --+

Input: [smv, ax, ay, az, roll, pitch, yaw] - 7ch Kalman-fused
Channel split: ACC = 4ch [smv, ax, ay, az], ORI = 3ch [roll, pitch, yaw]

CFEB Design:
- Two Conv1d layers with BatchNorm and SiLU activation
- Residual connection from input (with 1x1 conv for channel matching)
- Regularization via Dropout between layers
"""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


class CFEB(nn.Module):
    """Convolutional Feature Extractor Block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
        )

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        residual = self.residual(x)
        out = self.conv_block(x)
        return self.activation(out + residual)


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
        scale = x.mean(dim=1)
        scale = self.fc(scale).unsqueeze(1)
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
        scores = self.attention(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.einsum('bt,btc->bc', weights, x)
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


class DualStreamCFEB(nn.Module):
    """Dual-stream transformer with CFEB, SE, and Temporal Attention Pooling."""

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
                 acc_ratio: float = 0.75,
                 cfeb_kernel_size: int = 3,
                 cfeb_dropout: float = 0.1,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # 7ch Kalman input: [smv, ax, ay, az, roll, pitch, yaw]
        # ACC = 4ch [smv, ax, ay, az], ORI = 3ch [roll, pitch, yaw]
        if self.imu_channels == 7:
            self.acc_in_channels = 4
            self.ori_in_channels = 3
        elif self.imu_channels == 6:
            self.acc_in_channels = 3
            self.ori_in_channels = 3
        else:
            self.acc_in_channels = self.imu_channels // 2
            self.ori_in_channels = self.imu_channels - self.acc_in_channels

        # Asymmetric embedding allocation
        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim

        # CFEB for ACC stream (larger, more features)
        self.cfeb_acc = CFEB(
            in_channels=self.acc_in_channels,
            out_channels=acc_dim,
            kernel_size=cfeb_kernel_size,
            dropout=cfeb_dropout
        )

        # CFEB for ORI stream (smaller, orientation is smoother)
        self.cfeb_ori = CFEB(
            in_channels=self.ori_in_channels,
            out_channels=ori_dim,
            kernel_size=cfeb_kernel_size,
            dropout=cfeb_dropout * 1.5  # Slightly higher dropout for orientation
        )

        # Additional dropout after CFEB
        self.acc_dropout = nn.Dropout(dropout * 0.2)
        self.ori_dropout = nn.Dropout(dropout * 0.4)

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
            acc_data: (B, T, 7) IMU data [smv, ax, ay, az, roll, pitch, yaw]
            skl_data: Unused, for API compatibility

        Returns:
            logits: (B, num_classes) classification logits
            features: (B, T, embed_dim) encoder features before pooling
        """
        # Split modalities
        acc = acc_data[:, :, :self.acc_in_channels]
        ori = acc_data[:, :, self.acc_in_channels:self.acc_in_channels + self.ori_in_channels]

        # Rearrange for Conv1d: (B, T, C) -> (B, C, T)
        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        # CFEB feature extraction
        acc_feat = self.cfeb_acc(acc)
        acc_feat = self.acc_dropout(acc_feat)

        ori_feat = self.cfeb_ori(ori)
        ori_feat = self.ori_dropout(ori_feat)

        # Concatenate and normalize: (B, embed_dim, T) -> (B, T, embed_dim)
        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        # Transformer: (B, T, C) -> (T, B, C)
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)

        # SE module: (T, B, C) -> (B, T, C)
        x = rearrange(x, 't b c -> b t c')
        x = self.se(x)
        features = x

        # Temporal attention pooling: (B, T, C) -> (B, C)
        x, attn_weights = self.temporal_pool(x)

        # Classification
        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


if __name__ == "__main__":
    print("=" * 60)
    print("DualStreamCFEB Model Architecture Test")
    print("=" * 60)

    # Test with 7 channels (Kalman)
    print("\n" + "=" * 50)
    print("Test: 7 channels (smv, ax, ay, az, roll, pitch, yaw)")
    print("=" * 50)
    model = DualStreamCFEB(imu_frames=128, imu_channels=7, embed_dim=64, num_classes=1)
    x = torch.randn(16, 128, 7)
    logits, features = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape: {x.shape}")
    print(f"ACC channels: {model.acc_in_channels}, ORI channels: {model.ori_in_channels}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Total parameters: {total_params:,}")

    # Test with 6 channels
    print("\n" + "=" * 50)
    print("Test: 6 channels (ax, ay, az, roll, pitch, yaw)")
    print("=" * 50)
    model_6ch = DualStreamCFEB(imu_frames=128, imu_channels=6, embed_dim=64, num_classes=1)
    x_6ch = torch.randn(16, 128, 6)
    logits, features = model_6ch(x_6ch)

    total_params = sum(p.numel() for p in model_6ch.parameters())
    print(f"Input shape: {x_6ch.shape}")
    print(f"ACC channels: {model_6ch.acc_in_channels}, ORI channels: {model_6ch.ori_in_channels}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
