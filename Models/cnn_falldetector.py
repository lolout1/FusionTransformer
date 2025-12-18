"""
CNN-based Fall Detection Models for Ablation Study.

This module provides CNN architectures to compare against Transformer-based models.
All models follow the same interface: forward(acc_data, skl_data=None, **kwargs) -> (logits, features)

CNN Variants:
    1. CNNFallDetector - Single-stream CNN for raw 6ch input
    2. CNNKalmanFallDetector - Single-stream CNN for 7ch Kalman input
    3. CNNDualStream - Dual-stream CNN for raw 6ch input
    4. CNNKalmanDualStream - Dual-stream CNN for 7ch Kalman input

Design Rationale:
    - Match ~35K parameters for fair comparison with transformers
    - Use similar components (SE, TAP) for controlled ablation
    - Temporal convolutions capture local patterns vs global attention
"""

import torch
from torch import nn
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


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with optional downsampling.

    Uses pre-activation design (BN -> Act -> Conv) for stable training.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5,
                 stride: int = 1, dropout: float = 0.3):
        super().__init__()

        padding = kernel_size // 2

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        identity = self.shortcut(x)

        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.dropout(out)

        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.dropout(out)

        return out + identity


class CNNFallDetector(nn.Module):
    """
    Single-stream CNN for IMU input (dynamic channel count).

    Architecture:
        - Initial projection conv
        - 3 residual blocks with progressive channel expansion
        - SE attention for channel recalibration
        - Temporal attention pooling

    Input: Variable channels (typically 8ch raw or 7ch Kalman)
           8ch raw: [smv, ax, ay, az, gyro_mag, gx, gy, gz]
           7ch Kalman: [smv, ax, ay, az, roll, pitch, yaw]
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 8,
                 acc_frames: int = 128,
                 acc_coords: int = 8,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 num_classes: int = 1,
                 embed_dim: int = 64,
                 dropout: float = 0.5,
                 se_reduction: int = 4,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        # Use imu_channels if provided, else acc_coords, else default to 8
        self.imu_channels = imu_channels if imu_channels else (acc_coords if acc_coords else 8)
        self.embed_dim = embed_dim

        # Initial projection - dynamically sized to input channels
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.imu_channels, embed_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        # Residual blocks
        self.res1 = ResidualBlock(embed_dim // 2, embed_dim // 2, kernel_size=5, dropout=dropout * 0.3)
        self.res2 = ResidualBlock(embed_dim // 2, embed_dim, kernel_size=5, dropout=dropout * 0.4)
        self.res3 = ResidualBlock(embed_dim, embed_dim, kernel_size=5, dropout=dropout * 0.4)

        # Output layers
        self.final_norm = nn.LayerNorm(embed_dim)
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        self.temporal_pool = TemporalAttentionPooling(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        # Input: (B, T, C) -> (B, C, T) for Conv1d
        x = rearrange(acc_data, 'b t c -> b c t')

        # CNN backbone
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        # Back to (B, T, C) for pooling
        x = rearrange(x, 'b c t -> b t c')
        x = self.final_norm(x)
        x = self.se(x)
        features = x

        # Temporal pooling and classification
        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


class CNNKalmanFallDetector(nn.Module):
    """
    Single-stream CNN for 7-channel Kalman-fused input.

    Input: 7ch [smv, ax, ay, az, roll, pitch, yaw]

    Same architecture as CNNFallDetector but for Kalman preprocessing.
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 7,
                 acc_frames: int = 128,
                 acc_coords: int = 7,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 num_classes: int = 1,
                 embed_dim: int = 64,
                 dropout: float = 0.5,
                 se_reduction: int = 4,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = 7  # Fixed for Kalman: smv + acc + euler
        self.embed_dim = embed_dim

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(7, embed_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(embed_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        # Residual blocks
        self.res1 = ResidualBlock(embed_dim // 2, embed_dim // 2, kernel_size=5, dropout=dropout * 0.3)
        self.res2 = ResidualBlock(embed_dim // 2, embed_dim, kernel_size=5, dropout=dropout * 0.4)
        self.res3 = ResidualBlock(embed_dim, embed_dim, kernel_size=5, dropout=dropout * 0.4)

        # Output layers
        self.final_norm = nn.LayerNorm(embed_dim)
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        self.temporal_pool = TemporalAttentionPooling(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        # Use first 7 channels for Kalman input
        x = acc_data[:, :, :7]
        x = rearrange(x, 'b t c -> b c t')

        # CNN backbone
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        # Back to (B, T, C) for pooling
        x = rearrange(x, 'b c t -> b t c')
        x = self.final_norm(x)
        x = self.se(x)
        features = x

        # Temporal pooling and classification
        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


class CNNDualStream(nn.Module):
    """
    Dual-stream CNN for raw 8-channel IMU input.

    Architecture:
        - Separate streams for accelerometer (4ch) and gyroscope (4ch)
        - Each stream has residual blocks
        - Late fusion via concatenation
        - SE attention and TAP pooling

    Input: 8ch [smv, ax, ay, az, gyro_mag, gx, gy, gz]
           Stream 1: acc 4ch [smv, ax, ay, az]
           Stream 2: gyro 4ch [gyro_mag, gx, gy, gz]
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 8,
                 acc_frames: int = 128,
                 acc_coords: int = 8,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 num_classes: int = 1,
                 embed_dim: int = 64,
                 dropout: float = 0.5,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.65,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else (acc_coords if acc_coords else 8)
        self.embed_dim = embed_dim

        # For 8ch raw: [smv, ax, ay, az, gyro_mag, gx, gy, gz]
        # acc_channels = 4 (first 4), gyro_channels = 4 (last 4)
        self.acc_channels = 4
        self.gyro_channels = self.imu_channels - 4

        # Split dimensions (65% acc, 35% gyro like transformer)
        acc_dim = int(embed_dim * acc_ratio)
        gyro_dim = embed_dim - acc_dim

        # Accelerometer stream (4ch -> acc_dim)
        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(acc_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )
        self.acc_res1 = ResidualBlock(acc_dim // 2, acc_dim, kernel_size=5, dropout=dropout * 0.3)
        self.acc_res2 = ResidualBlock(acc_dim, acc_dim, kernel_size=5, dropout=dropout * 0.3)

        # Gyroscope stream (4ch -> gyro_dim)
        self.gyro_proj = nn.Sequential(
            nn.Conv1d(self.gyro_channels, gyro_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(gyro_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3)
        )
        self.gyro_res1 = ResidualBlock(gyro_dim // 2, gyro_dim, kernel_size=5, dropout=dropout * 0.4)
        self.gyro_res2 = ResidualBlock(gyro_dim, gyro_dim, kernel_size=5, dropout=dropout * 0.4)

        # Fusion and output
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.fusion_res = ResidualBlock(embed_dim, embed_dim, kernel_size=5, dropout=dropout * 0.4)

        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        self.temporal_pool = TemporalAttentionPooling(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        # Split into acc and gyro streams
        # 8ch raw: [smv, ax, ay, az, gyro_mag, gx, gy, gz]
        acc = acc_data[:, :, :self.acc_channels]  # First 4 channels
        gyro = acc_data[:, :, self.acc_channels:self.acc_channels + self.gyro_channels]  # Remaining channels

        # Process streams
        acc = rearrange(acc, 'b t c -> b c t')
        gyro = rearrange(gyro, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        acc_feat = self.acc_res1(acc_feat)
        acc_feat = self.acc_res2(acc_feat)

        gyro_feat = self.gyro_proj(gyro)
        gyro_feat = self.gyro_res1(gyro_feat)
        gyro_feat = self.gyro_res2(gyro_feat)

        # Concatenate streams
        x = torch.cat([acc_feat, gyro_feat], dim=1)

        # Fusion block
        x = self.fusion_res(x)

        # Back to (B, T, C)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)
        x = self.se(x)
        features = x

        # Temporal pooling and classification
        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


class CNNKalmanDualStream(nn.Module):
    """
    Dual-stream CNN for 7-channel Kalman-fused input.

    Architecture:
        - Accelerometer stream: 4ch [smv, ax, ay, az] -> 65% capacity
        - Orientation stream: 3ch [roll, pitch, yaw] -> 35% capacity
        - Late fusion with residual block
        - SE attention and TAP pooling

    Input: 7ch [smv, ax, ay, az, roll, pitch, yaw]
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 7,
                 acc_frames: int = 128,
                 acc_coords: int = 7,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 num_classes: int = 1,
                 embed_dim: int = 64,
                 dropout: float = 0.5,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.65,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = 7  # Fixed for Kalman
        self.embed_dim = embed_dim

        self.acc_channels = 4  # smv, ax, ay, az
        self.ori_channels = 3  # roll, pitch, yaw

        # Split dimensions
        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim

        # Accelerometer stream (4ch -> acc_dim)
        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(acc_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )
        self.acc_res1 = ResidualBlock(acc_dim // 2, acc_dim, kernel_size=5, dropout=dropout * 0.3)
        self.acc_res2 = ResidualBlock(acc_dim, acc_dim, kernel_size=5, dropout=dropout * 0.3)

        # Orientation stream (3ch -> ori_dim)
        self.ori_proj = nn.Sequential(
            nn.Conv1d(self.ori_channels, ori_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(ori_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3)
        )
        self.ori_res1 = ResidualBlock(ori_dim // 2, ori_dim, kernel_size=5, dropout=dropout * 0.4)
        self.ori_res2 = ResidualBlock(ori_dim, ori_dim, kernel_size=5, dropout=dropout * 0.4)

        # Fusion and output
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.fusion_res = ResidualBlock(embed_dim, embed_dim, kernel_size=5, dropout=dropout * 0.4)

        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        self.temporal_pool = TemporalAttentionPooling(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        # Split into acc and orientation
        acc = acc_data[:, :, :self.acc_channels]  # smv, ax, ay, az
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]  # roll, pitch, yaw

        # Process streams
        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        acc_feat = self.acc_res1(acc_feat)
        acc_feat = self.acc_res2(acc_feat)

        ori_feat = self.ori_proj(ori)
        ori_feat = self.ori_res1(ori_feat)
        ori_feat = self.ori_res2(ori_feat)

        # Concatenate streams
        x = torch.cat([acc_feat, ori_feat], dim=1)

        # Fusion block
        x = self.fusion_res(x)

        # Back to (B, T, C)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)
        x = self.se(x)
        features = x

        # Temporal pooling and classification
        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Test Script
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CNN Fall Detector - Architecture Test")
    print("=" * 70)

    variants = [
        ("CNNFallDetector (8ch raw)", CNNFallDetector, 8),
        ("CNNKalmanFallDetector (7ch Kalman)", CNNKalmanFallDetector, 7),
        ("CNNDualStream (8ch raw)", CNNDualStream, 8),
        ("CNNKalmanDualStream (7ch Kalman)", CNNKalmanDualStream, 7),
    ]

    for name, model_cls, channels in variants:
        print(f"\n{name}")
        print("-" * 50)

        model = model_cls(imu_frames=128, imu_channels=channels)
        x = torch.randn(8, 128, channels)
        logits, features = model(x)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Input:      ({8}, {128}, {channels})")
        print(f"  Output:     {logits.shape}")
        print(f"  Features:   {features.shape}")
        print(f"  Parameters: {total_params:,} ({trainable_params:,} trainable)")

    # Gradient check
    print("\n" + "=" * 70)
    print("Gradient Flow Check")
    print("=" * 70)

    for name, model_cls, channels in variants:
        model = model_cls(imu_frames=128, imu_channels=channels)
        model.train()
        x = torch.randn(4, 128, channels, requires_grad=True)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        grad_norm = x.grad.norm().item()
        status = "OK" if grad_norm > 0 else "FAIL"
        print(f"  {name}: grad_norm={grad_norm:.4f} [{status}]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
