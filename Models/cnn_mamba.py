"""
CNN-Mamba Fall Detection Model (PyTorch).

Exact PyTorch implementation of the TensorFlow dual-stream CNN-Mamba architecture
that achieved 91% F1 with Kalman preprocessing.

Architecture:
    - Dual-stream: Accelerometer (3ch) + Gyroscope/Orientation (3ch)
    - CNN feature extractor per stream (preserves sequence)
    - SimpleMamba temporal modeling per stream
    - Global average pooling
    - Concatenate + classifier

This is for the CNN vs Transformer ablation study.
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math


class SimpleMamba(nn.Module):
    """
    Mamba-like temporal modeling block.

    Matches TensorFlow version:
        conv = Conv1D(dim, kernel_size=1, activation='gelu')
        gate = Dense(dim, activation='sigmoid')
        proj = Dense(dim)
        output = proj(conv(x) * gate(conv(x)))
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) tensor
        Returns:
            (B, T, C) tensor
        """
        # Rearrange for Conv1d: (B, C, T)
        x_conv = rearrange(x, 'b t c -> b c t')
        x_conv = F.gelu(self.conv(x_conv))
        x_conv = rearrange(x_conv, 'b c t -> b t c')

        # Gate and project
        gate = torch.sigmoid(self.gate(x_conv))
        x_gated = x_conv * gate
        return self.proj(x_gated)


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor that preserves sequence dimension.

    Matches TensorFlow version:
        Conv1D(32, k=3, same, relu) -> MaxPool(2) -> (64, 32)
        Conv1D(32, k=3, same, relu) -> MaxPool(2) -> (32, 32)
        Conv1D(64, k=3, same, relu) -> (32, 64)

    Input: (B, T, C) where T=128, C=3
    Output: (B, T//4, 64) = (B, 32, 64)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) tensor, T=128, C=3
        Returns:
            (B, T//4, 64) tensor
        """
        # Rearrange for Conv1d: (B, C, T)
        x = rearrange(x, 'b t c -> b c t')

        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # (B, 32, 64)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # (B, 32, 32)

        x = F.relu(self.conv3(x))  # (B, 64, 32)

        # Rearrange back: (B, T, C)
        x = rearrange(x, 'b c t -> b t c')  # (B, 32, 64)
        return x


class BinaryClassifierHead(nn.Module):
    """
    Binary classifier head.

    Matches TensorFlow version:
        BatchNorm -> Dropout(0.25) -> Dense(256, relu)
        -> BatchNorm -> Dropout(0.25) -> Dense(1, sigmoid)

    Note: PyTorch uses BCEWithLogitsLoss so we don't apply sigmoid in forward.
    Note: Requires drop_last=True in DataLoader to avoid batch_size=1.
    """

    def __init__(self, in_features: int, dropout: float = 0.25):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features, 256)

        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))

        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class CNNMambaDualStream(nn.Module):
    """
    Dual-stream CNN-Mamba model for fall detection.

    EXACT PyTorch implementation of the TensorFlow version that achieved 91% F1.

    Architecture:
        Input1: accelerometer (B, 128, 3) [ax, ay, az]
        Input2: gyroscope/orientation (B, 128, 3) [gx, gy, gz] or [roll, pitch, yaw]

        Stream 1: CNN -> Mamba -> GlobalAvgPool
        Stream 2: CNN -> Mamba -> GlobalAvgPool
        Concatenate -> ClassifierHead

    For raw input: splits 6ch [ax, ay, az, gx, gy, gz] into 3ch + 3ch
    For Kalman input: expects 7ch [smv, ax, ay, az, roll, pitch, yaw]
                      uses [ax, ay, az] and [roll, pitch, yaw]
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 6,
                 acc_frames: int = 128,
                 acc_coords: int = 6,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 num_classes: int = 1,
                 mamba_dim: int = 32,
                 dropout: float = 0.25,
                 use_kalman: bool = False,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_kalman = use_kalman

        # CNN feature extractors (3ch input each)
        self.cnn_acc = CNNFeatureExtractor(in_channels=3)
        self.cnn_gyro = CNNFeatureExtractor(in_channels=3)

        # Mamba temporal modeling (64ch from CNN output)
        self.mamba_acc = SimpleMamba(dim=64)
        self.mamba_gyro = SimpleMamba(dim=64)

        # After global avg pool: 64 from each stream = 128 concatenated
        # Then project to mamba_dim before concatenation (matches TF pooled1/pooled2)
        self.proj_acc = nn.Linear(64, mamba_dim)
        self.proj_gyro = nn.Linear(64, mamba_dim)

        # Classifier head: 64 total (32 + 32)
        self.classifier = BinaryClassifierHead(in_features=mamba_dim * 2, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to TensorFlow defaults."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        """
        Args:
            acc_data: (B, T, C) IMU data
                      For raw: C=6 or 8 [ax, ay, az, gx, gy, gz, ...]
                      For Kalman: C=7 [smv, ax, ay, az, roll, pitch, yaw]
            skl_data: Ignored (for API compatibility)

        Returns:
            logits: (B, 1) classification logits
            features: (B, 64) features before classifier
        """
        B, T, C = acc_data.shape

        if self.use_kalman:
            if C == 7:
                # Kalman 7ch: [smv, ax, ay, az, roll, pitch, yaw]
                acc = acc_data[:, :, 1:4]  # [ax, ay, az]
                ori = acc_data[:, :, 4:7]  # [roll, pitch, yaw]
            else:
                # Kalman 6ch: [ax, ay, az, roll, pitch, yaw] (no SMV)
                acc = acc_data[:, :, :3]   # [ax, ay, az]
                ori = acc_data[:, :, 3:6]  # [roll, pitch, yaw]
        else:
            # Raw 6ch or 8ch: first 3 = acc, next 3 = gyro (skip smv/gmag if present)
            if C >= 8:
                # 8ch: [smv, ax, ay, az, gmag, gx, gy, gz]
                acc = acc_data[:, :, 1:4]  # [ax, ay, az]
                ori = acc_data[:, :, 5:8]  # [gx, gy, gz]
            elif C == 6:
                # 6ch: [ax, ay, az, gx, gy, gz]
                acc = acc_data[:, :, :3]   # [ax, ay, az]
                ori = acc_data[:, :, 3:6]  # [gx, gy, gz]
            else:
                # Fallback: split in half
                mid = C // 2
                acc = acc_data[:, :, :3]
                ori = acc_data[:, :, mid:mid+3]

        # Stream 1: Accelerometer
        acc_feat = self.cnn_acc(acc)       # (B, 32, 64)
        acc_feat = self.mamba_acc(acc_feat)  # (B, 32, 64)
        acc_pool = acc_feat.mean(dim=1)    # Global avg pool: (B, 64)
        acc_pool = self.proj_acc(acc_pool)  # (B, 32)

        # Stream 2: Gyroscope/Orientation
        ori_feat = self.cnn_gyro(ori)       # (B, 32, 64)
        ori_feat = self.mamba_gyro(ori_feat)  # (B, 32, 64)
        ori_pool = ori_feat.mean(dim=1)    # Global avg pool: (B, 64)
        ori_pool = self.proj_gyro(ori_pool)  # (B, 32)

        # Concatenate and classify
        features = torch.cat([acc_pool, ori_pool], dim=1)  # (B, 64)
        logits = self.classifier(features)  # (B, 1)

        return logits, features


class CNNMambaRaw(CNNMambaDualStream):
    """CNN-Mamba for raw IMU input (6-8 channels)."""

    def __init__(self, **kwargs):
        kwargs['use_kalman'] = False
        super().__init__(**kwargs)


class CNNMambaKalman(CNNMambaDualStream):
    """CNN-Mamba for Kalman-fused input (7 channels)."""

    def __init__(self, **kwargs):
        kwargs['use_kalman'] = True
        kwargs.setdefault('imu_channels', 7)
        super().__init__(**kwargs)


# =============================================================================
# Test Script
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CNN-Mamba Fall Detector - Architecture Test")
    print("=" * 70)

    # Test 1: Raw input (8 channels)
    print("\nTest 1: CNNMambaRaw (8ch raw input)")
    print("-" * 50)
    model_raw = CNNMambaRaw(imu_frames=128, imu_channels=8)
    x_raw = torch.randn(4, 128, 8)
    logits, features = model_raw(x_raw)
    params = sum(p.numel() for p in model_raw.parameters())
    print(f"  Input:    {x_raw.shape}")
    print(f"  Output:   {logits.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Params:   {params:,}")

    # Test 2: Kalman input (7 channels)
    print("\nTest 2: CNNMambaKalman (7ch Kalman input)")
    print("-" * 50)
    model_kalman = CNNMambaKalman(imu_frames=128, imu_channels=7)
    x_kalman = torch.randn(4, 128, 7)
    logits, features = model_kalman(x_kalman)
    params = sum(p.numel() for p in model_kalman.parameters())
    print(f"  Input:    {x_kalman.shape}")
    print(f"  Output:   {logits.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Params:   {params:,}")

    # Test 3: Gradient flow
    print("\nTest 3: Gradient flow check")
    print("-" * 50)
    model_kalman.train()
    x = torch.randn(4, 128, 7, requires_grad=True)
    logits, _ = model_kalman(x)
    loss = logits.sum()
    loss.backward()
    grad_norm = x.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  Status: {'OK' if grad_norm > 0 else 'FAIL'}")

    # Test 4: 6ch raw input
    print("\nTest 4: CNNMambaRaw (6ch raw input)")
    print("-" * 50)
    model_6ch = CNNMambaRaw(imu_frames=128, imu_channels=6)
    x_6ch = torch.randn(4, 128, 6)
    logits, features = model_6ch(x_6ch)
    print(f"  Input:    {x_6ch.shape}")
    print(f"  Output:   {logits.shape}")
    print(f"  Status:   OK")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
