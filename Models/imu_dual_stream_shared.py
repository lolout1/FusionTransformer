"""
Option 1: Shared-Weight Dual Stream IMU Transformer
Minimizes overfitting through weight sharing between accelerometer and gyroscope streams
Optimal for limited training data (e.g., 12-30 subjects)
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


class DualStreamSharedIMU(nn.Module):
    """
    Shared-weight dual-stream architecture for accelerometer + gyroscope

    Key Features:
    - Shares encoder weights between acc and gyro (weight sharing = regularization)
    - lightweight design (~15K parameters with embed_dim=16)
    - Separate processing paths but shared parameters

    Args:
        imu_frames: Number of time steps (default: 128)
        imu_channels: Total IMU channels, must be 6 for acc+gyro (default: 6)
        num_classes: Number of output classes (1 for binary fall detection)
        num_heads: Number of attention heads (default: 2)
        num_layers: Number of transformer layers (default: 1)
        embed_dim: Embedding dimension (default: 16, kept very small)
        dropout: Dropout rate (default: 0.65 for strong regularization)
        activation: Activation function (default: 'relu')
        norm_first: Whether to apply normalization first (default: True)
    """

    def __init__(self,
                 imu_frames: int = 128,
                 mocap_frames: int = 128,  # For compatibility, not used
                 num_joints: int = 32,     # For compatibility, not used
                 acc_frames: int = 128,    
                 imu_channels: int = 6,
                 acc_coords: int = 6,      
                 num_classes: int = 1,
                 num_heads: int = 2,
                 num_layers: int = 1,
                 embed_dim: int = 16,
                 dropout: float = 0.65,
                 activation: str = 'relu',
                 norm_first: bool = True,
                 **kwargs):
        super().__init__()

        # Handle both old and new parameter names for backward compatibility
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.embed_dim = embed_dim

        assert self.imu_channels == 6, "DualStreamSharedIMU requires 6 channels (3 acc + 3 gyro)"

        # Shared encoder for both modalities (3 channels -> embed_dim)
        # Weight sharing is the key to preventing overfitting
        self.shared_encoder = nn.Sequential(
            nn.Conv1d(3, embed_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout * 0.4)  # Lighter dropout on input
        )

        # Shared transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim,  # Keep FFN dimension very small
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )

        self.shared_transformer = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # Fusion layer: combines features from both streams
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        # Initialize fusion layer with small weights
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, acc_data, skl_data=None, **kwargs):
        """
        Forward pass

        Args:
            acc_data: IMU data tensor of shape (batch, time, 6)
                      Channels = [ax, ay, az, gx, gy, gz]
            skl_data: Skeleton data (not used, for compatibility)

        Returns:
            logits: Output predictions (B, num_classes)
            features: Concatenated features (B, embed_dim * 2) for distillation
        """
        # Split into accelerometer and gyroscope
        # Input: (B, T, 6)
        acc = acc_data[:, :, :3]   # (B, T, 3) - accelerometer
        gyro = acc_data[:, :, 3:]  # (B, T, 3) - gyroscope

        # Process accelerometer stream through shared encoder
        # (B, T, 3) -> (B, 3, T)
        acc_x = rearrange(acc, 'b t c -> b c t')
        acc_x = self.shared_encoder(acc_x)  # (B, embed_dim, T)

        # Prepare for transformer: (T, B, embed_dim)
        acc_x = rearrange(acc_x, 'b c t -> t b c')
        acc_feat = self.shared_transformer(acc_x)  # (T, B, embed_dim)

        # Back to (B, T, embed_dim)
        acc_feat = rearrange(acc_feat, 't b c -> b t c')

        # Global average pooling over time: (B, T, embed_dim) -> (B, embed_dim)
        acc_feat = torch.mean(acc_feat, dim=1)

        # Process gyroscope stream through SAME shared encoder
        # (B, T, 3) -> (B, 3, T)
        gyro_x = rearrange(gyro, 'b t c -> b c t')
        gyro_x = self.shared_encoder(gyro_x)  # (B, embed_dim, T)

        # Prepare for transformer: (T, B, embed_dim)
        gyro_x = rearrange(gyro_x, 'b c t -> t b c')
        gyro_feat = self.shared_transformer(gyro_x)  # (T, B, embed_dim)

        # Back to (B, T, embed_dim)
        gyro_feat = rearrange(gyro_feat, 't b c -> b t c')

        # Global average pooling over time: (B, T, embed_dim) -> (B, embed_dim)
        gyro_feat = torch.mean(gyro_feat, dim=1)

        # Concatenate features from both streams
        features = torch.cat([acc_feat, gyro_feat], dim=-1)  # (B, embed_dim * 2)

        # Final classification through fusion layer
        logits = self.fusion(features)

        return logits, features


# Test the model
if __name__ == "__main__":
    batch_size = 16
    seq_len = 128
    imu_channels = 6  # ax, ay, az, gx, gy, gz

    imu_data = torch.randn(batch_size, seq_len, imu_channels)

    model = DualStreamSharedIMU(
        imu_frames=seq_len,
        imu_channels=imu_channels,
        num_classes=1,
        num_layers=1,
        embed_dim=16,
        num_heads=2,
        dropout=0.65
    )

    logits, features = model(imu_data)

    print(f"Input shape: {imu_data.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print parameter breakdown
    print("\nParameter breakdown:")
    print(f"  Shared encoder: {sum(p.numel() for p in model.shared_encoder.parameters()):,}")
    print(f"  Shared transformer: {sum(p.numel() for p in model.shared_transformer.parameters()):,}")
    print(f"  Fusion layer: {sum(p.numel() for p in model.fusion.parameters()):,}")
