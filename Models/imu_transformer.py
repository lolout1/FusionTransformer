import torch
from torch import nn
from typing import Dict, Tuple
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


class TransformerEncoderWAttention(nn.TransformerEncoder):
    """Transformer encoder with attention weight tracking"""
    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = src
        for layer in self.layers:
            output, attn = layer.self_attn(output, output, output, attn_mask = mask,
                                            key_padding_mask = src_key_padding_mask, need_weights = True)
            output = layer(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
        return output


class IMUTransformer(nn.Module):
    """
    Lightweight IMU Transformer for Fall Detection

    Optimized for combined accelerometer + gyroscope data (6 channels: ax, ay, az, gx, gy, gz)
    Designed to minimize overfitting with limited training data (~1000 trials)
    Suitable for real-time inference on Android devices with sliding window

    Args:
        imu_frames: Number of time steps (default: 128)
        imu_channels: Number of IMU channels - 6 for acc+gyro (default: 6)
        num_classes: Number of output classes (1 for binary fall detection)
        num_heads: Number of attention heads (default: 2)
        num_layers: Number of transformer layers (default: 1-2 for simplicity)
        embed_dim: Embedding dimension (default: 16-32, kept small to prevent overfitting)
        dropout: Dropout rate (default: 0.5 for regularization)
        activation: Activation function (default: 'relu')
    """
    def __init__(self,
                 imu_frames: int = 128,
                 mocap_frames: int = 128,  # For compatibility, not used
                 num_joints: int = 32,     # For compatibility, not used
                 acc_frames: int = 128,    # Alias for imu_frames for compatibility
                 imu_channels: int = 6,
                 acc_coords: int = 6,      # Alias for imu_channels for compatibility
                 num_classes: int = 1,
                 num_heads: int = 2,
                 num_layers: int = 2,
                 embed_dim: int = 32,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 norm_first: bool = True,
                 **kwargs):
        super().__init__()

        # Handle both old and new parameter names for backward compatibility
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Simple 1D convolution for temporal embedding
        # kernel_size=8 with padding='same' maintains temporal resolution
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.imu_channels, embed_dim, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout * 0.5)  # Light dropout on input
        )

        # Lightweight transformer encoder
        # Using fewer layers and smaller feedforward dim to prevent overfitting
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,  # Keep feedforward small
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )

        self.encoder = TransformerEncoderWAttention(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # Output layers
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        # Initialize output layer with small weights
        nn.init.normal_(self.output.weight, 0, math.sqrt(2. / num_classes))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data, skl_data=None, **kwargs):
        """
        Forward pass

        Args:
            acc_data: IMU data tensor of shape (batch, time, channels)
                      For acc+gyro: (B, 128, 6) where channels = [ax, ay, az, gx, gy, gz]
                      For acc only: (B, 128, 4) where channels = [smv, ax, ay, az]
            skl_data: Skeleton data (not used, for compatibility with teacher model)

        Returns:
            logits: Output predictions (B, num_classes)
            features: Intermediate features (B, time, embed_dim) for distillation
        """
        # Input shape: (batch, time, channels)
        # Rearrange to (batch, channels, time) for Conv1d
        x = rearrange(acc_data, 'b t c -> b c t')

        # Project to embedding space: (B, embed_dim, time)
        x = self.input_proj(x)

        # Rearrange for transformer: (time, batch, embed_dim)
        x = rearrange(x, 'b c t -> t b c')

        # Transformer encoding
        x = self.encoder(x)

        # Rearrange back: (batch, time, embed_dim)
        x = rearrange(x, 't b c -> b t c')

        # Normalize features
        x = self.temporal_norm(x)

        # Store features for knowledge distillation
        features = x

        # Global average pooling over time dimension
        # (batch, time, embed_dim) -> (batch, embed_dim)
        x = rearrange(x, 'b t c -> b c t')
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride=1)
        x = rearrange(x, 'b c t -> b (c t)')

        # Apply dropout before final layer
        x = self.dropout(x)

        # Final classification
        logits = self.output(x)

        return logits, features


class IMUTransformerLight(nn.Module):
    """
    Ultra-lightweight version with even fewer parameters
    Use this if overfitting is severe
    """
    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 6,
                 num_classes: int = 1,
                 embed_dim: int = 16,
                 dropout: float = 0.6,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames
        self.imu_channels = imu_channels

        # Even simpler architecture
        self.input_proj = nn.Sequential(
            nn.Conv1d(imu_channels, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Single transformer layer
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=2,
            dim_feedforward=embed_dim,  # Very small feedforward
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

    def forward(self, acc_data, skl_data=None, **kwargs):
        # (B, T, C) -> (B, C, T)
        x = rearrange(acc_data, 'b t c -> b c t')
        x = self.input_proj(x)

        # (B, C, T) -> (B, T, C)
        x = rearrange(x, 'b c t -> b t c')
        x = self.encoder_layer(x)
        x = self.norm(x)

        # Global average pooling
        features = x
        x = torch.mean(x, dim=1)

        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


# Test the model
if __name__ == "__main__":
    # Test with 6-channel IMU data (acc + gyro)
    batch_size = 16
    seq_len = 128
    imu_channels = 6  # ax, ay, az, gx, gy, gz

    imu_data = torch.randn(batch_size, seq_len, imu_channels)
    skl_data = torch.randn(batch_size, seq_len, 32, 3)  # Dummy skeleton data

    # Test standard model
    model = IMUTransformer(
        imu_frames=seq_len,
        imu_channels=imu_channels,
        num_classes=1,
        num_layers=2,
        embed_dim=32,
        num_heads=2,
        dropout=0.5
    )

    logits, features = model(imu_data, skl_data)

    print(f"Input shape: {imu_data.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test lightweight model
    model_light = IMUTransformerLight(
        imu_frames=seq_len,
        imu_channels=imu_channels,
        num_classes=1,
        embed_dim=16
    )

    logits_light, features_light = model_light(imu_data, skl_data)
    print(f"\nLightweight model parameters: {sum(p.numel() for p in model_light.parameters()):,}")
