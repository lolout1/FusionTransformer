import torch
from torch import nn
from typing import Dict, Tuple, Optional
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


def get_optimal_config(num_channels: int) -> Dict[str, int]:
    """
    Determine optimal architecture hyperparameters based on input modality complexity.

    Configurations are tuned based on the number of input channels, which reflects
    the feature richness and complexity of the IMU representation:

    - 4 channels (accelerometer-only): Baseline with SMV
        Architecture: Lightweight (4 heads, 2 layers, 64 dim)
        Rationale: Simple feature space requires smaller capacity

    - 6 channels (raw acc + gyro): Direct concatenation
        Architecture: Moderate (4 heads, 2 layers, 80 dim)
        Rationale: Added gyro increases feature space but no orientation

    - 7 channels (acc + orientation): Sensor fusion with Madgwick/Complementary
        Architecture: Enhanced (4 heads, 3 layers, 96 dim)
        Rationale: Orientation features (roll, pitch, yaw) capture device pose,
                   requiring deeper network for complex temporal patterns
        References: Zhang et al. (2024) DOI: 10.3390/app14093637
                    (97.13% accuracy using Madgwick + ResNet)

    - 8+ channels (engineered features): Full feature engineering
        Architecture: Maximum (8 heads, 3 layers, 128 dim)
        Rationale: Rich feature space (SMV, magnitude, angles) benefits from
                   increased model capacity for feature interaction learning

    Args:
        num_channels: Number of input channels in IMU data

    Returns:
        config: Dictionary with optimal hyperparameters
            - num_heads: Number of attention heads
            - num_layers: Number of transformer encoder layers
            - embed_dim: Embedding dimension
            - dim_feedforward: Feedforward network dimension (2x embed_dim)

    Example:
        >>> config = get_optimal_config(7)  # acc + orientation
        >>> print(f"Using {config['num_heads']} heads, {config['num_layers']} layers")
        Using 4 heads, 3 layers

    References:
        Vaswani et al. (2017). "Attention is All You Need." NeurIPS.
        Zhang et al. (2024). "Human Activity Recognition Based on Deep Learning
        Regardless of Sensor Orientation." Applied Sciences, 14(9), 3637.
        DOI: 10.3390/app14093637
    """
    if num_channels <= 4:
        # Accelerometer-only with SMV [smv, ax, ay, az]
        # Lightweight architecture sufficient for simple feature space
        config = {
            'num_heads': 4,
            'num_layers': 2,
            'embed_dim': 64,
            'dim_feedforward': 128
        }
    elif num_channels <= 6:
        # Raw acc + gyro [ax, ay, az, gx, gy, gz]
        # Moderate capacity for direct sensor concatenation
        config = {
            'num_heads': 4,
            'num_layers': 2,
            'embed_dim': 80,
            'dim_feedforward': 160
        }
    elif num_channels == 7:
        # Acc + orientation from sensor fusion [smv, ax, ay, az, roll, pitch, yaw]
        # Enhanced architecture for orientation-aware features
        # Zhang et al. (2024) showed Madgwick fusion improves recognition
        config = {
            'num_heads': 4,
            'num_layers': 3,
            'embed_dim': 96,
            'dim_feedforward': 192
        }
    else:  # 8+ channels
        # Full engineered features [acc_smv, ax, ay, az, gyro_mag, gx, gy, gz]
        # Maximum capacity for rich feature interactions
        config = {
            'num_heads': 8,
            'num_layers': 3,
            'embed_dim': 128,
            'dim_feedforward': 256
        }

    return config


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
    IMU Transformer for Activity Recognition with Auto-Tuned Architecture

    Automatically selects optimal architecture based on input modality complexity.
    Supports multiple IMU configurations with channel-specific tuning:

    Supported configurations (auto-tuned):
    - 4 channels (acc-only): [smv, ax, ay, az]
        Auto config: 4 heads, 2 layers, 64 dim
    - 6 channels (raw acc+gyro): [ax, ay, az, gx, gy, gz]
        Auto config: 4 heads, 2 layers, 80 dim
    - 7 channels (acc+orientation): [smv, ax, ay, az, roll, pitch, yaw]
        Auto config: 4 heads, 3 layers, 96 dim
        (orientation from Madgwick/Complementary sensor fusion)
    - 8 channels (engineered features): [acc_smv, ax, ay, az, gyro_mag, gx, gy, gz]
        Auto config: 8 heads, 3 layers, 128 dim

    Args:
        imu_frames: Number of time steps (default: 128)
        imu_channels: Number of IMU channels (default: 8)
        num_classes: Number of output classes (default: 2 for binary fall detection)
        num_heads: Number of attention heads (default: None, auto-tuned based on channels)
        num_layers: Number of transformer layers (default: None, auto-tuned based on channels)
        embed_dim: Embedding dimension (default: None, auto-tuned based on channels)
        dim_feedforward: Feedforward dimension (default: None, auto-tuned based on channels)
        dropout: Dropout rate (default: 0.5)
        activation: Activation function (default: 'relu')
        norm_first: Whether to apply normalization first (default: True)
        auto_tune: Enable automatic architecture tuning (default: True)
            Set to False to use explicit parameters or legacy defaults

    References:
        Vaswani et al. (2017). "Attention is All You Need." NeurIPS.
        Zhang et al. (2024). "Human Activity Recognition Based on Deep Learning
        Regardless of Sensor Orientation." Applied Sciences, 14(9), 3637.
        DOI: 10.3390/app14093637

    Example:
        >>> # Auto-tuned for 7-channel input (acc + orientation)
        >>> model = IMUTransformer(imu_channels=7)
        >>> # Uses: 4 heads, 3 layers, 96 dim (auto-selected)
        >>>
        >>> # Manual override (disable auto-tuning)
        >>> model = IMUTransformer(imu_channels=7, num_heads=8, num_layers=4,
        ...                        embed_dim=128, auto_tune=False)
    """
    def __init__(self,
                 imu_frames: int = 128,
                 mocap_frames: int = 128,  # For compatibility, not used
                 num_joints: int = 32,     # For compatibility, not used
                 acc_frames: int = 128,    # Alias for imu_frames for compatibility
                 imu_channels: int = 8,    # 8 channels: acc_smv, ax, ay, az, gyro_mag, gx, gy, gz
                 acc_coords: int = 8,      # Alias for imu_channels for compatibility
                 num_classes: int = 2,     # Matching TransModel default
                 num_heads: Optional[int] = None,       # Auto-tuned based on channels if None
                 num_layers: Optional[int] = None,      # Auto-tuned based on channels if None
                 embed_dim: Optional[int] = None,       # Auto-tuned based on channels if None
                 dim_feedforward: Optional[int] = None,  # Auto-tuned based on channels if None
                 dropout: float = 0.5,     # Matching TransModel
                 activation: str = 'relu',
                 norm_first: bool = True,
                 auto_tune: bool = True,   # Enable auto-tuning based on channel count
                 **kwargs):
        super().__init__()

        # Handle both old and new parameter names for backward compatibility
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Auto-tune architecture based on channel count if parameters not explicitly provided
        if auto_tune and (num_heads is None or num_layers is None or
                          embed_dim is None or dim_feedforward is None):
            optimal_config = get_optimal_config(self.imu_channels)

            # Use optimal config for any parameter that wasn't explicitly provided
            num_heads = num_heads if num_heads is not None else optimal_config['num_heads']
            num_layers = num_layers if num_layers is not None else optimal_config['num_layers']
            embed_dim = embed_dim if embed_dim is not None else optimal_config['embed_dim']
            dim_feedforward = dim_feedforward if dim_feedforward is not None else optimal_config['dim_feedforward']
        else:
            # Backward compatibility: use defaults if not auto-tuning
            num_heads = num_heads if num_heads is not None else 4
            num_layers = num_layers if num_layers is not None else 2
            embed_dim = embed_dim if embed_dim is not None else 64
            dim_feedforward = dim_feedforward if dim_feedforward is not None else embed_dim * 2

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
            dim_feedforward=dim_feedforward,
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
                      Expected: (B, 128, 8) where channels = [acc_smv, ax, ay, az, gyro_mag, gx, gy, gz]
                      Also supports 7ch: [ax, ay, az, gx, gy, gz, smv] or 6ch: [ax, ay, az, gx, gy, gz]
            skl_data: Skeleton data (not used, for compatibility)

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
    # Test with 7-channel IMU data (acc + gyro + smv)
    batch_size = 16
    seq_len = 128
    imu_channels = 7  # ax, ay, az, gx, gy, gz, smv

    imu_data = torch.randn(batch_size, seq_len, imu_channels)
    skl_data = torch.randn(batch_size, seq_len, 32, 3)  # Dummy skeleton data

    # Test standard model (matching TransModel parameters)
    model = IMUTransformer(
        imu_frames=seq_len,
        imu_channels=imu_channels,
        num_classes=2,   # Matching TransModel default
        num_layers=2,
        embed_dim=64,    # Matching TransModel (was 32)
        num_heads=4,     # Matching TransModel (was 2)
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
        num_classes=2,
        embed_dim=16
    )

    logits_light, features_light = model_light(imu_data, skl_data)
    print(f"\nLightweight model parameters: {sum(p.numel() for p in model_light.parameters()):,}")
