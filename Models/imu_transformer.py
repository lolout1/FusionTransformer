import torch
from torch import nn
from typing import Dict, Tuple, Optional
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


def get_optimal_config(num_channels: int) -> Dict[str, int]:
    """Return a small preset of transformer hyperparameters per channel count."""
    if num_channels <= 4:
        config = {
            'num_heads': 4,
            'num_layers': 2,
            'embed_dim': 64,
            'dim_feedforward': 128
        }
    elif num_channels <= 6:
        config = {
            'num_heads': 4,
            'num_layers': 2,
            'embed_dim': 80,
            'dim_feedforward': 160
        }
    elif num_channels == 7:
        config = {
            'num_heads': 4,
            'num_layers': 3,
            'embed_dim': 96,
            'dim_feedforward': 192
        }
    else:
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
    """Transformer backbone with optional auto-tuning per channel count."""
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

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        if auto_tune and (num_heads is None or num_layers is None or
                          embed_dim is None or dim_feedforward is None):
            optimal_config = get_optimal_config(self.imu_channels)

            num_heads = num_heads if num_heads is not None else optimal_config['num_heads']
            num_layers = num_layers if num_layers is not None else optimal_config['num_layers']
            embed_dim = embed_dim if embed_dim is not None else optimal_config['embed_dim']
            dim_feedforward = dim_feedforward if dim_feedforward is not None else optimal_config['dim_feedforward']
        else:
            num_heads = num_heads if num_heads is not None else 4
            num_layers = num_layers if num_layers is not None else 2
            embed_dim = embed_dim if embed_dim is not None else 64
            dim_feedforward = dim_feedforward if dim_feedforward is not None else embed_dim * 2

        self.input_proj = nn.Sequential(
            nn.Conv1d(self.imu_channels, embed_dim, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout * 0.5)
        )

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

        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        nn.init.normal_(self.output.weight, 0, math.sqrt(2. / num_classes))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data, skl_data=None, **kwargs):
        """Return logits and per-frame features for an IMU clip."""
        x = rearrange(acc_data, 'b t c -> b c t')
        x = self.input_proj(x)
        x = rearrange(x, 'b c t -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')
        x = self.temporal_norm(x)
        features = x
        x = rearrange(x, 'b t c -> b c t')
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride=1)
        x = rearrange(x, 'b c t -> b (c t)')
        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


class IMUTransformerLight(nn.Module):
    """Smaller variant for aggressive regularization."""
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

        self.input_proj = nn.Sequential(
            nn.Conv1d(imu_channels, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        self.encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=2,
            dim_feedforward=embed_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

    def forward(self, acc_data, skl_data=None, **kwargs):
        x = rearrange(acc_data, 'b t c -> b c t')
        x = self.input_proj(x)

        x = rearrange(x, 'b c t -> b t c')
        x = self.encoder_layer(x)
        x = self.norm(x)

        features = x
        x = torch.mean(x, dim=1)

        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


if __name__ == "__main__":
    batch_size = 16
    seq_len = 128
    imu_channels = 7

    imu_data = torch.randn(batch_size, seq_len, imu_channels)
    skl_data = torch.randn(batch_size, seq_len, 32, 3)

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

    model_light = IMUTransformerLight(
        imu_frames=seq_len,
        imu_channels=imu_channels,
        num_classes=2,
        embed_dim=16
    )

    logits_light, features_light = model_light(imu_data, skl_data)
    print(f"\nLightweight model parameters: {sum(p.numel() for p in model_light.parameters()):,}")
