import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


class DualStreamLightIMU(nn.Module):
    """Lightweight dual-stream IMU encoder with late feature fusion."""

    def __init__(self,
                 imu_frames: int = 128,
                 mocap_frames: int = 128,  # For compatibility, not used
                 num_joints: int = 32,     # For compatibility, not used
                 acc_frames: int = 128,    # Alias for imu_frames
                 imu_channels: int = 6,
                 acc_coords: int = 6,      # Alias for imu_channels
                 num_classes: int = 1,
                 num_heads: int = 2,
                 num_layers: int = 1,
                 stream_dim: int = 8,      # Very small embedding per stream
                 embed_dim: int = 8,       # Alias for stream_dim
                 dropout: float = 0.65,
                 activation: str = 'relu',
                 norm_first: bool = True,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        self.stream_dim = stream_dim if stream_dim != 8 or embed_dim == 8 else embed_dim

        assert self.imu_channels == 6, "DualStreamLightIMU requires 6 channels (3 acc + 3 gyro)"

        self.acc_encoder = nn.Sequential(
            nn.Conv1d(3, self.stream_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(self.stream_dim),
            nn.Dropout(dropout * 0.4)
        )

        self.gyro_encoder = nn.Sequential(
            nn.Conv1d(3, self.stream_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(self.stream_dim),
            nn.Dropout(dropout * 0.4)
        )

        self.acc_transformer = TransformerEncoderLayer(
            d_model=self.stream_dim,
            nhead=num_heads,
            dim_feedforward=self.stream_dim,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )

        self.gyro_transformer = TransformerEncoderLayer(
            d_model=self.stream_dim,
            nhead=num_heads,
            dim_feedforward=self.stream_dim,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )

        self.acc_norm = nn.LayerNorm(self.stream_dim)
        self.gyro_norm = nn.LayerNorm(self.stream_dim)

        fusion_input_dim = self.stream_dim * 2

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.stream_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.stream_dim * 2, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, acc_data, skl_data=None, **kwargs):
        """Return logits and fused representations for accelerometer + gyro."""
        acc = acc_data[:, :, :3]
        gyro = acc_data[:, :, 3:]

        acc_x = rearrange(acc, 'b t c -> b c t')
        acc_x = self.acc_encoder(acc_x)
        acc_x = rearrange(acc_x, 'b c t -> t b c')
        acc_feat = self.acc_transformer(acc_x)
        acc_feat = rearrange(acc_feat, 't b c -> b t c')
        acc_feat = self.acc_norm(acc_feat)
        acc_feat = torch.mean(acc_feat, dim=1)

        gyro_x = rearrange(gyro, 'b t c -> b c t')
        gyro_x = self.gyro_encoder(gyro_x)
        gyro_x = rearrange(gyro_x, 'b c t -> t b c')
        gyro_feat = self.gyro_transformer(gyro_x)
        gyro_feat = rearrange(gyro_feat, 't b c -> b t c')
        gyro_feat = self.gyro_norm(gyro_feat)
        gyro_feat = torch.mean(gyro_feat, dim=1)

        features = torch.cat([acc_feat, gyro_feat], dim=-1)
        logits = self.fusion(features)

        return logits, features


if __name__ == "__main__":
    batch_size = 16
    seq_len = 128
    imu_channels = 6

    imu_data = torch.randn(batch_size, seq_len, imu_channels)

    model = DualStreamLightIMU(
        imu_frames=seq_len,
        imu_channels=imu_channels,
        num_classes=1,
        num_layers=1,
        stream_dim=8,
        num_heads=2,
        dropout=0.65
    )

    logits, features = model(imu_data)

    print(f"Input shape: {imu_data.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nParameter breakdown:")
    print(f"  Acc encoder: {sum(p.numel() for p in model.acc_encoder.parameters()):,}")
    print(f"  Gyro encoder: {sum(p.numel() for p in model.gyro_encoder.parameters()):,}")
    print(f"  Acc transformer: {sum(p.numel() for p in model.acc_transformer.parameters()):,}")
    print(f"  Gyro transformer: {sum(p.numel() for p in model.gyro_transformer.parameters()):,}")
    print(f"  Fusion layer: {sum(p.numel() for p in model.fusion.parameters()):,}")
