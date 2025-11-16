import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


class DualStreamAsymmetricIMU(nn.Module):
    """Dual-stream IMU encoder with asymmetric capacity and learnable fusion."""

    def __init__(self,
                 imu_frames: int = 128,
                 mocap_frames: int = 128,  # For compatibility, not used
                 num_joints: int = 32,     # For compatibility, not used
                 acc_frames: int = 128,    # Alias for imu_frames
                 imu_channels: int = 6,
                 acc_coords: int = 6,      # Alias for imu_channels
                 num_classes: int = 1,
                 num_heads: int = 2,
                 num_layers: int = 2,      # Used for acc_layers if acc_layers not specified
                 acc_layers: int = None,   # Separate control for acc layers
                 gyro_layers: int = 1,
                 acc_dim: int = 16,
                 gyro_dim: int = 8,
                 embed_dim: int = 16,      # Fallback if acc_dim not specified
                 dropout: float = 0.6,
                 activation: str = 'relu',
                 norm_first: bool = True,
                 **kwargs):
        super().__init__()

        # Handle both old and new parameter names
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Use provided acc_dim or fall back to embed_dim or default
        if acc_dim == 16 and embed_dim != 16:
            self.acc_dim = embed_dim
        else:
            self.acc_dim = acc_dim

        self.gyro_dim = gyro_dim

        # Use provided acc_layers or fall back to num_layers
        if acc_layers is None:
            self.acc_layers = num_layers
        else:
            self.acc_layers = acc_layers

        self.gyro_layers = gyro_layers

        assert self.imu_channels == 6, "DualStreamAsymmetricIMU requires 6 channels (3 acc + 3 gyro)"

        # ========== Accelerometer Branch (Primary, Larger) ==========
        self.acc_encoder = nn.Sequential(
            nn.Conv1d(3, self.acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(self.acc_dim),
            nn.Dropout(dropout * 0.4)
        )

        acc_encoder_layer = TransformerEncoderLayer(
            d_model=self.acc_dim,
            nhead=num_heads,
            dim_feedforward=self.acc_dim * 2,  # Larger feedforward for primary branch
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )

        self.acc_transformer = TransformerEncoder(
            acc_encoder_layer,
            num_layers=self.acc_layers,
            norm=nn.LayerNorm(self.acc_dim)
        )

        # ========== Gyroscope Branch (Auxiliary, Smaller) ==========
        self.gyro_encoder = nn.Sequential(
            nn.Conv1d(3, self.gyro_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(self.gyro_dim),
            nn.Dropout(dropout * 0.4)
        )

        gyro_encoder_layer = TransformerEncoderLayer(
            d_model=self.gyro_dim,
            nhead=2,
            dim_feedforward=self.gyro_dim,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )

        self.gyro_transformer = TransformerEncoder(
            gyro_encoder_layer,
            num_layers=self.gyro_layers,
            norm=nn.LayerNorm(self.gyro_dim)
        )

        self.acc_weight = nn.Parameter(torch.tensor(0.7))
        self.gyro_weight = nn.Parameter(torch.tensor(0.3))

        fusion_input_dim = self.acc_dim + self.gyro_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.acc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.acc_dim, num_classes)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, math.sqrt(2. / m.out_features))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, acc_data, skl_data=None, **kwargs):
        """Return logits and fused features for accelerometer + gyro inputs."""
        acc = acc_data[:, :, :3]
        gyro = acc_data[:, :, 3:]

        acc_x = rearrange(acc, 'b t c -> b c t')
        acc_x = self.acc_encoder(acc_x)
        acc_x = rearrange(acc_x, 'b c t -> t b c')
        acc_feat = self.acc_transformer(acc_x)
        acc_feat = rearrange(acc_feat, 't b c -> b t c')
        acc_feat = torch.mean(acc_feat, dim=1)

        gyro_x = rearrange(gyro, 'b t c -> b c t')
        gyro_x = self.gyro_encoder(gyro_x)
        gyro_x = rearrange(gyro_x, 'b c t -> t b c')
        gyro_feat = self.gyro_transformer(gyro_x)
        gyro_feat = rearrange(gyro_feat, 't b c -> b t c')
        gyro_feat = torch.mean(gyro_feat, dim=1)

        total_weight = torch.abs(self.acc_weight) + torch.abs(self.gyro_weight)
        normalized_acc_weight = torch.abs(self.acc_weight) / total_weight
        normalized_gyro_weight = torch.abs(self.gyro_weight) / total_weight

        weighted_acc_feat = acc_feat * normalized_acc_weight
        weighted_gyro_feat = gyro_feat * normalized_gyro_weight

        features = torch.cat([weighted_acc_feat, weighted_gyro_feat], dim=-1)

        logits = self.fusion(features)

        return logits, features

    def get_fusion_weights(self):
        """Return normalized fusion weights for each stream."""
        total_weight = torch.abs(self.acc_weight) + torch.abs(self.gyro_weight)
        return {
            'acc_weight': (torch.abs(self.acc_weight) / total_weight).item(),
            'gyro_weight': (torch.abs(self.gyro_weight) / total_weight).item()
        }


if __name__ == "__main__":
    batch_size = 16
    seq_len = 128
    imu_channels = 6

    imu_data = torch.randn(batch_size, seq_len, imu_channels)

    model = DualStreamAsymmetricIMU(
        imu_frames=seq_len,
        imu_channels=imu_channels,
        num_classes=1,
        acc_layers=2,
        gyro_layers=1,
        acc_dim=16,
        gyro_dim=8,
        num_heads=2,
        dropout=0.6
    )

    logits, features = model(imu_data)

    print(f"Input shape: {imu_data.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nLearned fusion weights: {model.get_fusion_weights()}")

    print("\nParameter breakdown:")
    print(f"  Acc encoder: {sum(p.numel() for p in model.acc_encoder.parameters()):,}")
    print(f"  Gyro encoder: {sum(p.numel() for p in model.gyro_encoder.parameters()):,}")
    print(f"  Acc transformer: {sum(p.numel() for p in model.acc_transformer.parameters()):,}")
    print(f"  Gyro transformer: {sum(p.numel() for p in model.gyro_transformer.parameters()):,}")
    print(f"  Fusion layer: {sum(p.numel() for p in model.fusion.parameters()):,}")
