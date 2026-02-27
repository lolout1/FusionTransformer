"""Basic transformer model for accelerometer-only baseline (no SE, no TAP)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import TransformerEncoderLayer


class TransformerEncoderWAttention(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output, attn = layer.self_attn(
                output, output, output,
                attn_mask=mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True
            )
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TransModel(nn.Module):
    def __init__(
        self,
        mocap_frames: int = 128,
        num_joints: int = 32,
        acc_frames: int = 128,
        imu_frames: int = 128,
        num_classes: int = 1,
        num_heads: int = 4,
        acc_coords: int = 4,
        imu_channels: int = 4,
        av: bool = False,
        num_layer: int = 2,
        num_layers: int = 2,
        norm_first: bool = True,
        embed_dim: int = 48,
        activation: str = 'relu',
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()

        # Handle both naming conventions
        frames = imu_frames or acc_frames
        channels = imu_channels or acc_coords
        layers = num_layers or num_layer

        self.data_shape = (frames, channels)
        self.length = self.data_shape[0]
        self.channels = channels  # Store for input slicing

        self.input_proj = nn.Sequential(
            nn.Conv1d(channels, embed_dim, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(embed_dim)
        )

        self.encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            activation=activation,
            dim_feedforward=embed_dim * 2,
            nhead=num_heads,
            dropout=dropout,
            norm_first=norm_first,
            batch_first=False
        )

        self.encoder = TransformerEncoderWAttention(
            encoder_layer=self.encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(embed_dim)
        )

        self.output = nn.Linear(embed_dim, num_classes)
        self.temporal_norm = nn.LayerNorm(embed_dim)

    def forward(self, acc_data, skl_data=None, **kwargs):
        # Slice to expected channels (handles case where loader provides more channels)
        x = acc_data[:, :, :self.channels]
        x = rearrange(x, 'b l c -> b c l')
        x = self.input_proj(x)
        x = rearrange(x, 'b c l -> l b c')
        x = self.encoder(x)
        x = rearrange(x, 'l b c -> b l c')

        x = self.temporal_norm(x)
        feature = x

        x = rearrange(x, 'b f c -> b c f')
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride=1)
        x = rearrange(x, 'b c f -> b (c f)')
        x = self.output(x)

        return x, feature


if __name__ == "__main__":
    data = torch.randn(size=(16, 128, 4))
    model = TransModel()
    output, feature = model(data)
    print(f"Output shape: {output.shape}, Feature shape: {feature.shape}")
