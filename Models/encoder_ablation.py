"""Kalman encoder ablation models."""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Literal, Optional


class SqueezeExcitation(nn.Module):
    """Channel attention."""

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
        scale = x.mean(dim=1)
        scale = self.fc(scale).unsqueeze(1)
        return x * scale


class TemporalAttentionPooling(nn.Module):
    """Temporal attention pooling."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        scores = self.attention(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.einsum('bt,btc->bc', weights, x)
        return context, weights


class TransformerEncoderWithNorm(nn.TransformerEncoder):
    """Transformer encoder with final norm."""

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Conv1DEncoder(nn.Module):
    """Conv1D temporal encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b t c -> b c t')
        x = self.encoder(x)
        x = rearrange(x, 'b c t -> b t c')
        return x


class MultiKernelConv1DEncoder(nn.Module):
    """Multi-kernel Conv1D encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (3, 5, 8, 13),
        dropout: float = 0.1
    ):
        super().__init__()
        n_kernels = len(kernel_sizes)
        ch_per_kernel = out_channels // n_kernels
        remainder = out_channels % n_kernels

        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            ch = ch_per_kernel + (1 if i < remainder else 0)
            self.convs.append(
                nn.Conv1d(in_channels, ch, kernel_size=k, padding='same')
            )

        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b t c -> b c t')

        features = [conv(x) for conv in self.convs]
        x = torch.cat(features, dim=1)

        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = rearrange(x, 'b c t -> b t c')
        return x


class LinearEncoder(nn.Module):
    """Per-timestep linear encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def create_encoder(
    encoder_type: Literal['conv1d', 'linear', 'multikernel'],
    in_channels: int,
    out_channels: int,
    kernel_size: int = 8,
    dropout: float = 0.1
) -> nn.Module:
    if encoder_type == 'conv1d':
        return Conv1DEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
    elif encoder_type == 'linear':
        return LinearEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout
        )
    elif encoder_type == 'multikernel':
        return MultiKernelConv1DEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=(3, 5, 8, 13),
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'conv1d', 'linear', or 'multikernel'.")


class KalmanEncoderAblation(nn.Module):
    """Configurable dual-stream encoder ablation model."""

    def __init__(
        self,
        acc_encoder: Literal['conv1d', 'linear'] = 'conv1d',
        ori_encoder: Literal['conv1d', 'linear'] = 'conv1d',
        acc_kernel_size: int = 8,
        ori_kernel_size: int = 8,
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
        acc_ratio: float = 0.65,
        **kwargs
    ):
        super().__init__()

        self.acc_encoder_type = acc_encoder
        self.ori_encoder_type = ori_encoder
        self.config_name = f"{acc_encoder}_{ori_encoder}"

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        self.acc_channels = 4
        self.ori_channels = self.imu_channels - 4

        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim
        self.acc_dim = acc_dim
        self.ori_dim = ori_dim

        self.acc_proj = create_encoder(
            encoder_type=acc_encoder,
            in_channels=self.acc_channels,
            out_channels=acc_dim,
            kernel_size=acc_kernel_size,
            dropout=dropout * 0.2
        )

        self.ori_proj = create_encoder(
            encoder_type=ori_encoder,
            in_channels=self.ori_channels,
            out_channels=ori_dim,
            kernel_size=ori_kernel_size,
            dropout=dropout * 0.3
        )

        self.fusion_norm = nn.LayerNorm(embed_dim)

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

        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)

        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        x = torch.cat([acc_feat, ori_feat], dim=-1)
        x = self.fusion_norm(x)

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)

        x = rearrange(x, 't b c -> b t c')

        x = self.se(x)
        features = x

        x, _ = self.temporal_pool(x)

        x = self.dropout(x)
        logits = self.output(x)

        return logits, features

    def get_encoder_info(self) -> dict:
        acc_params = sum(p.numel() for p in self.acc_proj.parameters())
        ori_params = sum(p.numel() for p in self.ori_proj.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'config_name': self.config_name,
            'acc_encoder': self.acc_encoder_type,
            'ori_encoder': self.ori_encoder_type,
            'acc_channels': self.acc_channels,
            'ori_channels': self.ori_channels,
            'acc_dim': self.acc_dim,
            'ori_dim': self.ori_dim,
            'acc_encoder_params': acc_params,
            'ori_encoder_params': ori_params,
            'total_params': total_params,
        }


class KalmanConv1dConv1d(KalmanEncoderAblation):
    """Conv1D for both streams."""
    def __init__(self, **kwargs):
        kwargs.pop('acc_encoder', None)
        kwargs.pop('ori_encoder', None)
        super().__init__(acc_encoder='conv1d', ori_encoder='conv1d', **kwargs)


class KalmanConv1dLinear(KalmanEncoderAblation):
    """Conv1D for acc, linear for ori."""
    def __init__(self, **kwargs):
        kwargs.pop('acc_encoder', None)
        kwargs.pop('ori_encoder', None)
        super().__init__(acc_encoder='conv1d', ori_encoder='linear', **kwargs)


class KalmanLinearConv1d(KalmanEncoderAblation):
    """Linear for acc, Conv1D for ori."""
    def __init__(self, **kwargs):
        kwargs.pop('acc_encoder', None)
        kwargs.pop('ori_encoder', None)
        super().__init__(acc_encoder='linear', ori_encoder='conv1d', **kwargs)


class KalmanLinearLinear(KalmanEncoderAblation):
    """Linear for both streams."""
    def __init__(self, **kwargs):
        kwargs.pop('acc_encoder', None)
        kwargs.pop('ori_encoder', None)
        super().__init__(acc_encoder='linear', ori_encoder='linear', **kwargs)


class KalmanMultiKernelLinear(KalmanEncoderAblation):
    """Multi-kernel Conv1D for acc, linear for ori."""
    def __init__(self, **kwargs):
        kwargs.pop('acc_encoder', None)
        kwargs.pop('ori_encoder', None)
        super().__init__(acc_encoder='multikernel', ori_encoder='linear', **kwargs)
