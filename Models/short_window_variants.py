"""Short-window transformer variants."""

import math
from typing import List, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=1)
        scale = self.fc(scale).unsqueeze(1)
        return x * scale


class TemporalAttentionPooling(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.einsum('bt,btc->bc', weights, x)
        return context, weights


class TransformerEncoderWithNorm(nn.TransformerEncoder):

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DeepCNNEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [8, 5, 3]
        num_stages = len(kernel_sizes)
        ch_per_stage = out_channels // num_stages
        remainder = out_channels % num_stages

        self.stages = nn.ModuleList()
        current_ch = in_channels
        for i, k in enumerate(kernel_sizes):
            out_ch = ch_per_stage + (remainder if i == num_stages - 1 else 0)
            self.stages.append(
                nn.Sequential(
                    nn.Conv1d(current_ch, out_ch, kernel_size=k, padding='same'),
                    nn.BatchNorm1d(out_ch),
                    nn.SiLU(),
                    nn.Dropout(dropout) if i < num_stages - 1 else nn.Identity(),
                )
            )
            current_ch = out_ch

        self.proj = nn.Conv1d(current_ch, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b t c -> b c t')
        for stage in self.stages:
            x = stage(x)
        x = self.proj(x)
        x = self.norm(x)
        x = rearrange(x, 'b c t -> b t c')
        return x


class DeepCNNTransformer(nn.Module):

    def __init__(
        self,
        imu_frames: int = 128,
        imu_channels: int = 7,
        acc_frames: int = 128,
        acc_coords: int = 7,
        mocap_frames: int = 128,
        num_joints: int = 32,
        num_classes: int = 1,
        num_heads: int = 4,
        num_layers: int = 2,
        embed_dim: int = 48,
        dropout: float = 0.5,
        activation: str = 'relu',
        norm_first: bool = True,
        se_reduction: int = 4,
        acc_ratio: float = 0.65,
        cnn_stages: int = 3,
        kernel_sizes: List[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.imu_frames = imu_frames or acc_frames
        self.imu_channels = imu_channels or acc_coords

        self.acc_channels = 4
        self.ori_channels = self.imu_channels - 4

        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim
        self.acc_dim = acc_dim
        self.ori_dim = ori_dim

        self.acc_encoder = DeepCNNEncoder(
            in_channels=self.acc_channels,
            out_channels=acc_dim,
            kernel_sizes=kernel_sizes,
            dropout=dropout * 0.3,
        )

        self.ori_encoder = nn.Sequential(
            nn.Linear(self.ori_channels, ori_dim),
            nn.LayerNorm(ori_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3),
        )

        self.fusion_norm = nn.LayerNorm(embed_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False,
        )

        self.encoder = TransformerEncoderWithNorm(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc_feat = self.acc_encoder(acc)
        ori_feat = self.ori_encoder(ori)

        x = torch.cat([acc_feat, ori_feat], dim=-1)
        x = self.fusion_norm(x)

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        x = self.se(x)
        features = x
        x, _ = self.temporal_pool(x)

        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


class DeepCNNTransformerKalman(DeepCNNTransformer):

    def __init__(self, **kwargs):
        kwargs['imu_channels'] = 7
        super().__init__(**kwargs)


class DeepCNNTransformerRaw(DeepCNNTransformer):

    def __init__(self, **kwargs):
        kwargs['imu_channels'] = 6
        super().__init__(**kwargs)
