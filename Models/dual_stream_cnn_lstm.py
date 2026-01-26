"""Dual-stream CNN/LSTM models."""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional


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


class CNNBlock(nn.Module):
    """CNN block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (8, 5, 3),
        dropout: float = 0.2
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_channels

        for i, k in enumerate(kernel_sizes):
            is_last = (i == len(kernel_sizes) - 1)
            out_ch = out_channels if is_last else out_channels // 2

            self.layers.append(nn.Sequential(
                nn.Conv1d(ch, out_ch, kernel_size=k, padding='same'),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if not is_last else nn.Identity()
            ))
            ch = out_ch

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        for layer in self.layers:
            x = layer(x)
        return x + residual


class DualStreamCNN(nn.Module):
    """Dual-stream CNN for IMU data."""

    def __init__(
        self,
        imu_frames: int = 128,
        imu_channels: int = 7,
        acc_frames: int = 128,
        acc_coords: int = 7,
        num_classes: int = 1,
        embed_dim: int = 64,
        dropout: float = 0.5,
        acc_ratio: float = 0.65,
        se_reduction: int = 4,
        num_cnn_layers: int = 3,
        **kwargs
    ):
        super().__init__()
        self.imu_frames = imu_frames or acc_frames
        self.imu_channels = imu_channels or acc_coords

        if self.imu_channels == 7:
            self.acc_channels = 4
            self.ori_channels = 3
        else:
            self.acc_channels = 3
            self.ori_channels = 3

        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim
        self.acc_dim = acc_dim
        self.ori_dim = ori_dim

        self.acc_cnn = nn.Sequential(
            CNNBlock(self.acc_channels, acc_dim // 2, kernel_sizes=(8, 5), dropout=dropout * 0.3),
            CNNBlock(acc_dim // 2, acc_dim, kernel_sizes=(5, 3), dropout=dropout * 0.3),
        )

        self.ori_cnn = nn.Sequential(
            CNNBlock(self.ori_channels, ori_dim // 2, kernel_sizes=(8, 5), dropout=dropout * 0.3),
            CNNBlock(ori_dim // 2, ori_dim, kernel_sizes=(5, 3), dropout=dropout * 0.3),
        )

        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        self.dropout = nn.Dropout(dropout)
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

        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_cnn(acc)
        ori_feat = self.ori_cnn(ori)

        acc_feat = rearrange(acc_feat, 'b c t -> b t c')
        ori_feat = rearrange(ori_feat, 'b c t -> b t c')

        x = torch.cat([acc_feat, ori_feat], dim=-1)
        x = self.fusion_norm(x)
        x = self.se(x)
        features = x

        x, _ = self.temporal_pool(x)
        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output


class DualStreamLSTM(nn.Module):
    """Dual-stream LSTM for IMU data."""

    def __init__(
        self,
        imu_frames: int = 128,
        imu_channels: int = 7,
        acc_frames: int = 128,
        acc_coords: int = 7,
        num_classes: int = 1,
        embed_dim: int = 64,
        dropout: float = 0.5,
        acc_ratio: float = 0.65,
        se_reduction: int = 4,
        num_lstm_layers: int = 2,
        **kwargs
    ):
        super().__init__()
        self.imu_frames = imu_frames or acc_frames
        self.imu_channels = imu_channels or acc_coords

        if self.imu_channels == 7:
            self.acc_channels = 4
            self.ori_channels = 3
        else:
            self.acc_channels = 3
            self.ori_channels = 3

        total_hidden = embed_dim // 2
        acc_hidden = int(total_hidden * acc_ratio)
        ori_hidden = total_hidden - acc_hidden

        self.acc_dim = acc_hidden * 2
        self.ori_dim = ori_hidden * 2

        self.acc_lstm = LSTMEncoder(
            in_channels=self.acc_channels,
            hidden_dim=acc_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout * 0.5,
            bidirectional=True
        )

        self.ori_lstm = LSTMEncoder(
            in_channels=self.ori_channels,
            hidden_dim=ori_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout * 0.5,
            bidirectional=True
        )

        fused_dim = self.acc_dim + self.ori_dim
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.se = SqueezeExcitation(fused_dim, reduction=se_reduction)
        self.temporal_pool = TemporalAttentionPooling(fused_dim)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(fused_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                if 'bias_ih' in name or 'bias_hh' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc_feat = self.acc_lstm(acc)
        ori_feat = self.ori_lstm(ori)

        x = torch.cat([acc_feat, ori_feat], dim=-1)
        x = self.fusion_norm(x)
        x = self.se(x)
        features = x

        x, _ = self.temporal_pool(x)
        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


class DualStreamCNNKalman(DualStreamCNN):
    """CNN with 7-channel input."""
    def __init__(self, **kwargs):
        kwargs['imu_channels'] = 7
        super().__init__(**kwargs)


class DualStreamCNNRaw(DualStreamCNN):
    """CNN with 6-channel input."""
    def __init__(self, **kwargs):
        kwargs['imu_channels'] = 6
        super().__init__(**kwargs)


class DualStreamLSTMKalman(DualStreamLSTM):
    """LSTM with 7-channel input."""
    def __init__(self, **kwargs):
        kwargs['imu_channels'] = 7
        super().__init__(**kwargs)


class DualStreamLSTMRaw(DualStreamLSTM):
    """LSTM with 6-channel input."""
    def __init__(self, **kwargs):
        kwargs['imu_channels'] = 6
        super().__init__(**kwargs)
