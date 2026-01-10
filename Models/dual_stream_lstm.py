import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Tuple


class SqueezeExcitation(nn.Module):
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
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.einsum('bt,btc->bc', weights, x)
        return context, weights


class DualStreamLSTM(nn.Module):
    def __init__(self,
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
                 acc_ratio: float = 0.5,
                 use_se: bool = True,
                 use_tap: bool = True,
                 use_pos_encoding: bool = False,
                 acc_channels: int = 4,
                 bidirectional: bool = True,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_se = use_se
        self.use_tap = use_tap
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.acc_channels = acc_channels
        self.ori_channels = self.imu_channels - acc_channels

        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim

        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        self.ori_proj = nn.Sequential(
            nn.Conv1d(self.ori_channels, ori_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(ori_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3)
        )

        self.fusion_norm = nn.LayerNorm(embed_dim)

        lstm_hidden = embed_dim // 2 if bidirectional else embed_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.encoder_norm = nn.LayerNorm(embed_dim)

        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = None

        if use_tap:
            self.temporal_pool = TemporalAttentionPooling(embed_dim)
        else:
            self.temporal_pool = None

        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        x, _ = self.lstm(x)
        x = self.encoder_norm(x)

        if self.se is not None:
            x = self.se(x)

        features = x

        if self.temporal_pool is not None:
            x, attn_weights = self.temporal_pool(x)
        else:
            x = x.mean(dim=1)

        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


class DualStreamLSTMFlexible(DualStreamLSTM):
    pass


class DualStreamGRU(nn.Module):
    def __init__(self,
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
                 acc_ratio: float = 0.5,
                 use_se: bool = True,
                 use_tap: bool = True,
                 use_pos_encoding: bool = False,
                 acc_channels: int = 4,
                 bidirectional: bool = True,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_se = use_se
        self.use_tap = use_tap
        self.bidirectional = bidirectional

        self.acc_channels = acc_channels
        self.ori_channels = self.imu_channels - acc_channels

        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim

        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        self.ori_proj = nn.Sequential(
            nn.Conv1d(self.ori_channels, ori_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(ori_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3)
        )

        self.fusion_norm = nn.LayerNorm(embed_dim)

        gru_hidden = embed_dim // 2 if bidirectional else embed_dim
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.encoder_norm = nn.LayerNorm(embed_dim)

        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = None

        if use_tap:
            self.temporal_pool = TemporalAttentionPooling(embed_dim)
        else:
            self.temporal_pool = None

        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        x, _ = self.gru(x)
        x = self.encoder_norm(x)

        if self.se is not None:
            x = self.se(x)

        features = x

        if self.temporal_pool is not None:
            x, _ = self.temporal_pool(x)
        else:
            x = x.mean(dim=1)

        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


class SingleStreamLSTM(nn.Module):
    def __init__(self,
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
                 acc_ratio: float = 0.5,
                 use_se: bool = True,
                 use_tap: bool = True,
                 use_pos_encoding: bool = False,
                 acc_channels: int = 4,
                 bidirectional: bool = True,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_se = use_se
        self.use_tap = use_tap
        self.bidirectional = bidirectional

        self.proj = nn.Sequential(
            nn.Conv1d(self.imu_channels, embed_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        self.input_norm = nn.LayerNorm(embed_dim)

        lstm_hidden = embed_dim // 2 if bidirectional else embed_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.encoder_norm = nn.LayerNorm(embed_dim)

        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = None

        if use_tap:
            self.temporal_pool = TemporalAttentionPooling(embed_dim)
        else:
            self.temporal_pool = None

        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = rearrange(acc_data, 'b t c -> b c t')
        x = self.proj(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.input_norm(x)

        x, _ = self.lstm(x)
        x = self.encoder_norm(x)

        if self.se is not None:
            x = self.se(x)

        features = x

        if self.temporal_pool is not None:
            x, _ = self.temporal_pool(x)
        else:
            x = x.mean(dim=1)

        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


LSTM_MODELS = {
    'DualStreamLSTM': DualStreamLSTM,
    'DualStreamLSTMFlexible': DualStreamLSTMFlexible,
    'DualStreamGRU': DualStreamGRU,
    'SingleStreamLSTM': SingleStreamLSTM,
}


def get_model(name: str, **kwargs) -> nn.Module:
    if name not in LSTM_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(LSTM_MODELS.keys())}")
    return LSTM_MODELS[name](**kwargs)


if __name__ == '__main__':
    batch_size, seq_len, channels = 4, 128, 7
    x = torch.randn(batch_size, seq_len, channels)

    print("Testing DualStreamLSTM...")
    model = DualStreamLSTM(imu_frames=128, imu_channels=7)
    logits, features = model(x)
    print(f"  Input: {x.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting SingleStreamLSTM...")
    model2 = SingleStreamLSTM(imu_frames=128, imu_channels=7)
    logits2, features2 = model2(x)
    print(f"  Logits: {logits2.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model2.parameters()):,}")

    print("\nTesting DualStreamGRU...")
    model3 = DualStreamGRU(imu_frames=128, imu_channels=7)
    logits3, features3 = model3(x)
    print(f"  Logits: {logits3.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model3.parameters()):,}")

    print("\nAll models working correctly!")
