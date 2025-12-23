"""
DualStreamEnhanced: Enhanced Dual-Stream Transformer with Advanced SE/TAP variants.

Novel improvements over DualStreamSE (90.89% baseline):
1. Pre-fusion Stream SE - SE on each stream before concatenation
2. CFEB (Cross-Feature Enhancement Block) - Cross-attention between acc/gyro
3. Multi-Head TAP - Multiple attention heads for temporal pooling
4. Layerwise SE - SE after each transformer layer
5. Temporal SE - SE on temporal dimension

Architecture:
    ACC [4ch] --> Conv1d --> StreamSE --> acc_feat --+
                                                      +--> CFEB --> Concat --> Transformer+SE --> MultiTAP --> Classifier
    ORI [3ch] --> Conv1d --> StreamSE --> ori_feat --+
"""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


class SqueezeExcitation(nn.Module):
    """Channel attention module with optional residual."""

    def __init__(self, channels: int, reduction: int = 4, use_residual: bool = True):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention. x: (B, T, C)"""
        scale = x.mean(dim=1)  # (B, C)
        scale = self.fc(scale).unsqueeze(1)  # (B, 1, C)
        out = x * scale
        if self.use_residual:
            return out + x * 0.1  # Small residual for gradient flow
        return out


class TemporalSE(nn.Module):
    """Temporal attention - SE on time dimension instead of channels."""

    def __init__(self, seq_len: int, reduction: int = 8):
        super().__init__()
        reduced = max(seq_len // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(seq_len, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, seq_len, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention. x: (B, T, C)"""
        # Global channel pooling
        scale = x.mean(dim=2)  # (B, T)
        scale = self.fc(scale).unsqueeze(2)  # (B, T, 1)
        return x * scale


class CrossFeatureEnhancement(nn.Module):
    """Cross-attention between two streams (CFEB).

    Allows acc stream to attend to gyro features and vice versa.
    """

    def __init__(self, acc_dim: int, gyro_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()

        # Cross-attention: acc queries, gyro keys/values
        self.acc_to_gyro = nn.MultiheadAttention(
            embed_dim=acc_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=gyro_dim,
            vdim=gyro_dim,
            batch_first=True
        )

        # Cross-attention: gyro queries, acc keys/values
        self.gyro_to_acc = nn.MultiheadAttention(
            embed_dim=gyro_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=acc_dim,
            vdim=acc_dim,
            batch_first=True
        )

        # Normalization
        self.acc_norm = nn.LayerNorm(acc_dim)
        self.gyro_norm = nn.LayerNorm(gyro_dim)

        # Gating for residual
        self.acc_gate = nn.Sequential(
            nn.Linear(acc_dim, acc_dim),
            nn.Sigmoid()
        )
        self.gyro_gate = nn.Sequential(
            nn.Linear(gyro_dim, gyro_dim),
            nn.Sigmoid()
        )

    def forward(self, acc_feat: torch.Tensor, gyro_feat: torch.Tensor) -> tuple:
        """
        Args:
            acc_feat: (B, T, acc_dim)
            gyro_feat: (B, T, gyro_dim)
        Returns:
            enhanced_acc: (B, T, acc_dim)
            enhanced_gyro: (B, T, gyro_dim)
        """
        # Acc attends to gyro
        acc_cross, _ = self.acc_to_gyro(acc_feat, gyro_feat, gyro_feat)
        acc_gate = self.acc_gate(acc_feat.mean(dim=1, keepdim=True))
        enhanced_acc = self.acc_norm(acc_feat + acc_gate * acc_cross)

        # Gyro attends to acc
        gyro_cross, _ = self.gyro_to_acc(gyro_feat, acc_feat, acc_feat)
        gyro_gate = self.gyro_gate(gyro_feat.mean(dim=1, keepdim=True))
        enhanced_gyro = self.gyro_norm(gyro_feat + gyro_gate * gyro_cross)

        return enhanced_acc, enhanced_gyro


class MultiHeadTAP(nn.Module):
    """Multi-head Temporal Attention Pooling.

    Multiple attention heads capture different temporal patterns:
    - One head may focus on sudden changes (falls)
    - Another may focus on sustained patterns (ADL)
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Per-head attention projections
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, self.head_dim),
                nn.Tanh(),
                nn.Linear(self.head_dim, 1, bias=False)
            ) for _ in range(num_heads)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * num_heads, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C)
        Returns:
            context: (B, C) pooled representation
            weights: (B, num_heads, T) attention weights per head
        """
        B, T, C = x.shape
        contexts = []
        all_weights = []

        for head in self.attention_heads:
            scores = head(x).squeeze(-1)  # (B, T)
            weights = F.softmax(scores, dim=1)  # (B, T)
            all_weights.append(weights)
            context = torch.einsum('bt,btc->bc', weights, x)  # (B, C)
            contexts.append(context)

        # Concatenate and project
        multi_context = torch.cat(contexts, dim=1)  # (B, C*num_heads)
        output = self.output_proj(multi_context)  # (B, C)

        weights_tensor = torch.stack(all_weights, dim=1)  # (B, num_heads, T)
        return output, weights_tensor


class TemporalAttentionPooling(nn.Module):
    """Standard learnable temporal pooling (for ablation comparison)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.einsum('bt,btc->bc', weights, x)
        return context, weights


class TransformerLayerWithSE(nn.Module):
    """Transformer encoder layer with integrated SE module."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, activation: str, norm_first: bool,
                 se_reduction: int = 4):
        super().__init__()

        self.transformer_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )
        self.se = SqueezeExcitation(d_model, reduction=se_reduction, use_residual=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer_layer(x)
        x = self.se(x)
        return x


class DualStreamEnhanced(nn.Module):
    """Enhanced dual-stream transformer with advanced attention mechanisms.

    Configurable enhancements:
    - use_stream_se: SE on each stream before fusion
    - use_cfeb: Cross-feature enhancement between streams
    - use_temporal_se: Temporal SE after transformer
    - use_layerwise_se: SE after each transformer layer
    - use_multihead_tap: Multi-head temporal attention pooling
    """

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
                 activation: str = 'gelu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.5,
                 # Enhancement flags
                 use_stream_se: bool = True,
                 use_cfeb: bool = True,
                 use_temporal_se: bool = False,
                 use_layerwise_se: bool = False,
                 use_multihead_tap: bool = True,
                 tap_heads: int = 4,
                 **kwargs):
        super().__init__()

        # Store config
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_stream_se = use_stream_se
        self.use_cfeb = use_cfeb
        self.use_temporal_se = use_temporal_se
        self.use_layerwise_se = use_layerwise_se
        self.use_multihead_tap = use_multihead_tap

        # Channel split for Kalman 7ch: [smv, ax, ay, az, roll, pitch, yaw]
        # acc: [smv, ax, ay, az] = 4 channels
        # ori: [roll, pitch, yaw] = 3 channels
        if self.imu_channels == 7:
            self.acc_in_channels = 4
            self.ori_in_channels = 3
        elif self.imu_channels == 8:
            self.acc_in_channels = 4
            self.ori_in_channels = 4
        elif self.imu_channels == 6:
            self.acc_in_channels = 3
            self.ori_in_channels = 3
        else:
            self.acc_in_channels = self.imu_channels // 2
            self.ori_in_channels = self.imu_channels - self.acc_in_channels

        # Asymmetric embedding allocation
        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim
        self.acc_dim = acc_dim
        self.ori_dim = ori_dim

        # Stream projections with deeper CNN
        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_in_channels, acc_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(acc_dim),
            nn.GELU(),
            nn.Conv1d(acc_dim, acc_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(acc_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.2)
        )

        self.ori_proj = nn.Sequential(
            nn.Conv1d(self.ori_in_channels, ori_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(ori_dim),
            nn.GELU(),
            nn.Conv1d(ori_dim, ori_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(ori_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.4)
        )

        # Pre-fusion Stream SE (Enhancement 1)
        if use_stream_se:
            self.acc_se = SqueezeExcitation(acc_dim, reduction=se_reduction)
            self.ori_se = SqueezeExcitation(ori_dim, reduction=se_reduction)

        # Cross-Feature Enhancement Block (Enhancement 2)
        if use_cfeb:
            self.cfeb = CrossFeatureEnhancement(acc_dim, ori_dim, num_heads=2, dropout=dropout)

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder (with or without layerwise SE)
        if use_layerwise_se:
            self.encoder_layers = nn.ModuleList([
                TransformerLayerWithSE(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 2,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    se_reduction=se_reduction
                ) for _ in range(num_layers)
            ])
            self.encoder_norm = nn.LayerNorm(embed_dim)
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(embed_dim)
            )

        # Post-transformer SE (standard)
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)

        # Temporal SE (Enhancement 5)
        if use_temporal_se:
            self.temporal_se = TemporalSE(self.imu_frames, reduction=8)

        # Temporal pooling
        if use_multihead_tap:
            self.temporal_pool = MultiHeadTAP(embed_dim, num_heads=tap_heads, dropout=dropout)
        else:
            self.temporal_pool = TemporalAttentionPooling(embed_dim)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        """
        Args:
            acc_data: (B, T, C) IMU data
        Returns:
            logits: (B, num_classes)
            features: (B, T, embed_dim)
        """
        # Split modalities
        acc = acc_data[:, :, :self.acc_in_channels]
        ori = acc_data[:, :, self.acc_in_channels:self.acc_in_channels + self.ori_in_channels]

        # Conv projections: (B, T, C) -> (B, C, T) -> (B, dim, T)
        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        # Rearrange to (B, T, dim) for attention
        acc_feat = rearrange(acc_feat, 'b c t -> b t c')
        ori_feat = rearrange(ori_feat, 'b c t -> b t c')

        # Enhancement 1: Pre-fusion Stream SE
        if self.use_stream_se:
            acc_feat = self.acc_se(acc_feat)
            ori_feat = self.ori_se(ori_feat)

        # Enhancement 2: Cross-Feature Enhancement
        if self.use_cfeb:
            acc_feat, ori_feat = self.cfeb(acc_feat, ori_feat)

        # Fusion
        x = torch.cat([acc_feat, ori_feat], dim=2)  # (B, T, embed_dim)
        x = self.fusion_norm(x)

        # Transformer encoding
        if self.use_layerwise_se:
            for layer in self.encoder_layers:
                x = layer(x)
            x = self.encoder_norm(x)
        else:
            x = self.encoder(x)

        # Post-transformer SE
        x = self.se(x)

        # Enhancement 5: Temporal SE
        if self.use_temporal_se:
            x = self.temporal_se(x)

        features = x

        # Temporal pooling
        x, attn_weights = self.temporal_pool(x)

        # Classification
        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


if __name__ == "__main__":
    print("=" * 60)
    print("DualStreamEnhanced Model Test")
    print("=" * 60)

    # Test configurations
    configs = [
        {"name": "Full Enhanced", "use_stream_se": True, "use_cfeb": True, "use_multihead_tap": True},
        {"name": "No CFEB", "use_stream_se": True, "use_cfeb": False, "use_multihead_tap": True},
        {"name": "No Stream SE", "use_stream_se": False, "use_cfeb": True, "use_multihead_tap": True},
        {"name": "Single TAP", "use_stream_se": True, "use_cfeb": True, "use_multihead_tap": False},
        {"name": "Baseline (all off)", "use_stream_se": False, "use_cfeb": False, "use_multihead_tap": False},
    ]

    for cfg in configs:
        name = cfg.pop("name")
        model = DualStreamEnhanced(imu_frames=128, imu_channels=7, embed_dim=64, **cfg)
        x = torch.randn(16, 128, 7)
        logits, features = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} params, output={logits.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
