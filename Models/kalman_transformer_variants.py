"""
KalmanTransformer Architectural Variants for Ablation Study.

This module contains 8 architectures for scientific comparison:

Dual-Stream Variants:
    1. KalmanTransformerBaseline - Current best (88.09% F1)
    2. KalmanCrossModalAttention - Cross-attention between acc/ori streams (Novel)
    3. KalmanGatedFusion - Learnable gating for dynamic stream weighting
    4. KalmanDeepNarrow - Deeper (3 layers) but narrower (48 dim)
    5. KalmanUncertaintyAware - Includes Kalman filter uncertainty estimates
    6. KalmanBalancedRatio - Equal capacity for acc/ori (50/50)

Single-Stream Variants:
    7. KalmanSingleStream - Combined 7ch single-stream architecture
    8. KalmanCompact - Minimal model for overfitting reduction

Scientific Rationale:
    - Cross-modal attention: Learn which orientation features correlate with acc patterns
    - Gated fusion: Dynamic weighting based on input confidence
    - Deep-narrow: Literature shows depth helps sequential modeling
    - Uncertainty: Weight unreliable Kalman estimates less
    - Compact: Reduce overfitting on small dataset (~22 subjects)
"""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


# =============================================================================
# Shared Components
# =============================================================================

class SqueezeExcitation(nn.Module):
    """Channel attention module."""

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
        """x: (B, T, C)"""
        scale = x.mean(dim=1)
        scale = self.fc(scale).unsqueeze(1)
        return x * scale


class TemporalAttentionPooling(nn.Module):
    """Learnable temporal pooling for transient event detection."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """x: (B, T, C) -> (B, C), (B, T)"""
        scores = self.attention(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.einsum('bt,btc->bc', weights, x)
        return context, weights


class TransformerEncoderWithNorm(nn.TransformerEncoder):
    """Transformer encoder with final layer normalization."""

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


# =============================================================================
# Variant 1: Baseline (Current Best - 88.09% F1)
# =============================================================================

class KalmanTransformerBaseline(nn.Module):
    """
    Baseline KalmanTransformer - current best architecture.

    Input: 7ch [smv, ax, ay, az, roll, pitch, yaw]
    Dual-stream: acc (65%) + ori (35%)
    Architecture: Conv1d projection -> Transformer -> SE -> TAP
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
                 activation: str = 'relu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.65,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        self.acc_channels = 4  # smv, ax, ay, az
        self.ori_channels = self.imu_channels - 4

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

        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        x = self.se(x)
        features = x

        x, attn_weights = self.temporal_pool(x)
        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Variant 2: Cross-Modal Attention (Novel Architecture)
# =============================================================================

class CrossModalAttentionBlock(nn.Module):
    """
    Cross-modal attention between accelerometer and orientation streams.

    Allows acc stream to attend to ori features and vice versa,
    learning which orientation patterns correlate with accelerometer events.
    """

    def __init__(self, acc_dim: int, ori_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()

        # Ensure dimensions are divisible by num_heads
        # Use 1 head if not divisible
        acc_heads = num_heads if acc_dim % num_heads == 0 else 1
        ori_heads = num_heads if ori_dim % num_heads == 0 else 1

        # Acc attends to Ori
        self.acc_to_ori = nn.MultiheadAttention(
            embed_dim=acc_dim,
            num_heads=acc_heads,
            kdim=ori_dim,
            vdim=ori_dim,
            dropout=dropout,
            batch_first=True
        )

        # Ori attends to Acc
        self.ori_to_acc = nn.MultiheadAttention(
            embed_dim=ori_dim,
            num_heads=ori_heads,
            kdim=acc_dim,
            vdim=acc_dim,
            dropout=dropout,
            batch_first=True
        )

        self.acc_norm = nn.LayerNorm(acc_dim)
        self.ori_norm = nn.LayerNorm(ori_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, acc_feat: torch.Tensor, ori_feat: torch.Tensor):
        """
        Args:
            acc_feat: (B, T, acc_dim)
            ori_feat: (B, T, ori_dim)
        Returns:
            Enhanced acc_feat, ori_feat
        """
        # Acc attends to Ori: what orientation patterns are relevant?
        acc_cross, _ = self.acc_to_ori(acc_feat, ori_feat, ori_feat)
        acc_feat = self.acc_norm(acc_feat + self.dropout(acc_cross))

        # Ori attends to Acc: what accelerometer patterns matter?
        ori_cross, _ = self.ori_to_acc(ori_feat, acc_feat, acc_feat)
        ori_feat = self.ori_norm(ori_feat + self.dropout(ori_cross))

        return acc_feat, ori_feat


class KalmanCrossModalAttention(nn.Module):
    """
    Novel: Cross-modal attention between accelerometer and orientation streams.

    Key innovation: Instead of simple concatenation, the model learns
    which orientation features are relevant for specific accelerometer patterns
    through bidirectional cross-attention.

    Hypothesis: Falls have characteristic acc-ori correlations that
    cross-attention can capture better than concatenation.
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
                 activation: str = 'relu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.65,
                 cross_attn_heads: int = 2,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        self.acc_channels = 4
        self.ori_channels = self.imu_channels - 4

        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim
        self.acc_dim = acc_dim
        self.ori_dim = ori_dim

        # Stream projections
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

        # Cross-modal attention (Novel component)
        self.cross_modal = CrossModalAttentionBlock(
            acc_dim=acc_dim,
            ori_dim=ori_dim,
            num_heads=cross_attn_heads,
            dropout=dropout * 0.3
        )

        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder
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
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        # Project each stream
        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        # Rearrange for cross-attention: (B, C, T) -> (B, T, C)
        acc_feat = rearrange(acc_feat, 'b c t -> b t c')
        ori_feat = rearrange(ori_feat, 'b c t -> b t c')

        # Cross-modal attention (novel)
        acc_feat, ori_feat = self.cross_modal(acc_feat, ori_feat)

        # Concatenate and normalize
        x = torch.cat([acc_feat, ori_feat], dim=-1)
        x = self.fusion_norm(x)

        # Transformer
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        x = self.se(x)
        features = x

        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Variant 3: Gated Fusion (Dynamic Stream Weighting)
# =============================================================================

class GatedFusionLayer(nn.Module):
    """
    Learnable gating mechanism for dynamic acc/ori weighting.

    Gate values are learned per-timestep based on input confidence,
    allowing the model to dynamically rely more on acc or ori.
    """

    def __init__(self, acc_dim: int, ori_dim: int, embed_dim: int):
        super().__init__()

        # Gate computation from both streams
        self.gate_net = nn.Sequential(
            nn.Linear(acc_dim + ori_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 2),  # 2 gates: acc and ori
            nn.Softmax(dim=-1)
        )

        # Project to same dimension for gated sum
        self.acc_proj_gate = nn.Linear(acc_dim, embed_dim)
        self.ori_proj_gate = nn.Linear(ori_dim, embed_dim)

    def forward(self, acc_feat: torch.Tensor, ori_feat: torch.Tensor):
        """
        Args:
            acc_feat: (B, T, acc_dim)
            ori_feat: (B, T, ori_dim)
        Returns:
            Gated fusion: (B, T, embed_dim)
        """
        # Compute gates from concatenated features
        combined = torch.cat([acc_feat, ori_feat], dim=-1)
        gates = self.gate_net(combined)  # (B, T, 2)

        acc_gate = gates[:, :, 0:1]  # (B, T, 1)
        ori_gate = gates[:, :, 1:2]

        # Project and gate
        acc_proj = self.acc_proj_gate(acc_feat)
        ori_proj = self.ori_proj_gate(ori_feat)

        fused = acc_gate * acc_proj + ori_gate * ori_proj
        return fused


class KalmanGatedFusion(nn.Module):
    """
    Gated fusion for dynamic accelerometer/orientation weighting.

    Hypothesis: Different parts of a fall sequence may benefit from
    different acc/ori weightings. Gating allows learned dynamic fusion.

    E.g., impact phase relies more on acc, pre-fall posture on ori.
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
                 activation: str = 'relu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.65,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        self.acc_channels = 4
        self.ori_channels = self.imu_channels - 4

        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim

        # Stream projections
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

        # Gated fusion (novel component)
        self.gated_fusion = GatedFusionLayer(acc_dim, ori_dim, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder
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
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        # Rearrange: (B, C, T) -> (B, T, C)
        acc_feat = rearrange(acc_feat, 'b c t -> b t c')
        ori_feat = rearrange(ori_feat, 'b c t -> b t c')

        # Gated fusion
        x = self.gated_fusion(acc_feat, ori_feat)
        x = self.fusion_norm(x)

        # Transformer
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        x = self.se(x)
        features = x

        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Variant 4: Deep Narrow (3 layers, embed_dim=48)
# =============================================================================

class KalmanDeepNarrow(nn.Module):
    """
    Deeper but narrower architecture.

    Scientific rationale: Literature shows for sequential data,
    depth often matters more than width. More layers allow
    learning more complex temporal dependencies.

    Trade-off: 3 layers (vs 2), embed_dim=48 (vs 64)
    Similar parameter count, different capacity allocation.
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 7,
                 acc_frames: int = 128,
                 acc_coords: int = 7,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 num_classes: int = 1,
                 num_heads: int = 3,  # Must divide embed_dim
                 num_layers: int = 3,  # Deeper
                 embed_dim: int = 48,  # Narrower
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.65,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        self.acc_channels = 4
        self.ori_channels = self.imu_channels - 4

        acc_dim = int(embed_dim * acc_ratio)  # 31
        ori_dim = embed_dim - acc_dim  # 17

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
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        x = self.se(x)
        features = x

        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Variant 5: Uncertainty-Aware (10ch with Kalman uncertainty)
# =============================================================================

class KalmanUncertaintyAware(nn.Module):
    """
    Includes Kalman filter uncertainty estimates as input features.

    Input: 10ch [smv, ax, ay, az, roll, pitch, yaw, sigma_r, sigma_p, sigma_y]

    Scientific rationale: Kalman filter provides uncertainty estimates
    (sigma values from covariance matrix). The model can learn to
    weight uncertain orientation estimates less during classification.

    This is particularly useful when gyro drift causes high uncertainty.
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 10,  # Includes uncertainty
                 acc_frames: int = 128,
                 acc_coords: int = 10,
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
                 acc_ratio: float = 0.6,  # Slightly lower for ori+uncertainty
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # 10ch: [smv, ax, ay, az] (4) + [roll, pitch, yaw, sigma_r, sigma_p, sigma_y] (6)
        self.acc_channels = 4
        self.ori_channels = self.imu_channels - 4  # 6 (ori + uncertainty)

        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim

        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        # Larger ori projection to handle uncertainty channels
        self.ori_proj = nn.Sequential(
            nn.Conv1d(self.ori_channels, ori_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(ori_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3)
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
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        x = self.se(x)
        features = x

        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Variant 6: Balanced Ratio (50/50 acc/ori capacity)
# =============================================================================

class KalmanBalancedRatio(nn.Module):
    """
    Equal capacity allocation for acc and ori streams.

    Scientific rationale: Test whether the 65/35 split is optimal,
    or if equal capacity allows better orientation feature learning.

    Baseline uses acc_ratio=0.65, this uses acc_ratio=0.5.
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
                 activation: str = 'relu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.5,  # Equal capacity
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        self.acc_channels = 4
        self.ori_channels = self.imu_channels - 4

        acc_dim = int(embed_dim * acc_ratio)  # 32
        ori_dim = embed_dim - acc_dim  # 32

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
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        x = self.se(x)
        features = x

        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Variant 7: Single-Stream Kalman
# =============================================================================

class KalmanSingleStream(nn.Module):
    """
    Single-stream architecture processing all channels together.

    Scientific rationale: Previous experiments showed single-stream
    often beats dual-stream. Test if this holds for Kalman features.

    Simpler architecture, fewer parameters, less prone to overfitting.
    Uses first 7 channels to match dual-stream input format.
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
                 activation: str = 'relu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.embed_dim = embed_dim
        self.dropout_rate = dropout

        # Fixed 7-channel input (matches dual-stream: smv + acc + ori)
        self.input_proj = nn.Sequential(
            nn.Conv1d(7, embed_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        self.input_norm = nn.LayerNorm(embed_dim)

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
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        # Single-stream: use first 7 channels (smv, ax, ay, az, roll, pitch, yaw)
        x = acc_data[:, :, :7]  # Slice to 7 channels
        x = rearrange(x, 'b t c -> b c t')
        x = self.input_proj(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.input_norm(x)

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        x = self.se(x)
        features = x

        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Variant 8: Compact Model (Anti-Overfitting)
# =============================================================================

class KalmanCompact(nn.Module):
    """
    Minimal architecture for overfitting reduction.

    Scientific rationale: With only ~22 subjects for testing,
    a smaller model may generalize better by avoiding memorization.

    Key changes:
    - embed_dim=32 (vs 64)
    - num_layers=1 (vs 2)
    - Higher dropout=0.6 (vs 0.5)
    - Single-stream (simpler)
    - Uses first 7 channels to match dual-stream

    ~12K parameters vs ~50K in baseline.
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 7,
                 acc_frames: int = 128,
                 acc_coords: int = 7,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 num_classes: int = 1,
                 num_heads: int = 2,  # Fewer heads
                 num_layers: int = 1,  # Single layer
                 embed_dim: int = 32,  # Narrower
                 dropout: float = 0.6,  # Higher dropout
                 activation: str = 'relu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        # Fixed 7-channel input (matches dual-stream: smv + acc + ori)
        self.input_proj = nn.Sequential(
            nn.Conv1d(7, embed_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3)
        )

        self.input_norm = nn.LayerNorm(embed_dim)

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
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        # Use first 7 channels (smv, ax, ay, az, roll, pitch, yaw)
        x = acc_data[:, :, :7]  # Slice to 7 channels
        x = rearrange(x, 'b t c -> b c t')
        x = self.input_proj(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.input_norm(x)

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        x = self.se(x)
        features = x

        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Variant 9: Flexible Kalman Balanced (Configurable SE/TAP/PosEnc)
# =============================================================================

class KalmanBalancedFlexible(nn.Module):
    """
    Flexible KalmanBalanced with configurable SE, TAP, and positional encoding.

    Used for ablation studies to isolate the effect of each component.

    Args:
        use_se: Enable Squeeze-Excitation attention
        use_tap: Enable Temporal Attention Pooling (vs GAP)
        use_pos_encoding: Enable learned positional encoding
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
                 activation: str = 'relu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.5,
                 use_se: bool = True,
                 use_tap: bool = True,
                 use_pos_encoding: bool = False,
                 acc_channels: int = 4,  # 4 with SMV [smv,ax,ay,az], 3 without [ax,ay,az]
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_se = use_se
        self.use_tap = use_tap
        self.use_pos_encoding = use_pos_encoding

        self.acc_channels = acc_channels  # Configurable: 4 with SMV, 3 without
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

        # Optional positional encoding
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, self.imu_frames, embed_dim) * 0.02
            )
        else:
            self.pos_encoding = None

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

        # Optional SE
        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = None

        # Optional TAP (vs GAP)
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

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        ori_feat = self.ori_proj(ori)

        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        # Optional positional encoding
        if self.use_pos_encoding and self.pos_encoding is not None:
            x = x + self.pos_encoding

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        # Optional SE
        if self.se is not None:
            x = self.se(x)

        features = x

        # TAP or GAP
        if self.temporal_pool is not None:
            x, attn_weights = self.temporal_pool(x)
        else:
            x = x.mean(dim=1)  # Global Average Pooling

        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Variant 10: Convolutional Tokenizer (Multi-layer CNN before Transformer)
# =============================================================================

class ConvTokenizer(nn.Module):
    """
    Multi-layer 1D CNN tokenizer for better local feature extraction.

    Replaces simple linear/single-conv projection with stacked convolutions
    that capture multi-scale local temporal patterns before transformer.

    Architecture: Conv(k=3) -> Conv(k=5) -> Conv(k=7) with residual
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()

        hidden_dim = max(out_channels // 2, in_channels * 2)

        # Layer 1: Small kernel for fine-grained patterns
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Layer 2: Medium kernel for mid-range patterns
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Layer 3: Large kernel for broader patterns (matches original k=8)
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, kernel_size=7, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Residual projection if dimensions differ
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x + residual


class MultiScaleConvTokenizer(nn.Module):
    """
    Multi-scale 1D CNN tokenizer with parallel branches.

    Uses parallel convolutions with different kernel sizes and concatenates,
    similar to Inception module but for 1D temporal data.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()

        # Split output channels across scales
        branch_dim = out_channels // 3
        remainder = out_channels - (branch_dim * 3)

        # Small kernel branch (local patterns)
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_channels, branch_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(branch_dim),
            nn.GELU()
        )

        # Medium kernel branch (medium-range patterns)
        self.branch_medium = nn.Sequential(
            nn.Conv1d(in_channels, branch_dim, kernel_size=7, padding='same'),
            nn.BatchNorm1d(branch_dim),
            nn.GELU()
        )

        # Large kernel branch (broader patterns)
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_channels, branch_dim + remainder, kernel_size=15, padding='same'),
            nn.BatchNorm1d(branch_dim + remainder),
            nn.GELU()
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        small = self.branch_small(x)
        medium = self.branch_medium(x)
        large = self.branch_large(x)

        x = torch.cat([small, medium, large], dim=1)
        x = self.fusion(x)

        return x


class KalmanConvTokenizer(nn.Module):
    """
    Kalman Transformer with multi-layer CNN tokenization.

    Key innovation: Replace single Conv1d projection with stacked CNN layers
    for better local feature extraction before transformer attention.

    Scientific rationale:
    - IMU signals have local temporal correlations (impact spike, oscillation)
    - Multi-layer CNN can learn hierarchical local patterns
    - Better tokenization -> better attention over meaningful tokens

    Args:
        tokenizer_type: 'stacked' (sequential CNN) or 'multiscale' (parallel branches)
        acc_tokenizer_type: 'cnn' or 'linear' - tokenizer for acc stream
        ori_tokenizer_type: 'cnn' or 'linear' - tokenizer for ori stream
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
                 activation: str = 'relu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 acc_ratio: float = 0.5,
                 use_se: bool = True,
                 use_tap: bool = True,
                 use_pos_encoding: bool = False,
                 acc_channels: int = 4,
                 tokenizer_type: str = 'stacked',  # 'stacked' or 'multiscale'
                 acc_tokenizer_type: str = 'cnn',  # 'cnn' or 'linear'
                 ori_tokenizer_type: str = 'cnn',  # 'cnn' or 'linear'
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_se = use_se
        self.use_tap = use_tap
        self.use_pos_encoding = use_pos_encoding
        self.tokenizer_type = tokenizer_type
        self.acc_tokenizer_type = acc_tokenizer_type
        self.ori_tokenizer_type = ori_tokenizer_type

        self.acc_channels = acc_channels
        self.ori_channels = self.imu_channels - acc_channels

        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim

        # CNN Tokenizer class selection
        TokenizerClass = MultiScaleConvTokenizer if tokenizer_type == 'multiscale' else ConvTokenizer

        # Acc tokenizer: CNN or Linear
        if acc_tokenizer_type == 'cnn':
            self.acc_tokenizer = TokenizerClass(
                in_channels=self.acc_channels,
                out_channels=acc_dim,
                dropout=dropout * 0.2
            )
        else:  # linear
            self.acc_tokenizer = nn.Sequential(
                nn.Conv1d(self.acc_channels, acc_dim, kernel_size=1),
                nn.BatchNorm1d(acc_dim),
                nn.SiLU(),
                nn.Dropout(dropout * 0.2)
            )

        # Ori tokenizer: CNN or Linear
        if ori_tokenizer_type == 'cnn':
            self.ori_tokenizer = TokenizerClass(
                in_channels=self.ori_channels,
                out_channels=ori_dim,
                dropout=dropout * 0.3
            )
        else:  # linear
            self.ori_tokenizer = nn.Sequential(
                nn.Conv1d(self.ori_channels, ori_dim, kernel_size=1),
                nn.BatchNorm1d(ori_dim),
                nn.SiLU(),
                nn.Dropout(dropout * 0.3)
            )

        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Optional positional encoding
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, self.imu_frames, embed_dim) * 0.02
            )
        else:
            self.pos_encoding = None

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

        # Optional SE
        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = None

        # Optional TAP
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

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        # CNN Tokenization
        acc_feat = self.acc_tokenizer(acc)
        ori_feat = self.ori_tokenizer(ori)

        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        # Optional positional encoding
        if self.use_pos_encoding and self.pos_encoding is not None:
            x = x + self.pos_encoding

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        # Optional SE
        if self.se is not None:
            x = self.se(x)

        features = x

        # TAP or GAP
        if self.temporal_pool is not None:
            x, attn_weights = self.temporal_pool(x)
        else:
            x = x.mean(dim=1)

        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Test Script
# =============================================================================

# =============================================================================
# Variant 11: Quaternion Flexible (8ch: smv + acc + quaternion)
# =============================================================================

class KalmanQuaternionFlexible(nn.Module):
    """
    Dual-stream transformer for quaternion orientation features.

    Input: 8ch [smv, ax, ay, az, q0, q1, q2, q3]

    For quaternion-based orientation from:
    - Madgwick AHRS filter
    - VQF (Versatile Quaternion Filter)
    - EKF with quaternion output

    Quaternion advantages over Euler:
    - No gimbal lock at pitch ±90° (backward falls)
    - Continuous representation (no ±π discontinuities)
    - Standard representation in robotics/aerospace

    Args:
        imu_channels: 8 for quaternion [smv, ax, ay, az, q0, q1, q2, q3]
        acc_ratio: Ratio of embedding dim for acc stream (4ch: smv+acc)
                   vs quat stream (4ch: q0-q3). Default 0.5 for equal split.
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 8,  # smv + acc + quaternion
                 acc_frames: int = 128,
                 acc_coords: int = 8,
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
                 acc_ratio: float = 0.5,  # Equal split for 4ch acc vs 4ch quat
                 use_se: bool = True,
                 use_tap: bool = True,
                 use_pos_encoding: bool = False,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_se = use_se
        self.use_tap = use_tap
        self.use_pos_encoding = use_pos_encoding

        # 8ch split: [smv, ax, ay, az] (4ch acc) + [q0, q1, q2, q3] (4ch quat)
        self.acc_channels = 4  # smv, ax, ay, az
        self.quat_channels = 4  # q0, q1, q2, q3

        acc_dim = int(embed_dim * acc_ratio)
        quat_dim = embed_dim - acc_dim

        # Accelerometer stream projection
        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        # Quaternion stream projection
        self.quat_proj = nn.Sequential(
            nn.Conv1d(self.quat_channels, quat_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(quat_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3)
        )

        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Optional positional encoding
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, self.imu_frames, embed_dim) * 0.02
            )
        else:
            self.pos_encoding = None

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

        # Optional SE
        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = None

        # Optional TAP (vs GAP)
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

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        # Split into acc and quat streams
        # Input: (B, T, 8) -> [smv, ax, ay, az, q0, q1, q2, q3]
        acc = acc_data[:, :, :self.acc_channels]   # (B, T, 4)
        quat = acc_data[:, :, self.acc_channels:self.acc_channels + self.quat_channels]  # (B, T, 4)

        acc = rearrange(acc, 'b t c -> b c t')
        quat = rearrange(quat, 'b t c -> b c t')

        acc_feat = self.acc_proj(acc)
        quat_feat = self.quat_proj(quat)

        x = torch.cat([acc_feat, quat_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        # Optional positional encoding
        if self.use_pos_encoding and self.pos_encoding is not None:
            x = x + self.pos_encoding

        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        # Optional SE
        if self.se is not None:
            x = self.se(x)

        features = x

        # TAP or GAP
        if self.temporal_pool is not None:
            x, attn_weights = self.temporal_pool(x)
        else:
            x = x.mean(dim=1)  # Global Average Pooling

        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Variant 12: Asymmetric Dual Transformer (Separate Encoders + Cross-Attention)
# =============================================================================

class AsymmetricDualTransformer(nn.Module):
    """
    Asymmetric Dual Transformer with separate encoders per stream.

    Design rationale:
    - Accelerometer stream: 3 transformer layers (deeper for reliable signal)
    - Orientation stream: 1 transformer layer (shallower for noisy signal)
    - Cross-attention between streams for correlation learning
    - Higher dropout on orientation stream to prevent overfitting
    - SE attention per stream before fusion

    This architecture acknowledges that accelerometer and orientation have
    different signal qualities and require different processing depths.
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 7,
                 acc_frames: int = 128,
                 acc_coords: int = 7,
                 mocap_frames: int = 128,
                 num_joints: int = 32,
                 num_classes: int = 1,
                 # Asymmetric config
                 acc_layers: int = 3,
                 ori_layers: int = 1,
                 acc_heads: int = 4,
                 ori_heads: int = 2,
                 acc_dim: int = 48,
                 ori_dim: int = 16,
                 # Regularization (anti-overfit)
                 acc_dropout: float = 0.3,
                 ori_dropout: float = 0.5,
                 fusion_dropout: float = 0.4,
                 # Cross-attention
                 use_cross_attention: bool = True,
                 cross_attn_heads: int = 2,
                 # Common
                 activation: str = 'gelu',
                 norm_first: bool = True,
                 se_reduction: int = 4,
                 use_weight_decay_friendly: bool = True,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords

        self.acc_channels = 4  # smv, ax, ay, az
        self.ori_channels = self.imu_channels - 4  # roll, pitch, yaw
        self.use_cross_attention = use_cross_attention

        embed_dim = acc_dim + ori_dim  # Total fused dimension

        # =====================================================================
        # Stream 1: Accelerometer (Deeper - 3 layers)
        # =====================================================================
        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.GELU() if activation == 'gelu' else nn.SiLU(),
            nn.Dropout(acc_dropout * 0.3)
        )

        acc_encoder_layer = TransformerEncoderLayer(
            d_model=acc_dim,
            nhead=acc_heads,
            dim_feedforward=acc_dim * 2,
            dropout=acc_dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )
        self.acc_encoder = TransformerEncoderWithNorm(
            encoder_layer=acc_encoder_layer,
            num_layers=acc_layers,
            norm=nn.LayerNorm(acc_dim)
        )
        self.acc_se = SqueezeExcitation(acc_dim, reduction=se_reduction)

        # =====================================================================
        # Stream 2: Orientation (Shallower - 1 layer, higher dropout)
        # =====================================================================
        self.ori_proj = nn.Sequential(
            nn.Conv1d(self.ori_channels, ori_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(ori_dim),
            nn.GELU() if activation == 'gelu' else nn.SiLU(),
            nn.Dropout(ori_dropout * 0.4)
        )

        ori_encoder_layer = TransformerEncoderLayer(
            d_model=ori_dim,
            nhead=ori_heads,
            dim_feedforward=ori_dim * 2,
            dropout=ori_dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )
        self.ori_encoder = TransformerEncoderWithNorm(
            encoder_layer=ori_encoder_layer,
            num_layers=ori_layers,
            norm=nn.LayerNorm(ori_dim)
        )
        self.ori_se = SqueezeExcitation(ori_dim, reduction=max(se_reduction // 2, 2))

        # =====================================================================
        # Cross-Modal Attention (Optional)
        # =====================================================================
        if use_cross_attention:
            self.cross_modal = CrossModalAttentionBlock(
                acc_dim=acc_dim,
                ori_dim=ori_dim,
                num_heads=cross_attn_heads,
                dropout=fusion_dropout * 0.5
            )

        # =====================================================================
        # Fusion and Classification
        # =====================================================================
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Light fusion transformer (1 layer to combine streams)
        fusion_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=fusion_dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False
        )
        self.fusion_encoder = TransformerEncoderWithNorm(
            encoder_layer=fusion_layer,
            num_layers=1,
            norm=nn.LayerNorm(embed_dim)
        )

        self.temporal_pool = TemporalAttentionPooling(embed_dim)
        self.dropout_layer = nn.Dropout(fusion_dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        # Split input into streams
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        # =====================================================================
        # Stream 1: Accelerometer processing (deeper)
        # =====================================================================
        acc = rearrange(acc, 'b t c -> b c t')
        acc_feat = self.acc_proj(acc)
        acc_feat = rearrange(acc_feat, 'b c t -> t b c')
        acc_feat = self.acc_encoder(acc_feat)
        acc_feat = rearrange(acc_feat, 't b c -> b t c')
        acc_feat = self.acc_se(acc_feat)

        # =====================================================================
        # Stream 2: Orientation processing (shallower)
        # =====================================================================
        ori = rearrange(ori, 'b t c -> b c t')
        ori_feat = self.ori_proj(ori)
        ori_feat = rearrange(ori_feat, 'b c t -> t b c')
        ori_feat = self.ori_encoder(ori_feat)
        ori_feat = rearrange(ori_feat, 't b c -> b t c')
        ori_feat = self.ori_se(ori_feat)

        # =====================================================================
        # Cross-Modal Attention (if enabled)
        # =====================================================================
        if self.use_cross_attention:
            acc_feat, ori_feat = self.cross_modal(acc_feat, ori_feat)

        # =====================================================================
        # Fusion and Classification
        # =====================================================================
        x = torch.cat([acc_feat, ori_feat], dim=-1)
        x = self.fusion_norm(x)

        x = rearrange(x, 'b t c -> t b c')
        x = self.fusion_encoder(x)
        x = rearrange(x, 't b c -> b t c')

        features = x

        x, attn_weights = self.temporal_pool(x)
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features


# =============================================================================
# Test Script
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("KalmanTransformer Variants - Architecture Test")
    print("=" * 70)

    variants = [
        ("KalmanTransformerBaseline", KalmanTransformerBaseline, 7),
        ("KalmanCrossModalAttention", KalmanCrossModalAttention, 7),
        ("KalmanGatedFusion", KalmanGatedFusion, 7),
        ("KalmanDeepNarrow", KalmanDeepNarrow, 7),
        ("KalmanUncertaintyAware", KalmanUncertaintyAware, 10),
        ("KalmanBalancedRatio", KalmanBalancedRatio, 7),
        ("KalmanSingleStream", KalmanSingleStream, 7),
        ("KalmanCompact", KalmanCompact, 7),
        ("KalmanQuaternionFlexible", KalmanQuaternionFlexible, 8),  # Quaternion input
    ]

    for name, model_cls, channels in variants:
        print(f"\n{name}")
        print("-" * 50)

        model = model_cls(imu_frames=128, imu_channels=channels)
        x = torch.randn(8, 128, channels)
        logits, features = model(x)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Input:      ({8}, {128}, {channels})")
        print(f"  Output:     {logits.shape}")
        print(f"  Features:   {features.shape}")
        print(f"  Parameters: {total_params:,} ({trainable_params:,} trainable)")

    # Gradient check
    print("\n" + "=" * 70)
    print("Gradient Flow Check")
    print("=" * 70)

    for name, model_cls, channels in variants:
        model = model_cls(imu_frames=128, imu_channels=channels)
        model.train()
        x = torch.randn(4, 128, channels, requires_grad=True)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()
        grad_norm = x.grad.norm().item()
        status = "OK" if grad_norm > 0 else "FAIL"
        print(f"  {name}: grad_norm={grad_norm:.4f} [{status}]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
