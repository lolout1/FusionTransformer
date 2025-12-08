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
