"""
Kalman Gated Hierarchical Fusion (KGHF) for Fall Detection.

Multi-scale dual-stream architecture with gated fusion.
Input: 7ch [smv, ax, ay, az, roll, pitch, yaw]
"""

import math
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F

try:
    from einops import rearrange
except ImportError:
    raise ImportError("Please install einops: pip install einops")


# =============================================================================
# Core Components
# =============================================================================

class SqueezeExcitation(nn.Module):
    """Channel attention with optional residual connection."""

    def __init__(self, channels: int, reduction: int = 4, use_residual: bool = False):
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
        scale = x.mean(dim=1)
        scale = self.fc(scale).unsqueeze(1)
        out = x * scale
        if self.use_residual:
            out = out + x
        return out


class TemporalAttentionPooling(nn.Module):
    """Learnable temporal pooling with attention weights output."""

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
    """Transformer encoder with guaranteed final normalization."""

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


# =============================================================================
# Multi-Scale Feature Extraction
# =============================================================================

class MultiScaleConv1D(nn.Module):
    """
    Multi-scale 1D convolution with parallel branches.

    Extracts features at three temporal scales:
    - Local (k=3): Captures fine-grained patterns like impact spikes
    - Medium (k=7): Captures mid-range patterns like oscillations
    - Global (k=15): Captures broader context like posture changes

    Args:
        in_channels: Input channels
        out_channels: Output channels per scale (total = out_channels)
        dropout: Dropout rate
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()

        # Split output channels across scales (roughly equal)
        self.scale_dim = out_channels // 3
        remainder = out_channels - (self.scale_dim * 3)

        # Local patterns (fine-grained, fast changes)
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels, self.scale_dim, kernel_size=3, padding='same'),
            nn.BatchNorm1d(self.scale_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Medium patterns (oscillations, transitions)
        self.medium_conv = nn.Sequential(
            nn.Conv1d(in_channels, self.scale_dim, kernel_size=7, padding='same'),
            nn.BatchNorm1d(self.scale_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Global patterns (posture, slow changes)
        self.global_conv = nn.Sequential(
            nn.Conv1d(in_channels, self.scale_dim + remainder, kernel_size=15, padding='same'),
            nn.BatchNorm1d(self.scale_dim + remainder),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Scale attention - learn importance of each scale
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T)
        Returns:
            Multi-scale features of shape (B, out_channels, T)
        """
        local = self.local_conv(x)    # (B, scale_dim, T)
        medium = self.medium_conv(x)  # (B, scale_dim, T)
        global_ = self.global_conv(x) # (B, scale_dim+r, T)

        # Concatenate scales
        multi_scale = torch.cat([local, medium, global_], dim=1)  # (B, out_channels, T)

        return multi_scale


# =============================================================================
# Gated Fusion Mechanisms
# =============================================================================

class HierarchicalGatedFusion(nn.Module):
    """
    Three-level hierarchical gated fusion between acc and ori streams.

    Level 1 - Feature Gates: Per-channel importance
    Level 2 - Temporal Gates: Per-timestep importance
    Level 3 - Stream Gates: Overall stream confidence

    This allows the model to dynamically decide:
    - Which features are important (Level 1)
    - When they are important (Level 2)
    - Which stream to trust more (Level 3)

    Args:
        acc_dim: Accelerometer stream dimension
        ori_dim: Orientation stream dimension
        out_dim: Output dimension
        dropout: Dropout rate
    """

    def __init__(self, acc_dim: int, ori_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()

        self.acc_dim = acc_dim
        self.ori_dim = ori_dim
        self.out_dim = out_dim

        # Level 1: Feature-level gates (per channel)
        self.acc_feature_gate = nn.Sequential(
            nn.Linear(acc_dim, acc_dim),
            nn.Sigmoid()
        )
        self.ori_feature_gate = nn.Sequential(
            nn.Linear(ori_dim, ori_dim),
            nn.Sigmoid()
        )

        # Level 2: Temporal gates (per timestep)
        self.temporal_gate = nn.Sequential(
            nn.Linear(acc_dim + ori_dim, (acc_dim + ori_dim) // 2),
            nn.Tanh(),
            nn.Linear((acc_dim + ori_dim) // 2, 2),  # 2 gates: acc, ori
            nn.Softmax(dim=-1)
        )

        # Level 3: Stream confidence gate (global)
        self.stream_confidence = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(acc_dim + ori_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

        # Projection to output dimension
        self.acc_proj = nn.Linear(acc_dim, out_dim // 2)
        self.ori_proj = nn.Linear(ori_dim, out_dim // 2)

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, acc_feat: torch.Tensor, ori_feat: torch.Tensor) -> tuple:
        """
        Args:
            acc_feat: (B, T, acc_dim)
            ori_feat: (B, T, ori_dim)
        Returns:
            fused: (B, T, out_dim)
            gate_info: dict with gating statistics for analysis
        """
        B, T, _ = acc_feat.shape

        # Level 1: Feature gates
        acc_fgate = self.acc_feature_gate(acc_feat)  # (B, T, acc_dim)
        ori_fgate = self.ori_feature_gate(ori_feat)  # (B, T, ori_dim)
        acc_gated = acc_feat * acc_fgate
        ori_gated = ori_feat * ori_fgate

        # Level 2: Temporal gates
        combined = torch.cat([acc_gated, ori_gated], dim=-1)  # (B, T, acc_dim+ori_dim)
        temporal_gates = self.temporal_gate(combined)  # (B, T, 2)
        acc_tgate = temporal_gates[:, :, 0:1]  # (B, T, 1)
        ori_tgate = temporal_gates[:, :, 1:2]

        # Level 3: Stream confidence
        combined_t = rearrange(combined, 'b t c -> b c t')
        stream_conf = self.stream_confidence(combined_t)  # (B, 2)
        acc_sgate = stream_conf[:, 0:1].unsqueeze(1)  # (B, 1, 1)
        ori_sgate = stream_conf[:, 1:2].unsqueeze(1)

        # Apply hierarchical gates
        acc_final = acc_gated * acc_tgate * acc_sgate
        ori_final = ori_gated * ori_tgate * ori_sgate

        # Project and fuse
        acc_proj = self.acc_proj(acc_final)  # (B, T, out_dim//2)
        ori_proj = self.ori_proj(ori_final)  # (B, T, out_dim//2)
        fused = torch.cat([acc_proj, ori_proj], dim=-1)  # (B, T, out_dim)
        fused = self.fusion(fused)

        # Gate info for analysis
        gate_info = {
            'acc_feature_gate_mean': acc_fgate.mean().item(),
            'ori_feature_gate_mean': ori_fgate.mean().item(),
            'acc_temporal_gate_mean': acc_tgate.mean().item(),
            'ori_temporal_gate_mean': ori_tgate.mean().item(),
            'acc_stream_conf': acc_sgate.mean().item(),
            'ori_stream_conf': ori_sgate.mean().item(),
        }

        return fused, gate_info


class ResidualGate(nn.Module):
    """
    Learnable residual gate that allows skipping transformations.

    output = gate * transformed + (1 - gate) * input

    This helps prevent overfitting by allowing the model to
    pass through inputs unchanged when transformations aren't helpful.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, transformed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Original input (B, T, C)
            transformed: Transformed input (B, T, C)
        Returns:
            Gated combination
        """
        g = self.gate(x.mean(dim=1, keepdim=True))  # (B, 1, 1)
        return g * transformed + (1 - g) * x


# =============================================================================
# Main Model
# =============================================================================

class KalmanGatedHierarchical(nn.Module):
    """
    Kalman Gated Hierarchical Fusion (KGHF) for fall detection.

    This model extends the baseline KalmanBest architecture with:
    1. Multi-scale temporal feature extraction
    2. Hierarchical gated fusion (3 levels)
    3. Residual gating for better generalization
    4. Enhanced regularization

    Design Philosophy:
    - The baseline model shows large val-test gaps on some subjects
    - This suggests subject-specific overfitting
    - Hierarchical features capture subject-invariant patterns
    - Gated fusion learns adaptive stream weighting
    - Residual gates allow skipping unhelpful transformations

    Anti-Overfitting Measures:
    - Multi-scale features (more robust representations)
    - Gated fusion (adaptive, not fixed ratios)
    - Residual gates (can skip transformations)
    - Higher base dropout with layer-specific scaling
    - Label smoothing compatible loss
    - Feature noise injection during training

    Args:
        imu_frames: Sequence length (default: 128)
        imu_channels: Input channels (default: 7)
        num_classes: Output classes (default: 1)
        num_heads: Transformer heads (default: 4)
        num_layers: Transformer layers (default: 2)
        embed_dim: Embedding dimension (default: 64)
        dropout: Base dropout rate (default: 0.4)
        acc_channels: Accelerometer channels (default: 4)
        use_multi_scale: Enable multi-scale conv (default: True)
        use_hierarchical_gate: Enable 3-level gating (default: True)
        use_residual_gate: Enable residual gating (default: True)
        feature_noise: Training feature noise std (default: 0.1)
    """

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
        embed_dim: int = 64,
        dropout: float = 0.4,
        activation: str = 'gelu',
        norm_first: bool = True,
        se_reduction: int = 4,
        acc_channels: int = 4,
        use_multi_scale: bool = True,
        use_hierarchical_gate: bool = True,
        use_residual_gate: bool = True,
        feature_noise: float = 0.1,
        **kwargs
    ):
        super().__init__()

        # Configuration
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.embed_dim = embed_dim
        self.use_multi_scale = use_multi_scale
        self.use_hierarchical_gate = use_hierarchical_gate
        self.use_residual_gate = use_residual_gate
        self.feature_noise = feature_noise

        # Channel split
        self.acc_channels = acc_channels  # [smv, ax, ay, az]
        self.ori_channels = self.imu_channels - acc_channels  # [roll, pitch, yaw]

        # Stream dimensions (50/50 split)
        acc_dim = embed_dim // 2
        ori_dim = embed_dim - acc_dim

        # ---------------------------------------------------------------------
        # Accelerometer Stream Processing
        # ---------------------------------------------------------------------
        if use_multi_scale:
            self.acc_proj = MultiScaleConv1D(
                in_channels=self.acc_channels,
                out_channels=acc_dim,
                dropout=dropout * 0.3
            )
        else:
            self.acc_proj = nn.Sequential(
                nn.Conv1d(self.acc_channels, acc_dim, kernel_size=8, padding='same'),
                nn.BatchNorm1d(acc_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.3)
            )

        self.acc_se = SqueezeExcitation(acc_dim, reduction=se_reduction, use_residual=True)

        # ---------------------------------------------------------------------
        # Orientation Stream Processing
        # ---------------------------------------------------------------------
        if use_multi_scale:
            self.ori_proj = MultiScaleConv1D(
                in_channels=self.ori_channels,
                out_channels=ori_dim,
                dropout=dropout * 0.4  # Higher dropout for noisy ori
            )
        else:
            self.ori_proj = nn.Sequential(
                nn.Conv1d(self.ori_channels, ori_dim, kernel_size=8, padding='same'),
                nn.BatchNorm1d(ori_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.4)
            )

        self.ori_se = SqueezeExcitation(ori_dim, reduction=se_reduction, use_residual=True)

        # ---------------------------------------------------------------------
        # Hierarchical Gated Fusion
        # ---------------------------------------------------------------------
        if use_hierarchical_gate:
            self.gated_fusion = HierarchicalGatedFusion(
                acc_dim=acc_dim,
                ori_dim=ori_dim,
                out_dim=embed_dim,
                dropout=dropout * 0.5
            )
        else:
            # Simple concatenation fallback
            self.gated_fusion = None
            self.simple_fusion = nn.Sequential(
                nn.Linear(acc_dim + ori_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5)
            )

        self.fusion_norm = nn.LayerNorm(embed_dim)

        # ---------------------------------------------------------------------
        # Transformer Encoder
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Residual Gate (after transformer)
        # ---------------------------------------------------------------------
        if use_residual_gate:
            self.residual_gate = ResidualGate(embed_dim)
        else:
            self.residual_gate = None

        # ---------------------------------------------------------------------
        # Final Processing
        # ---------------------------------------------------------------------
        self.se_final = SqueezeExcitation(embed_dim, reduction=se_reduction)
        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        # Classification head with extra regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, num_classes)
        )

        self._init_weights()

        # Store gate info for analysis
        self.last_gate_info = None

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs) -> tuple:
        """
        Forward pass with optional feature noise injection.

        Args:
            acc_data: IMU tensor (B, T, 7)
            skl_data: Unused (skeleton data)

        Returns:
            logits: (B, 1)
            features: (B, T, embed_dim)
        """
        # Split streams
        acc = acc_data[:, :, :self.acc_channels]
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

        # Feature noise injection during training (anti-overfit)
        if self.training and self.feature_noise > 0:
            acc = acc + torch.randn_like(acc) * self.feature_noise
            ori = ori + torch.randn_like(ori) * self.feature_noise * 0.5  # Less noise for ori

        # Reshape for Conv1d
        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        # Stream projections
        acc_feat = self.acc_proj(acc)  # (B, acc_dim, T)
        ori_feat = self.ori_proj(ori)  # (B, ori_dim, T)

        # Reshape for attention: (B, C, T) -> (B, T, C)
        acc_feat = rearrange(acc_feat, 'b c t -> b t c')
        ori_feat = rearrange(ori_feat, 'b c t -> b t c')

        # Stream SE attention
        acc_feat = self.acc_se(acc_feat)
        ori_feat = self.ori_se(ori_feat)

        # Hierarchical gated fusion
        if self.gated_fusion is not None:
            x, gate_info = self.gated_fusion(acc_feat, ori_feat)
            self.last_gate_info = gate_info
        else:
            combined = torch.cat([acc_feat, ori_feat], dim=-1)
            x = self.simple_fusion(combined)
            self.last_gate_info = None

        x = self.fusion_norm(x)

        # Store pre-transformer features for residual gate
        pre_transformer = x

        # Transformer
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        # Residual gate (allows skipping transformer if unhelpful)
        if self.residual_gate is not None:
            x = self.residual_gate(pre_transformer, x)

        # Final SE and pooling
        x = self.se_final(x)
        features = x

        x, attn_weights = self.temporal_pool(x)

        # Classification
        logits = self.classifier(x)

        return logits, features

    def get_gate_analysis(self) -> dict:
        """Get the last gate info for analysis."""
        return self.last_gate_info

    def get_attention_weights(self, acc_data: torch.Tensor) -> torch.Tensor:
        """Get temporal attention weights for visualization."""
        with torch.no_grad():
            _, features = self.forward(acc_data)
            _, weights = self.temporal_pool(features)
        return weights


# =============================================================================
# Model Variants for Ablation
# =============================================================================

class KGHFLite(KalmanGatedHierarchical):
    """
    Lighter version with fewer parameters for comparison.
    Disables multi-scale and residual gate.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('use_multi_scale', False)
        kwargs.setdefault('use_residual_gate', False)
        kwargs.setdefault('embed_dim', 48)
        super().__init__(**kwargs)


class KGHFNoGate(KalmanGatedHierarchical):
    """
    Version without hierarchical gating for ablation.
    Uses simple concatenation instead.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('use_hierarchical_gate', False)
        super().__init__(**kwargs)


class KGHFMaxReg(KalmanGatedHierarchical):
    """
    Maximum regularization version.
    Higher dropout, noise, designed for minimal overfitting.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('dropout', 0.5)
        kwargs.setdefault('feature_noise', 0.15)
        kwargs.setdefault('embed_dim', 48)
        super().__init__(**kwargs)


# =============================================================================
# Factory Functions
# =============================================================================

def create_kghf(variant: str = 'default', **kwargs) -> KalmanGatedHierarchical:
    """
    Factory function to create KGHF model variants.

    Args:
        variant: 'default', 'lite', 'no_gate', 'max_reg'
        **kwargs: Override model arguments

    Returns:
        Model instance
    """
    variants = {
        'default': KalmanGatedHierarchical,
        'lite': KGHFLite,
        'no_gate': KGHFNoGate,
        'max_reg': KGHFMaxReg,
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")

    return variants[variant](**kwargs)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Kalman Gated Hierarchical Fusion (KGHF) Model Test")
    print("=" * 70)

    variants = [
        ('KGHF Default', KalmanGatedHierarchical, {}),
        ('KGHF Lite', KGHFLite, {}),
        ('KGHF NoGate', KGHFNoGate, {}),
        ('KGHF MaxReg', KGHFMaxReg, {}),
    ]

    for name, model_cls, kwargs in variants:
        print(f"\n{name}")
        print("-" * 50)

        model = model_cls(**kwargs)
        model.train()

        x = torch.randn(4, 128, 7)
        logits, features = model(x)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Input:      (4, 128, 7)")
        print(f"  Output:     {logits.shape}")
        print(f"  Features:   {features.shape}")
        print(f"  Parameters: {params:,}")

        # Gate analysis
        gate_info = model.get_gate_analysis()
        if gate_info:
            print(f"  Gate Info:")
            print(f"    Acc stream conf: {gate_info['acc_stream_conf']:.3f}")
            print(f"    Ori stream conf: {gate_info['ori_stream_conf']:.3f}")

        # Gradient check
        loss = logits.sum()
        loss.backward()
        print(f"  Gradient:   OK")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
