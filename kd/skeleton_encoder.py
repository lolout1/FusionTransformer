"""
Skeleton encoder for teacher model.

Processes 32-joint skeleton data (96 channels = 32 joints * 3 xyz) through
a transformer architecture.
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointEmbedding(nn.Module):
    """
    Embed skeleton joints, optionally with joint-specific encodings.

    Input: (B, T, 96) where 96 = 32 joints * 3 coords
    Output: (B, T, embed_dim)
    """

    def __init__(
        self,
        num_joints: int = 32,
        coords_per_joint: int = 3,
        embed_dim: int = 64,
        use_joint_tokens: bool = False,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.coords_per_joint = coords_per_joint
        self.input_dim = num_joints * coords_per_joint
        self.use_joint_tokens = use_joint_tokens

        if use_joint_tokens:
            # Project each joint separately then aggregate
            self.joint_proj = nn.Linear(coords_per_joint, embed_dim // 4)
            self.joint_embed = nn.Parameter(torch.randn(num_joints, embed_dim // 4) * 0.02)
            self.aggregate = nn.Linear(num_joints * (embed_dim // 4), embed_dim)
        else:
            # Simple flatten and project
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.SiLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 96) skeleton sequence

        Returns:
            (B, T, embed_dim)
        """
        B, T, _ = x.shape

        if self.use_joint_tokens:
            # Reshape to (B, T, num_joints, 3)
            joints = x.view(B, T, self.num_joints, self.coords_per_joint)

            # Project each joint
            joint_feats = self.joint_proj(joints)  # (B, T, num_joints, embed_dim//4)

            # Add joint embeddings
            joint_feats = joint_feats + self.joint_embed.unsqueeze(0).unsqueeze(0)

            # Flatten and aggregate
            joint_feats = joint_feats.view(B, T, -1)
            return self.aggregate(joint_feats)
        else:
            return self.proj(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal or learnable positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 1024, learnable: bool = True):
        super().__init__()
        self.learnable = learnable

        if learnable:
            self.pe = nn.Parameter(torch.randn(max_len, embed_dim) * 0.02)
        else:
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class SkeletonTransformer(nn.Module):
    """
    Transformer-based skeleton encoder for teacher model.

    Architecture:
    1. JointEmbedding: (B, T, 96) -> (B, T, embed_dim)
    2. PositionalEncoding: add temporal position info
    3. TransformerEncoder: self-attention over time
    4. SqueezeExcitation: channel attention
    5. TemporalAttentionPooling: (B, T, embed_dim) -> (B, embed_dim)
    6. Classifier: (B, embed_dim) -> (B, 1)

    Follows same output convention as IMU models: returns (logits, features)
    """

    def __init__(
        self,
        num_joints: int = 32,
        coords_per_joint: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        se_reduction: int = 4,
        max_len: int = 1024,
        use_joint_tokens: bool = False,
        use_pos_enc: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_pos_enc = use_pos_enc

        # Joint embedding
        self.joint_embed = JointEmbedding(
            num_joints=num_joints,
            coords_per_joint=coords_per_joint,
            embed_dim=embed_dim,
            use_joint_tokens=use_joint_tokens,
        )

        # Positional encoding (optional)
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(embed_dim, max_len, learnable=True)
        else:
            self.pos_enc = None

        # Initial dropout
        self.dropout = nn.Dropout(dropout * 0.2)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Squeeze-Excitation for channel attention
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)

        # Temporal Attention Pooling
        self.pool = TemporalAttentionPooling(embed_dim, hidden_dim=embed_dim // 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        skeleton: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            skeleton: (B, T, 96) skeleton sequence
            mask: (B, T) True for valid frames
            return_attention: If True, return attention weights for KD

        Returns:
            logits: (B, 1)
            features: (B, embed_dim) pooled features for KD
            attention: (optional) Dict with 'pool': (B, T) attention weights
        """
        # Joint embedding
        x = self.joint_embed(skeleton)  # (B, T, embed_dim)

        # Add positional encoding (if enabled)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.dropout(x)

        # Transformer encoding
        if mask is not None:
            # Convert mask to attention mask format (True = ignore)
            src_key_padding_mask = ~mask
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer(x)

        # Channel attention
        x = self.se(x)  # (B, T, embed_dim)

        # Temporal pooling
        if return_attention:
            features, pool_attn = self.pool(x, return_weights=True)
        else:
            features = self.pool(x)  # (B, embed_dim)

        # Classification
        logits = self.classifier(features)  # (B, 1)

        if return_attention:
            return logits, features, {'pool': pool_attn}
        return logits, features

    def get_tokens(
        self,
        skeleton: torch.Tensor,
        num_tokens: int = 64,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get intermediate token representations for Gram KD.

        Pools skeleton frames to fixed number of tokens using average pooling.

        Args:
            skeleton: (B, T, 96)
            num_tokens: Output number of tokens
            mask: (B, T)

        Returns:
            tokens: (B, num_tokens, embed_dim)
        """
        # Get transformer output
        x = self.joint_embed(skeleton)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.dropout(x)

        if mask is not None:
            src_key_padding_mask = ~mask
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer(x)

        x = self.se(x)  # (B, T, embed_dim)

        # Pool to fixed number of tokens
        B, T, D = x.shape
        if T == num_tokens:
            return x

        # Use adaptive average pooling
        x = x.transpose(1, 2)  # (B, D, T)
        x = F.adaptive_avg_pool1d(x, num_tokens)  # (B, D, num_tokens)
        x = x.transpose(1, 2)  # (B, num_tokens, D)

        return x


class SqueezeExcitation(nn.Module):
    """Channel attention via squeeze-excitation."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        squeeze = x.mean(dim=1)  # (B, C)
        excitation = self.fc(squeeze)  # (B, C)
        return x * excitation.unsqueeze(1)


class TemporalAttentionPooling(nn.Module):
    """Attention-weighted temporal pooling."""

    def __init__(self, embed_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, T, C)
        B, T, C = x.shape

        # Handle empty sequence
        if T == 0:
            pooled = torch.zeros(B, C, device=x.device, dtype=x.dtype)
            if return_weights:
                return pooled, torch.zeros(B, 0, device=x.device)
            return pooled

        attn_scores = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)
        pooled = (x * attn_weights).sum(dim=1)  # (B, C)

        if return_weights:
            return pooled, attn_weights.squeeze(-1)  # (B, C), (B, T)
        return pooled


class JointSkeletonIMUTeacher(nn.Module):
    """
    Joint teacher model that processes both skeleton and IMU.

    For experiments where teacher sees both modalities during training.
    """

    def __init__(
        self,
        num_joints: int = 32,
        imu_channels: int = 7,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Skeleton encoder
        self.skeleton_encoder = SkeletonTransformer(
            num_joints=num_joints,
            embed_dim=embed_dim // 2,
            num_heads=num_heads // 2,
            num_layers=num_layers,
            dropout=dropout,
        )

        # IMU encoder (simplified - uses same transformer pattern)
        self.imu_proj = nn.Sequential(
            nn.Linear(imu_channels, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.SiLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim // 2,
            nhead=num_heads // 2,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.imu_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.imu_pool = TemporalAttentionPooling(embed_dim // 2)

        # Fusion and classifier
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(
        self,
        skeleton: torch.Tensor,
        imu: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            skeleton: (B, T_skel, 96)
            imu: (B, T_imu, imu_channels)

        Returns:
            logits: (B, 1)
            features: (B, embed_dim) fused features
        """
        # Encode skeleton
        _, skel_feat = self.skeleton_encoder(skeleton)  # (B, embed_dim//2)

        # Encode IMU
        imu_x = self.imu_proj(imu)
        imu_x = self.imu_transformer(imu_x)
        imu_feat = self.imu_pool(imu_x)  # (B, embed_dim//2)

        # Fuse
        fused = torch.cat([skel_feat, imu_feat], dim=-1)  # (B, embed_dim)
        features = self.fusion(fused)

        # Classify
        logits = self.classifier(features)

        return logits, features
