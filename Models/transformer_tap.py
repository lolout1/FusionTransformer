"""
Transformer with Temporal Attention Pooling ONLY (No SE module).

This is an ablation variant to isolate TAP's contribution.
Compared to TransformerSE: removes Squeeze-Excitation, keeps TAP.

Architecture:
    - Input: variable channels (acc, acc+smv, acc+gyro, etc.)
    - Conv1d input projection with BatchNorm
    - TransformerEncoder (2 layers, 4 heads, 64d)
    - NO Squeeze-Excitation (removed for ablation)
    - Temporal Attention Pooling (learnable temporal aggregation)
    - Linear classifier
"""

import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
from einops import rearrange
import math


class TemporalAttentionPooling(nn.Module):
    """Learnable temporal pooling to focus on fall events."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted temporal pooling. x: (B, T, C)"""
        scores = self.attention(x).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1)  # Normalized attention: (B, T)
        context = torch.einsum('bt,btc->bc', weights, x)  # Weighted sum: (B, C)
        return context, weights


class TransformerEncoderWithNorm(nn.TransformerEncoder):
    """Standard transformer encoder with final layer normalization."""

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransModelTAP(nn.Module):
    """
    Transformer with Temporal Attention Pooling ONLY (No SE).

    This model is identical to TransModelSE but WITHOUT the
    Squeeze-Excitation module, allowing us to isolate TAP's contribution.

    Architecture:
        - Conv1d input projection (kernel=8)
        - BatchNorm + Dropout
        - TransformerEncoder (2 layers, 4 heads)
        - NO Squeeze-Excitation
        - Temporal Attention Pooling (learnable)
        - Dropout + Linear classifier
    """

    def __init__(self,
                 acc_frames: int = 128,
                 acc_coords: int = 4,
                 num_classes: int = 1,
                 num_heads: int = 4,
                 num_layer: int = 2,
                 embed_dim: int = 64,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 norm_first: bool = True,
                 **kwargs):
        super().__init__()

        # Input projection: channels -> embed_dim with temporal conv
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout * 0.3)  # Same as TransformerSE
        )

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
            num_layers=num_layer,
            norm=nn.LayerNorm(embed_dim)
        )

        # TAP only - NO SE module
        self.temporal_pool = TemporalAttentionPooling(embed_dim)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize output layer with scaled normal distribution."""
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs):
        """
        Forward pass.

        Args:
            acc_data: (B, T, C) accelerometer data
            skl_data: Unused, for API compatibility

        Returns:
            logits: (B, num_classes) classification logits
            features: (B, T, embed_dim) encoder features before pooling
        """
        # Input projection: (B, T, C) -> (B, C, T) -> (B, embed_dim, T)
        x = rearrange(acc_data, 'b t c -> b c t')
        x = self.input_proj(x)

        # Transformer: (B, embed_dim, T) -> (T, B, embed_dim)
        x = rearrange(x, 'b c t -> t b c')
        x = self.encoder(x)

        # Rearrange: (T, B, C) -> (B, T, C)
        x = rearrange(x, 't b c -> b t c')
        features = x  # Save for potential feature extraction

        # Temporal attention pooling: (B, T, C) -> (B, C)
        # NO SE module here - direct to TAP
        x, attn_weights = self.temporal_pool(x)

        # Classification
        x = self.dropout(x)
        logits = self.output(x)

        return logits, features


if __name__ == "__main__":
    print("=" * 60)
    print("TransModelTAP: Transformer with TAP only (no SE)")
    print("=" * 60)

    # Test with different channel configurations
    for channels, desc in [(4, "acc+smv"), (6, "acc+gyro")]:
        model = TransModelTAP(acc_frames=128, acc_coords=channels, num_classes=1)
        x = torch.randn(16, 128, channels)
        logits, features = model(x)

        total_params = sum(p.numel() for p in model.parameters())
        tap_params = sum(p.numel() for p in model.temporal_pool.parameters())

        print(f"\n{desc} ({channels} channels):")
        print(f"  Total parameters: {total_params:,}")
        print(f"  TAP parameters: {tap_params:,}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Features shape: {features.shape}")
