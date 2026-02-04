"""
Event-to-Token Resampler for irregular time-series.

Maps variable-length events with irregular timestamps to fixed-length token sequences
via learned time-anchored cross-attention queries.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFeatureEncoder(nn.Module):
    """
    Encodes raw timestamps into drift-invariant time features.

    Features:
    - delta_t: time since previous event
    - log(1 + delta_t): compressed representation of large gaps
    - tau: normalized position in window [0, 1]
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(3, embed_dim)

    def forward(
        self,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            timestamps: (B, N) timestamps in seconds
            mask: (B, N) True for valid events

        Returns:
            time_features: (B, N, embed_dim)
        """
        B, N = timestamps.shape

        # Compute delta_t (time since previous event)
        delta_t = torch.zeros_like(timestamps)
        delta_t[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]

        # Handle invalid deltas (negative or very large)
        delta_t = torch.clamp(delta_t, min=0, max=60.0)  # Cap at 60 seconds

        # Log-compressed delta
        log_delta = torch.log1p(delta_t)

        # Normalized position tau in [0, 1]
        t_start = timestamps[:, :1]  # (B, 1)
        t_end = timestamps[:, -1:]   # (B, 1)
        duration = torch.clamp(t_end - t_start, min=1e-6)
        tau = (timestamps - t_start) / duration

        # Stack and project
        time_feats = torch.stack([delta_t, log_delta, tau], dim=-1)  # (B, N, 3)
        return self.proj(time_feats)


class EventTokenResampler(nn.Module):
    """
    Maps irregular events to fixed L tokens via learned time queries.

    Architecture:
    1. Encode events with time-aware features
    2. Create L learned queries anchored at tau_j = j/(L-1)
    3. Cross-attention: queries attend to events
    4. Output: L fixed tokens regardless of input length

    This is robust to irregular sampling because:
    - Time features are drift-invariant (delta_t, log-delta, normalized tau)
    - Cross-attention soft-aligns queries to events
    - No hard alignment between input positions and output positions
    """

    def __init__(
        self,
        input_dim: int = 6,
        embed_dim: int = 48,
        num_tokens: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        time_embed_dim: int = 16,
    ):
        """
        Args:
            input_dim: Dimension of input events (e.g., 6 for acc+gyro)
            embed_dim: Output embedding dimension
            num_tokens: Number of output tokens (L)
            num_heads: Attention heads
            dropout: Dropout rate
            time_embed_dim: Dimension of time feature encoding
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # Time feature encoder
        self.time_encoder = TimeFeatureEncoder(time_embed_dim)

        # Event projection (input_dim + time_embed_dim -> embed_dim)
        self.event_proj = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
        )

        # Learned time-anchored queries
        # Initialize at uniform positions tau_j = j/(L-1) for j=0..L-1
        tau_init = torch.linspace(0, 1, num_tokens).unsqueeze(-1)  # (L, 1)
        self.register_buffer('tau_anchors', tau_init)

        # Query embedding from tau position
        self.query_embed = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Learnable query bias (adds flexibility beyond just position)
        self.query_bias = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.02)

        # Cross-attention: queries attend to events
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output normalization
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        events: torch.Tensor,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            events: (B, N, input_dim) - variable N per sample
            timestamps: (B, N) - event timestamps in seconds
            mask: (B, N) - True for valid events, False for padding
            return_attn: Whether to return attention weights

        Returns:
            tokens: (B, L, embed_dim) - fixed L tokens
            attn_weights: (B, L, N) attention weights if return_attn=True
        """
        B, N, _ = events.shape
        device = events.device

        # Handle empty or very small sequences
        if N == 0:
            # Return zero tokens
            tokens = torch.zeros(B, self.num_tokens, self.embed_dim, device=device)
            attn = torch.zeros(B, self.num_tokens, 1, device=device) if return_attn else None
            return tokens, attn

        # Create mask if not provided (assume all valid)
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Encode time features
        time_feats = self.time_encoder(timestamps, mask)  # (B, N, time_embed_dim)

        # Concatenate events with time features
        event_input = torch.cat([events, time_feats], dim=-1)  # (B, N, input_dim + time_embed_dim)

        # Project events to embedding space
        event_embed = self.event_proj(event_input)  # (B, N, embed_dim)

        # Create queries from time anchors
        queries = self.query_embed(self.tau_anchors)  # (L, embed_dim)
        queries = queries.unsqueeze(0).expand(B, -1, -1)  # (B, L, embed_dim)
        queries = queries + self.query_bias.unsqueeze(0)  # Add learnable bias

        # Cross-attention: queries attend to events
        # key_padding_mask: True means IGNORE (opposite of our mask convention)
        key_padding_mask = ~mask  # (B, N)

        tokens, attn_weights = self.cross_attn(
            query=queries,
            key=event_embed,
            value=event_embed,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
        )

        # Normalize output
        tokens = self.out_norm(tokens)  # (B, L, embed_dim)

        if return_attn:
            return tokens, attn_weights
        return tokens, None


class TimestampAwareStudent(nn.Module):
    """
    Complete student model with EventTokenResampler + Transformer encoder.

    Architecture:
    1. EventTokenResampler: (B, N, input_dim) -> (B, L, embed_dim)
    2. TransformerEncoder: temporal modeling
    3. SqueezeExcitation: channel attention
    4. TemporalAttentionPooling: (B, L, embed_dim) -> (B, embed_dim)
    5. Classifier: (B, embed_dim) -> (B, 1)
    """

    def __init__(
        self,
        input_dim: int = 6,
        embed_dim: int = 48,
        num_tokens: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        se_reduction: int = 4,
    ):
        super().__init__()

        self.resampler = EventTokenResampler(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            dropout=dropout * 0.2,
        )

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

        # Squeeze-Excitation
        self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)

        # Temporal Attention Pooling
        self.pool = TemporalAttentionPooling(embed_dim, hidden_dim=embed_dim // 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        events: torch.Tensor,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            events: (B, N, input_dim)
            timestamps: (B, N)
            mask: (B, N)

        Returns:
            logits: (B, 1)
            features: (B, embed_dim) pooled features for KD
        """
        # Resample to fixed tokens
        tokens, _ = self.resampler(events, timestamps, mask)  # (B, L, embed_dim)

        # Transformer encoding
        tokens = self.transformer(tokens)  # (B, L, embed_dim)

        # Channel attention
        tokens = self.se(tokens)  # (B, L, embed_dim)

        # Temporal pooling
        features = self.pool(tokens)  # (B, embed_dim)

        # Classification
        logits = self.classifier(features)  # (B, 1)

        return logits, features


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        attn_scores = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)
        return (x * attn_weights).sum(dim=1)  # (B, C)
