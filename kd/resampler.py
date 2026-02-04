"""
Event-to-Token Resampler for irregular time-series.

Maps variable-length events with irregular timestamps to fixed-length token sequences
via learned time-anchored cross-attention queries.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFeatureEncoder(nn.Module):
    """
    Encodes timestamps into time features.

    Modes:
    - 'position': Use sequence position only (tau = i/N) - RECOMMENDED
    - 'timestamps': Use actual timestamps (delta_t, log_delta, tau)
    - 'cleaned': Use timestamps with gap clipping and duplicate handling

    Note: 'position' mode significantly outperforms timestamp-based modes on
    datasets with unreliable timestamps (e.g., SmartFallMM: 62.0% vs 55.3% F1).
    """

    def __init__(self, embed_dim: int = 16, mode: str = 'position'):
        super().__init__()
        self.mode = mode

        if mode == 'position':
            self.proj = nn.Linear(1, embed_dim)  # Only tau
        else:
            self.proj = nn.Linear(3, embed_dim)  # delta_t, log_delta, tau

    def forward(
        self,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            timestamps: (B, N) timestamps in seconds (ignored if mode='position')
            mask: (B, N) True for valid events

        Returns:
            time_features: (B, N, embed_dim)
        """
        B, N = timestamps.shape
        device = timestamps.device

        if self.mode == 'position':
            # Ignore timestamps, use sequence position
            tau = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)
            return self.proj(tau.unsqueeze(-1))

        # Compute delta_t (time since previous event)
        delta_t = torch.zeros_like(timestamps)
        delta_t[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]

        if self.mode == 'cleaned':
            # Clip large gaps and handle duplicates
            delta_t = torch.clamp(delta_t, min=1e-4, max=0.1)  # 0.1ms to 100ms
        else:
            # Cap at 60 seconds
            delta_t = torch.clamp(delta_t, min=0, max=60.0)

        # Log-compressed delta
        log_delta = torch.log1p(delta_t)

        # Normalized position tau in [0, 1]
        t_start = timestamps[:, :1]
        t_end = timestamps[:, -1:]
        duration = torch.clamp(t_end - t_start, min=1e-6)
        tau = (timestamps - t_start) / duration

        # Stack and project
        time_feats = torch.stack([delta_t, log_delta, tau], dim=-1)
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
        time_mode: str = 'position',
    ):
        """
        Args:
            input_dim: Dimension of input events (e.g., 6 for acc+gyro)
            embed_dim: Output embedding dimension
            num_tokens: Number of output tokens (L)
            num_heads: Attention heads
            dropout: Dropout rate
            time_embed_dim: Dimension of time feature encoding
            time_mode: 'position' (ignore timestamps), 'timestamps', or 'cleaned'
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.time_mode = time_mode

        # Time feature encoder
        self.time_encoder = TimeFeatureEncoder(time_embed_dim, mode=time_mode)

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
        time_mode: str = 'position',
    ):
        super().__init__()
        self.time_mode = time_mode

        self.resampler = EventTokenResampler(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            dropout=dropout * 0.2,
            time_mode=time_mode,
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
        return_attention: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            events: (B, N, input_dim)
            timestamps: (B, N)
            mask: (B, N)
            return_attention: If True, return attention weights for KD

        Returns:
            logits: (B, 1)
            features: (B, embed_dim) pooled features for KD
            attention: (optional) Dict with 'pool': (B, L) attention weights
        """
        # Resample to fixed tokens
        tokens, _ = self.resampler(events, timestamps, mask)  # (B, L, embed_dim)

        # Transformer encoding
        tokens = self.transformer(tokens)  # (B, L, embed_dim)

        # Channel attention
        tokens = self.se(tokens)  # (B, L, embed_dim)

        # Temporal pooling
        if return_attention:
            features, pool_attn = self.pool(tokens, return_weights=True)
        else:
            features = self.pool(tokens)  # (B, embed_dim)

        # Classification
        logits = self.classifier(features)  # (B, 1)

        if return_attention:
            return logits, features, {'pool': pool_attn}
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


class DualStreamResampler(nn.Module):
    """
    Dual-stream resampler with separate encoders for acc and gyro.

    Mirrors the main FusionTransformer's asymmetric capacity approach:
    - Accelerometer: 65% of embedding capacity (high-frequency impacts)
    - Gyroscope: 35% of embedding capacity (smoother rotations)

    Each stream has its own EventTokenResampler, outputs are concatenated.
    """

    def __init__(
        self,
        acc_dim: int = 3,
        gyro_dim: int = 3,
        embed_dim: int = 48,
        num_tokens: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        acc_ratio: float = 0.65,
        time_mode: str = 'position',
    ):
        super().__init__()
        self.acc_dim = acc_dim
        self.gyro_dim = gyro_dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Asymmetric capacity with head-count alignment
        acc_heads = max(1, int(num_heads * acc_ratio))
        gyro_heads = max(1, num_heads - acc_heads)

        # Compute embed dims divisible by head counts
        acc_embed = (embed_dim * acc_heads // num_heads)
        acc_embed = acc_embed - (acc_embed % acc_heads)  # Ensure divisible
        gyro_embed = embed_dim - acc_embed
        gyro_embed = gyro_embed - (gyro_embed % gyro_heads)  # Ensure divisible

        # Recompute total (may be slightly less than embed_dim)
        self._acc_embed = acc_embed
        self._gyro_embed = gyro_embed

        self.acc_resampler = EventTokenResampler(
            input_dim=acc_dim,
            embed_dim=acc_embed,
            num_tokens=num_tokens,
            num_heads=acc_heads,
            dropout=dropout,
            time_mode=time_mode,
        )

        self.gyro_resampler = EventTokenResampler(
            input_dim=gyro_dim,
            embed_dim=gyro_embed,
            num_tokens=num_tokens,
            num_heads=gyro_heads,
            dropout=dropout,
            time_mode=time_mode,
        )

        # Fusion layer (project concatenated streams to target embed_dim)
        concat_dim = acc_embed + gyro_embed
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        events: torch.Tensor,
        timestamps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            events: (B, N, acc_dim + gyro_dim) - concatenated acc + gyro
            timestamps: (B, N)
            mask: (B, N)

        Returns:
            tokens: (B, L, embed_dim)
            attn_weights: tuple of (acc_attn, gyro_attn) if return_attn
        """
        # Split into acc and gyro
        acc_events = events[..., :self.acc_dim]
        gyro_events = events[..., self.acc_dim:]

        # Process each stream
        acc_tokens, acc_attn = self.acc_resampler(acc_events, timestamps, mask, return_attn)
        gyro_tokens, gyro_attn = self.gyro_resampler(gyro_events, timestamps, mask, return_attn)

        # Concatenate along embedding dimension
        tokens = torch.cat([acc_tokens, gyro_tokens], dim=-1)  # (B, L, embed_dim)

        # Fusion
        tokens = self.fusion(tokens)

        if return_attn:
            return tokens, (acc_attn, gyro_attn)
        return tokens, None


class DualStreamStudent(nn.Module):
    """
    Dual-stream student with separate resamplers for acc and gyro.

    Architecture:
    1. DualStreamResampler: Split acc/gyro, resample separately, fuse
    2. TransformerEncoder: temporal modeling on fused tokens
    3. SqueezeExcitation: channel attention
    4. TemporalAttentionPooling: aggregate to single vector
    5. Classifier: binary fall detection
    """

    def __init__(
        self,
        acc_dim: int = 3,
        gyro_dim: int = 3,
        embed_dim: int = 48,
        num_tokens: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        acc_ratio: float = 0.65,
        se_reduction: int = 4,
        time_mode: str = 'position',
    ):
        super().__init__()
        self.time_mode = time_mode

        self.resampler = DualStreamResampler(
            acc_dim=acc_dim,
            gyro_dim=gyro_dim,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            dropout=dropout * 0.2,
            acc_ratio=acc_ratio,
            time_mode=time_mode,
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
        return_attention: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            events: (B, N, acc_dim + gyro_dim)
            timestamps: (B, N)
            mask: (B, N)
            return_attention: If True, return attention weights for KD

        Returns:
            logits: (B, 1)
            features: (B, embed_dim)
            attention: (optional) Dict with 'pool': (B, L) attention weights
        """
        # Dual-stream resampling
        tokens, _ = self.resampler(events, timestamps, mask)

        # Transformer encoding
        tokens = self.transformer(tokens)

        # Channel attention
        tokens = self.se(tokens)

        # Temporal pooling
        if return_attention:
            features, pool_attn = self.pool(tokens, return_weights=True)
        else:
            features = self.pool(tokens)

        # Classification
        logits = self.classifier(features)

        if return_attention:
            return logits, features, {'pool': pool_attn}
        return logits, features
