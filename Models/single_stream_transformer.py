"""
Single-Stream IMU Transformer

Concatenates acc+gyro at input and processes through single transformer.
NO modality-specific projections - all 6 channels processed together.

Variants:
- SingleStreamTransformer: No SE, no TAP (Model A, C in ablation)
- SingleStreamTransformerSE: With SE + TAP (Model E in ablation)

Architecture:
    IMU [6ch] --> Conv1d --> Transformer Encoder --> Pooling --> Classifier

    Single projection for all channels (no modality separation)
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange
from typing import Optional, Tuple


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.SiLU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) tensor
        Returns:
            (B, C, T) tensor with channel attention applied
        """
        # Global average pooling
        scale = x.mean(dim=2)  # (B, C)
        scale = self.fc(scale).unsqueeze(2)  # (B, C, 1)
        return x * scale


class TemporalAttentionPooling(nn.Module):
    """Temporal attention pooling for sequence aggregation."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C) tensor
        Returns:
            pooled: (B, C) tensor
            weights: (B, T) attention weights
        """
        # Compute attention weights
        scores = self.attention(x).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1)  # (B, T)

        # Weighted sum
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, C)

        return pooled, weights


class SingleStreamTransformer(nn.Module):
    """
    Single-stream transformer for 6-channel IMU data.

    All channels [ax, ay, az, gx, gy, gz] are projected together through
    a single Conv1d layer, then processed by a transformer encoder.

    This is the baseline architecture WITHOUT modality-specific processing.

    Args:
        imu_frames: Sequence length (default 128)
        imu_channels: Number of input channels (default 6)
        num_heads: Number of attention heads (default 4)
        num_layers: Number of transformer layers (default 2)
        embed_dim: Embedding dimension (default 64)
        dropout: Dropout rate (default 0.5)
        use_se: Whether to use Squeeze-Excitation (default False)
        use_temporal_attention: Whether to use temporal attention pooling (default False)
    """

    def __init__(self,
                 imu_frames: int = 128,
                 imu_channels: int = 6,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 embed_dim: int = 64,
                 dropout: float = 0.5,
                 use_se: bool = False,
                 use_temporal_attention: bool = False,
                 use_pos_encoding: bool = True,
                 se_reduction: int = 4,
                 activation: str = 'silu',
                 norm_first: bool = True,
                 **kwargs):
        super().__init__()

        self.imu_frames = imu_frames
        self.imu_channels = imu_channels
        self.embed_dim = embed_dim
        self.use_se = use_se
        self.use_temporal_attention = use_temporal_attention
        self.use_pos_encoding = use_pos_encoding

        # Activation function
        act_fn = nn.SiLU() if activation.lower() == 'silu' else nn.ReLU()

        # Single input projection for ALL channels together
        # This is the key difference from dual-stream: no modality separation
        self.input_proj = nn.Sequential(
            nn.Conv1d(imu_channels, embed_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(embed_dim),
            act_fn,
            nn.Dropout(dropout * 0.3)
        )

        # Optional SE module after projection
        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)

        # Optional positional encoding
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, imu_frames, embed_dim) * 0.02)
        else:
            self.pos_encoding = None

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            norm_first=norm_first,
            batch_first=False
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling strategy
        if use_temporal_attention:
            self.temporal_pool = TemporalAttentionPooling(embed_dim)
        else:
            self.temporal_pool = None

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, 1)

        # Store attention weights for visualization
        self.last_attention_weights = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                acc_data: torch.Tensor,
                skl_data: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            acc_data: (B, T, C) IMU data where C=6 [ax,ay,az,gx,gy,gz]
            skl_data: Ignored (for API compatibility)

        Returns:
            logits: (B, 1) classification logits
            features: (B, embed_dim) features before classifier
        """
        # acc_data: (B, T, 6) = [ax, ay, az, gx, gy, gz]
        B, T, C = acc_data.shape

        # Rearrange for Conv1d: (B, C, T)
        x = rearrange(acc_data, 'b t c -> b c t')

        # Project all channels together
        x = self.input_proj(x)  # (B, embed_dim, T)

        # Optional SE attention
        if self.use_se:
            x = self.se(x)

        # Rearrange for transformer: (T, B, C)
        x = rearrange(x, 'b c t -> b t c')  # (B, T, embed_dim)

        # Add positional encoding (if enabled)
        if self.use_pos_encoding and self.pos_encoding is not None:
            if T <= self.imu_frames:
                x = x + self.pos_encoding[:, :T, :]
            else:
                # Handle longer sequences by interpolating positional encoding
                pos = torch.nn.functional.interpolate(
                    self.pos_encoding.permute(0, 2, 1),
                    size=T,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
                x = x + pos

        # Transformer expects (T, B, C)
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')  # Back to (B, T, C)

        # Pooling
        if self.use_temporal_attention and self.temporal_pool is not None:
            features, attn_weights = self.temporal_pool(x)
            self.last_attention_weights = attn_weights.detach()
        else:
            # Global average pooling
            features = x.mean(dim=1)  # (B, embed_dim)
            self.last_attention_weights = None

        # Classification
        features = self.dropout(features)
        logits = self.output(features)

        return logits, features

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return last computed attention weights for visualization."""
        return self.last_attention_weights


class SingleStreamTransformerSE(SingleStreamTransformer):
    """
    Single-stream transformer with SE + Temporal Attention Pooling.

    This is Model E in the ablation study:
    - Single-stream architecture (no modality separation)
    - Kalman smoothing (applied in data loader)
    - SE attention for channel recalibration
    - Temporal attention pooling for sequence aggregation
    """

    def __init__(self, **kwargs):
        # Force SE and temporal attention to be enabled
        kwargs['use_se'] = True
        kwargs['use_temporal_attention'] = True
        kwargs.setdefault('se_reduction', 4)
        super().__init__(**kwargs)


# For backward compatibility and easy import
SingleStream = SingleStreamTransformer
SingleStreamSE = SingleStreamTransformerSE


if __name__ == '__main__':
    """Test the models."""
    print("=" * 60)
    print("SINGLE STREAM TRANSFORMER TESTS")
    print("=" * 60)

    # Test parameters
    batch_size = 4
    seq_len = 128
    channels = 6

    # Create dummy input
    x = torch.randn(batch_size, seq_len, channels)

    # Test 1: Basic SingleStreamTransformer (Model A/C)
    print("\nTest 1: SingleStreamTransformer (no SE, no TAP)")
    model = SingleStreamTransformer(
        imu_frames=seq_len,
        imu_channels=channels,
        use_se=False,
        use_temporal_attention=False
    )
    logits, features = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert logits.shape == (batch_size, 1)
    assert features.shape == (batch_size, 64)
    print("  PASSED")

    # Test 2: SingleStreamTransformerSE (Model E)
    print("\nTest 2: SingleStreamTransformerSE (with SE + TAP)")
    model_se = SingleStreamTransformerSE(
        imu_frames=seq_len,
        imu_channels=channels,
        se_reduction=4
    )
    logits, features = model_se(x)
    attn = model_se.get_attention_weights()
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Attention weights shape: {attn.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_se.parameters()):,}")
    assert logits.shape == (batch_size, 1)
    assert features.shape == (batch_size, 64)
    assert attn.shape == (batch_size, seq_len)
    assert torch.allclose(attn.sum(dim=1), torch.ones(batch_size))  # Attention sums to 1
    print("  PASSED")

    # Test 3: Gradient flow
    print("\nTest 3: Gradient flow")
    model_se.zero_grad()
    logits, _ = model_se(x)
    loss = logits.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model_se.parameters() if p.requires_grad)
    print(f"  All parameters have gradients: {has_grad}")
    assert has_grad
    print("  PASSED")

    # Test 4: Different sequence lengths
    print("\nTest 4: Variable sequence length")
    for seq in [64, 128, 256]:
        x_var = torch.randn(2, seq, 6)
        logits, _ = model(x_var)
        print(f"  Seq length {seq}: output shape {logits.shape}")
        assert logits.shape == (2, 1)
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
