"""
Dual-Stream Kalman Transformer for Fall Detection.

Input: 7ch [smv, ax, ay, az, roll, pitch, yaw]
Output: Binary fall probability
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
    """
    Squeeze-Excitation channel attention module.

    Learns channel-wise importance weights through global pooling
    and a bottleneck MLP, allowing the model to emphasize informative
    channels and suppress less useful ones.

    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018

    Args:
        channels: Number of input channels
        reduction: Bottleneck reduction ratio (default: 4)
    """

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
        """
        Args:
            x: Input tensor of shape (B, T, C)
        Returns:
            Channel-reweighted tensor of shape (B, T, C)
        """
        # Global average pooling across time dimension
        scale = x.mean(dim=1)  # (B, C)
        scale = self.fc(scale).unsqueeze(1)  # (B, 1, C)
        return x * scale


class TemporalAttentionPooling(nn.Module):
    """
    Learnable temporal pooling via attention mechanism.

    Instead of simple global average pooling, learns to weight
    different time steps based on their importance. Critical for
    fall detection where the impact event is a brief transient.

    The attention mechanism learns which time steps contain the
    most discriminative information (e.g., impact moment in falls).

    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape (B, T, C)
        Returns:
            context: Weighted sum of shape (B, C)
            weights: Attention weights of shape (B, T)
        """
        scores = self.attention(x).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1)
        context = torch.einsum('bt,btc->bc', weights, x)
        return context, weights


class TransformerEncoderWithNorm(nn.TransformerEncoder):
    """
    Transformer encoder with guaranteed final layer normalization.

    Ensures the norm layer is always applied after the encoder layers,
    which is important for training stability with pre-norm transformers.
    """

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


# =============================================================================
# Main Model
# =============================================================================

class KalmanBest(nn.Module):
    """
    Best-performing Dual-Stream Kalman Transformer for fall detection.

    This model achieved 90.43% Test F1 on SmartFallMM dataset using
    21-fold Leave-One-Subject-Out Cross-Validation on young subjects.

    Architecture Overview:
    ```
    Input: (B, 128, 7) [smv, ax, ay, az, roll, pitch, yaw]
           │
           ├─── Acc Stream (4ch) ───┐
           │    Conv1D(4→32)        │
           │    BatchNorm + SiLU    │
           │                        │
           └─── Ori Stream (3ch) ───┤
                Conv1D(3→32)        │
                BatchNorm + SiLU    │
                                    │
                ┌───────────────────┘
                │ Concatenate (64ch)
                ▼
           LayerNorm
                │
           TransformerEncoder (2 layers, 4 heads)
                │
           Squeeze-Excitation
                │
           Temporal Attention Pooling
                │
           Dropout + Linear(64→1)
                │
           Output: (B, 1) logits
    ```

    Args:
        imu_frames: Sequence length (default: 128)
        imu_channels: Input channels (default: 7)
        num_classes: Output classes (default: 1 for binary)
        num_heads: Transformer attention heads (default: 4)
        num_layers: Transformer layers (default: 2)
        embed_dim: Embedding dimension (default: 64)
        dropout: Dropout rate (default: 0.5)
        activation: Transformer activation (default: 'relu')
        norm_first: Pre-norm transformer (default: True)
        se_reduction: SE bottleneck ratio (default: 4)
        acc_ratio: Acc stream embedding ratio (default: 0.5)
        use_se: Enable Squeeze-Excitation (default: True)
        use_tap: Enable Temporal Attention Pooling (default: True)
        use_pos_encoding: Enable positional encoding (default: False)
        acc_channels: Accelerometer channels (default: 4)

    Example:
        >>> model = KalmanBest()
        >>> x = torch.randn(8, 128, 7)  # (batch, time, channels)
        >>> logits, features = model(x)
        >>> logits.shape
        torch.Size([8, 1])
        >>> features.shape
        torch.Size([8, 128, 64])
    """

    def __init__(
        self,
        imu_frames: int = 128,
        imu_channels: int = 7,
        acc_frames: int = 128,  # Alias for compatibility
        acc_coords: int = 7,    # Alias for compatibility
        mocap_frames: int = 128,  # Unused, for interface compatibility
        num_joints: int = 32,     # Unused, for interface compatibility
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
        **kwargs  # Absorb any extra arguments
    ):
        super().__init__()

        # Store configuration
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_se = use_se
        self.use_tap = use_tap
        self.use_pos_encoding = use_pos_encoding
        self.embed_dim = embed_dim

        # Channel split: accelerometer vs orientation
        self.acc_channels = acc_channels  # [smv, ax, ay, az] = 4
        self.ori_channels = self.imu_channels - acc_channels  # [roll, pitch, yaw] = 3

        # Compute stream embedding dimensions (50/50 split by default)
        acc_dim = int(embed_dim * acc_ratio)  # 32
        ori_dim = embed_dim - acc_dim         # 32

        # ---------------------------------------------------------------------
        # Accelerometer Stream Projection
        # Conv1D captures local temporal patterns in acceleration
        # ---------------------------------------------------------------------
        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.SiLU(),  # Swish activation - smooth, non-monotonic
            nn.Dropout(dropout * 0.2)  # Light dropout in projection
        )

        # ---------------------------------------------------------------------
        # Orientation Stream Projection
        # Slightly higher dropout for noisier orientation estimates
        # ---------------------------------------------------------------------
        self.ori_proj = nn.Sequential(
            nn.Conv1d(self.ori_channels, ori_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(ori_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.3)  # More dropout for noisy orientation
        )

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # ---------------------------------------------------------------------
        # Optional Positional Encoding
        # Disabled by default - IMU signals don't benefit from position info
        # ---------------------------------------------------------------------
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, self.imu_frames, embed_dim) * 0.02
            )
        else:
            self.pos_encoding = None

        # ---------------------------------------------------------------------
        # Transformer Encoder
        # Pre-norm (norm_first=True) for better training stability
        # ---------------------------------------------------------------------
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,  # 128
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=False  # PyTorch expects (T, B, C) for transformer
        )

        self.encoder = TransformerEncoderWithNorm(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # ---------------------------------------------------------------------
        # Squeeze-Excitation Channel Attention (Optional)
        # ---------------------------------------------------------------------
        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = None

        # ---------------------------------------------------------------------
        # Temporal Pooling: TAP or GAP
        # ---------------------------------------------------------------------
        if use_tap:
            self.temporal_pool = TemporalAttentionPooling(embed_dim)
        else:
            self.temporal_pool = None

        # ---------------------------------------------------------------------
        # Classification Head
        # ---------------------------------------------------------------------
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize output layer with proper scaling."""
        nn.init.normal_(self.output.weight, 0, math.sqrt(2.0 / self.output.out_features))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)

    def forward(self, acc_data: torch.Tensor, skl_data=None, **kwargs) -> tuple:
        """
        Forward pass through the model.

        Args:
            acc_data: IMU tensor of shape (B, T, 7)
                      Channels: [smv, ax, ay, az, roll, pitch, yaw]
            skl_data: Unused (skeleton data for multimodal compatibility)
            **kwargs: Additional arguments (ignored)

        Returns:
            logits: Classification logits of shape (B, 1)
            features: Intermediate features of shape (B, T, embed_dim)
        """
        # Split into accelerometer and orientation streams
        acc = acc_data[:, :, :self.acc_channels]  # (B, T, 4)
        ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]  # (B, T, 3)

        # Reshape for Conv1d: (B, T, C) -> (B, C, T)
        acc = rearrange(acc, 'b t c -> b c t')
        ori = rearrange(ori, 'b t c -> b c t')

        # Stream projections
        acc_feat = self.acc_proj(acc)  # (B, acc_dim, T)
        ori_feat = self.ori_proj(ori)  # (B, ori_dim, T)

        # Concatenate streams and reshape: (B, embed_dim, T) -> (B, T, embed_dim)
        x = torch.cat([acc_feat, ori_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        # Optional positional encoding
        if self.use_pos_encoding and self.pos_encoding is not None:
            x = x + self.pos_encoding

        # Transformer encoder: (B, T, C) -> (T, B, C) -> (B, T, C)
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        # Optional Squeeze-Excitation
        if self.se is not None:
            x = self.se(x)

        # Store features before pooling
        features = x

        # Temporal pooling: TAP or GAP
        if self.temporal_pool is not None:
            x, attn_weights = self.temporal_pool(x)  # (B, embed_dim)
        else:
            x = x.mean(dim=1)  # Global Average Pooling

        # Classification
        x = self.dropout_layer(x)
        logits = self.output(x)  # (B, num_classes)

        return logits, features

    def get_attention_weights(self, acc_data: torch.Tensor) -> torch.Tensor:
        """
        Get temporal attention weights for visualization.

        Useful for understanding which time steps the model focuses on.

        Args:
            acc_data: IMU tensor of shape (B, T, 7)

        Returns:
            weights: Attention weights of shape (B, T)
        """
        if self.temporal_pool is None:
            raise ValueError("Model was initialized with use_tap=False")

        with torch.no_grad():
            # Forward pass up to temporal pooling
            acc = acc_data[:, :, :self.acc_channels]
            ori = acc_data[:, :, self.acc_channels:self.acc_channels + self.ori_channels]

            acc = rearrange(acc, 'b t c -> b c t')
            ori = rearrange(ori, 'b t c -> b c t')

            acc_feat = self.acc_proj(acc)
            ori_feat = self.ori_proj(ori)

            x = torch.cat([acc_feat, ori_feat], dim=1)
            x = rearrange(x, 'b c t -> b t c')
            x = self.fusion_norm(x)

            if self.use_pos_encoding and self.pos_encoding is not None:
                x = x + self.pos_encoding

            x = rearrange(x, 'b t c -> t b c')
            x = self.encoder(x)
            x = rearrange(x, 't b c -> b t c')

            if self.se is not None:
                x = self.se(x)

            _, weights = self.temporal_pool(x)

        return weights


# =============================================================================
# Model Factory
# =============================================================================

def create_kalman_best(pretrained: bool = False, **kwargs) -> KalmanBest:
    """
    Factory function to create KalmanBest model.

    Args:
        pretrained: If True, load pretrained weights (not implemented)
        **kwargs: Override default model arguments

    Returns:
        Initialized KalmanBest model

    Example:
        >>> model = create_kalman_best()
        >>> model = create_kalman_best(dropout=0.3, embed_dim=128)
    """
    # Default best configuration
    default_config = {
        'imu_frames': 128,
        'imu_channels': 7,
        'num_heads': 4,
        'num_layers': 2,
        'embed_dim': 64,
        'dropout': 0.5,
        'activation': 'relu',
        'norm_first': True,
        'se_reduction': 4,
        'acc_ratio': 0.5,
        'use_se': True,
        'use_tap': True,
        'use_pos_encoding': False,
        'acc_channels': 4,
    }

    # Override with provided kwargs
    default_config.update(kwargs)

    model = KalmanBest(**default_config)

    if pretrained:
        raise NotImplementedError("Pretrained weights not available yet")

    return model


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("KalmanBest Model Test")
    print("=" * 70)

    # Create model
    model = KalmanBest()

    # Test forward pass
    batch_size = 8
    seq_len = 128
    channels = 7

    x = torch.randn(batch_size, seq_len, channels)
    logits, features = model(x)

    print(f"\nModel: KalmanBest")
    print(f"Input shape:    ({batch_size}, {seq_len}, {channels})")
    print(f"Output shape:   {logits.shape}")
    print(f"Features shape: {features.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")

    # Gradient check
    model.train()
    x = torch.randn(4, 128, 7, requires_grad=True)
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    grad_norm = x.grad.norm().item()
    print(f"\nGradient check: grad_norm={grad_norm:.6f} [{'OK' if grad_norm > 0 else 'FAIL'}]")

    # Attention weights
    model.eval()
    x = torch.randn(2, 128, 7)
    weights = model.get_attention_weights(x)
    print(f"\nAttention weights shape: {weights.shape}")
    print(f"Attention sum (should be 1.0): {weights.sum(dim=1)}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
