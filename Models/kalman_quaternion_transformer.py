"""
Kalman Quaternion Transformer for Fall Detection.

Input: 8ch [SMV, ax, ay, az, q0, q1, q2, q3]
    - SMV: Signal magnitude vector from accelerometer
    - ax, ay, az: Raw acceleration (3ch)
    - q0, q1, q2, q3: Quaternion orientation (4ch)

Output: Binary fall probability

Advantages over Euler angles:
    - No gimbal lock at any orientation
    - Continuous representation (no +/- pi discontinuities)
    - Natural for EKF/UKF state (no conversion needed)
    - Bounded output [-1, 1] for each component
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


class SqueezeExcitation(nn.Module):
    """Squeeze-Excitation channel attention module."""

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
        scale = x.mean(dim=1)
        scale = self.fc(scale).unsqueeze(1)
        return x * scale


class TemporalAttentionPooling(nn.Module):
    """Learnable temporal pooling via attention mechanism."""

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
    """Transformer encoder with guaranteed final layer normalization."""

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class KalmanQuaternionTransformer(nn.Module):
    """
    Dual-Stream Kalman Quaternion Transformer for fall detection.

    Uses quaternion output from Kalman filter (EKF/UKF) for gimbal-lock-free
    orientation representation.

    Architecture:
    ```
    Input: (B, 128, 8) [SMV, ax, ay, az, q0, q1, q2, q3]
           |
           +--- ACC Stream (4ch) ----+
           |    Conv1D(4->32)        |
           |    BatchNorm + SiLU     |
           |                         |
           +--- QUAT Stream (4ch) ---+
                Conv1D(4->32)        |
                BatchNorm + SiLU     |
                                     |
                +--------------------+
                | Concatenate (64ch)
                v
           LayerNorm
                |
           TransformerEncoder (2 layers, 4 heads)
                |
           Squeeze-Excitation
                |
           Temporal Attention Pooling
                |
           Dropout + Linear(64->1)
                |
           Output: (B, 1) logits
    ```

    Args:
        imu_frames: Sequence length (default: 128)
        imu_channels: Input channels (default: 8)
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
        acc_channels: Accelerometer channels including SMV (default: 4)
        quat_channels: Quaternion channels (default: 4)
    """

    def __init__(
        self,
        imu_frames: int = 128,
        imu_channels: int = 8,
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
        acc_ratio: float = 0.5,
        use_se: bool = True,
        use_tap: bool = True,
        use_pos_encoding: bool = False,
        acc_channels: int = 4,
        quat_channels: int = 4,
        **kwargs
    ):
        super().__init__()

        # Store configuration
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.use_se = use_se
        self.use_tap = use_tap
        self.use_pos_encoding = use_pos_encoding
        self.embed_dim = embed_dim

        # Channel configuration
        self.acc_channels = acc_channels  # [SMV, ax, ay, az] = 4
        self.quat_channels = quat_channels  # [q0, q1, q2, q3] = 4

        # Compute stream embedding dimensions (50/50 split - both streams are 4ch)
        acc_dim = int(embed_dim * acc_ratio)
        quat_dim = embed_dim - acc_dim

        # Accelerometer Stream Projection
        self.acc_proj = nn.Sequential(
            nn.Conv1d(self.acc_channels, acc_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(acc_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)
        )

        # Quaternion Stream Projection
        # Quaternions are bounded [-1, 1], smooth, less noisy than Euler
        self.quat_proj = nn.Sequential(
            nn.Conv1d(self.quat_channels, quat_dim, kernel_size=8, padding='same'),
            nn.BatchNorm1d(quat_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.2)  # Same dropout as acc (quaternions are stable)
        )

        # Fusion normalization
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Optional Positional Encoding
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, self.imu_frames, embed_dim) * 0.02
            )
        else:
            self.pos_encoding = None

        # Transformer Encoder
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

        # Squeeze-Excitation Channel Attention
        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = None

        # Temporal Pooling
        if use_tap:
            self.temporal_pool = TemporalAttentionPooling(embed_dim)
        else:
            self.temporal_pool = None

        # Classification Head
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

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
            acc_data: IMU tensor of shape (B, T, 8)
                      Channels: [SMV, ax, ay, az, q0, q1, q2, q3]
            skl_data: Unused (skeleton data for multimodal compatibility)
            **kwargs: Additional arguments (ignored)

        Returns:
            logits: Classification logits of shape (B, 1)
            features: Intermediate features of shape (B, T, embed_dim)
        """
        # Split into accelerometer and quaternion streams
        acc = acc_data[:, :, :self.acc_channels]  # (B, T, 4)
        quat = acc_data[:, :, self.acc_channels:self.acc_channels + self.quat_channels]  # (B, T, 4)

        # Reshape for Conv1d: (B, T, C) -> (B, C, T)
        acc = rearrange(acc, 'b t c -> b c t')
        quat = rearrange(quat, 'b t c -> b c t')

        # Stream projections
        acc_feat = self.acc_proj(acc)
        quat_feat = self.quat_proj(quat)

        # Concatenate streams and reshape
        x = torch.cat([acc_feat, quat_feat], dim=1)
        x = rearrange(x, 'b c t -> b t c')
        x = self.fusion_norm(x)

        # Optional positional encoding
        if self.use_pos_encoding and self.pos_encoding is not None:
            x = x + self.pos_encoding

        # Transformer encoder
        x = rearrange(x, 'b t c -> t b c')
        x = self.encoder(x)
        x = rearrange(x, 't b c -> b t c')

        # Squeeze-Excitation
        if self.se is not None:
            x = self.se(x)

        # Store features before pooling
        features = x

        # Temporal pooling
        if self.temporal_pool is not None:
            x, attn_weights = self.temporal_pool(x)
        else:
            x = x.mean(dim=1)

        # Classification
        x = self.dropout_layer(x)
        logits = self.output(x)

        return logits, features

    def get_attention_weights(self, acc_data: torch.Tensor) -> torch.Tensor:
        """Get temporal attention weights for visualization."""
        if self.temporal_pool is None:
            raise ValueError("Model was initialized with use_tap=False")

        with torch.no_grad():
            acc = acc_data[:, :, :self.acc_channels]
            quat = acc_data[:, :, self.acc_channels:self.acc_channels + self.quat_channels]

            acc = rearrange(acc, 'b t c -> b c t')
            quat = rearrange(quat, 'b t c -> b c t')

            acc_feat = self.acc_proj(acc)
            quat_feat = self.quat_proj(quat)

            x = torch.cat([acc_feat, quat_feat], dim=1)
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


if __name__ == "__main__":
    print("=" * 70)
    print("KalmanQuaternionTransformer Model Test")
    print("=" * 70)

    model = KalmanQuaternionTransformer()

    batch_size = 8
    seq_len = 128
    channels = 8

    x = torch.randn(batch_size, seq_len, channels)
    logits, features = model(x)

    print(f"\nModel: KalmanQuaternionTransformer")
    print(f"Input shape:    ({batch_size}, {seq_len}, {channels})")
    print(f"  - Channels: [SMV, ax, ay, az, q0, q1, q2, q3]")
    print(f"Output shape:   {logits.shape}")
    print(f"Features shape: {features.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")

    model.train()
    x = torch.randn(4, 128, 8, requires_grad=True)
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    grad_norm = x.grad.norm().item()
    print(f"\nGradient check: grad_norm={grad_norm:.6f} [{'OK' if grad_norm > 0 else 'FAIL'}]")

    model.eval()
    x = torch.randn(2, 128, 8)
    weights = model.get_attention_weights(x)
    print(f"\nAttention weights shape: {weights.shape}")
    print(f"Attention sum (should be 1.0): {weights.sum(dim=1)}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
