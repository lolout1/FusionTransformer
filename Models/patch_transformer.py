"""
PatchTransformer: State-of-the-Art Patch-Based Transformer for IMU Time Series Classification.

Based on PatchTST (ICLR 2023) and recent HAR-specific adaptations (2024).
Key innovations:
    1. Patch embedding - segments time series into patches as tokens (not individual timesteps)
    2. CNN patch encoder - local feature extraction before transformer
    3. SE attention - channel recalibration (proven effective in prior ablations)
    4. Configurable patch sizes for ablation study

References:
    - PatchTST: "A Time Series is Worth 64 Words" (ICLR 2023)
    - Hi-WaveTST: Hybrid High-Frequency Wavelet-Transformer (2024)
    - PTN: Patch-Transformer Network for Fall Detection (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class SqueezeExcitation(nn.Module):
    """Channel attention module for feature recalibration."""

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
        """x: (B, T, C) or (B, N, C) where N is num_patches"""
        scale = x.mean(dim=1)  # Global average pooling over time/patches
        scale = self.fc(scale).unsqueeze(1)
        return x * scale


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that segments time series into patches.

    Uses 1D convolution with stride=patch_size for efficient patching,
    followed by layer normalization.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # CNN-based patch embedding (more expressive than linear projection)
        # First conv extracts local features, second projects to embed_dim
        self.patch_encoder = nn.Sequential(
            # Local feature extraction within each patch
            nn.Conv1d(in_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            # Patch projection with stride
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) - batch, time, channels
        Returns:
            patches: (B, N, D) - batch, num_patches, embed_dim
        """
        # Reshape for conv1d: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        # Apply patch encoding: (B, C, T) -> (B, D, N)
        x = self.patch_encoder(x)

        # Reshape back: (B, D, N) -> (B, N, D)
        x = x.transpose(1, 2)

        return self.norm(x)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patch positions."""

    def __init__(self, max_patches: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D)"""
        B, N, D = x.shape
        x = x + self.pos_embed[:, :N, :]
        return self.dropout(x)


class PatchTransformerEncoder(nn.Module):
    """
    Transformer encoder for processing patch embeddings.

    Uses pre-norm architecture (norm_first=True) which is more stable
    and commonly used in recent transformer variants.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        if ff_dim is None:
            ff_dim = embed_dim * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            norm_first=True,  # Pre-norm for stability
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) -> (B, N, D)"""
        return self.encoder(x)


class PatchTransformer(nn.Module):
    """
    Patch-based Transformer for IMU Time Series Classification.

    Architecture:
        Input (B, T, C) -> Patch Embedding (B, N, D) -> Pos Encoding ->
        Transformer Encoder -> SE Attention -> Global Pooling -> Classification

    Key features:
        - Patch-based tokenization (configurable patch_size)
        - CNN-based patch encoder for local feature extraction
        - SE attention for channel recalibration
        - Mean pooling for classification (robust to sequence length)

    Args:
        imu_frames: Input sequence length (default: 128)
        imu_channels: Number of input channels (default: 7 for Kalman)
        patch_size: Size of each patch (default: 16)
        embed_dim: Transformer embedding dimension (default: 64)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
        dropout: Dropout rate (default: 0.3)
        num_classes: Output classes (default: 1 for binary)
        use_se: Whether to use SE attention (default: True)
        use_pos_encoding: Whether to use positional encoding (default: True)
        activation: Activation function (default: 'gelu')
    """

    def __init__(
        self,
        imu_frames: int = 128,
        imu_channels: int = 7,
        acc_frames: int = 128,
        acc_coords: int = 7,
        mocap_frames: int = 128,
        num_joints: int = 32,
        patch_size: int = 16,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = None,
        dropout: float = 0.3,
        num_classes: int = 1,
        use_se: bool = True,
        use_pos_encoding: bool = True,
        se_reduction: int = 4,
        activation: str = 'gelu',
        **kwargs  # Absorb unused args for compatibility
    ):
        super().__init__()

        # Handle legacy parameter names
        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_se = use_se
        self.use_pos_encoding = use_pos_encoding

        # Calculate number of patches
        self.num_patches = self.imu_frames // patch_size

        # Patch embedding with CNN encoder
        self.patch_embed = PatchEmbedding(
            in_channels=self.imu_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            dropout=dropout
        )

        # Positional encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(
                max_patches=self.num_patches + 1,  # +1 for potential CLS token
                embed_dim=embed_dim,
                dropout=dropout
            )
        else:
            self.pos_encoding = nn.Identity()

        # Transformer encoder
        self.transformer = PatchTransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation
        )

        # SE attention (optional)
        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, skeleton: torch.Tensor = None) -> tuple:
        """
        Forward pass.

        Args:
            x: IMU data (B, T, C) - batch, time, channels
            skeleton: Unused, for API compatibility

        Returns:
            logits: (B, num_classes)
            features: (B, embed_dim) - for feature extraction/visualization
        """
        # Patch embedding: (B, T, C) -> (B, N, D)
        x = self.patch_embed(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # SE attention
        x = self.se(x)

        # Global mean pooling: (B, N, D) -> (B, D)
        features = x.mean(dim=1)

        # Classification
        logits = self.classifier(features)

        return logits, features


class PatchTransformerDualStream(nn.Module):
    """
    Dual-stream Patch Transformer for Kalman-fused IMU data.

    Processes accelerometer (4ch: smv, ax, ay, az) and orientation (3ch: roll, pitch, yaw)
    through separate patch encoders, then fuses with learned gating.

    This architecture is specifically designed for Kalman 7-channel input.
    """

    def __init__(
        self,
        imu_frames: int = 128,
        imu_channels: int = 7,
        acc_frames: int = 128,
        acc_coords: int = 7,
        mocap_frames: int = 128,
        num_joints: int = 32,
        patch_size: int = 16,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
        use_se: bool = True,
        use_pos_encoding: bool = True,
        se_reduction: int = 4,
        acc_ratio: float = 0.65,
        activation: str = 'gelu',
        **kwargs
    ):
        super().__init__()

        self.imu_frames = imu_frames if imu_frames else acc_frames
        self.imu_channels = imu_channels if imu_channels else acc_coords
        self.patch_size = patch_size
        self.num_patches = self.imu_frames // patch_size

        # Channel split for Kalman data: [smv, ax, ay, az, roll, pitch, yaw]
        self.acc_channels = 4  # smv + acc_xyz
        self.ori_channels = 3  # roll, pitch, yaw

        # Dimension split based on acc_ratio
        acc_dim = int(embed_dim * acc_ratio)
        ori_dim = embed_dim - acc_dim

        # Separate patch embeddings for each stream
        self.acc_patch_embed = PatchEmbedding(
            in_channels=self.acc_channels,
            embed_dim=acc_dim,
            patch_size=patch_size,
            dropout=dropout
        )

        self.ori_patch_embed = PatchEmbedding(
            in_channels=self.ori_channels,
            embed_dim=ori_dim,
            patch_size=patch_size,
            dropout=dropout
        )

        # Fusion layer norm
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Shared positional encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(
                max_patches=self.num_patches + 1,
                embed_dim=embed_dim,
                dropout=dropout
            )
        else:
            self.pos_encoding = nn.Identity()

        # Transformer encoder
        self.transformer = PatchTransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )

        # SE attention
        if use_se:
            self.se = SqueezeExcitation(embed_dim, reduction=se_reduction)
        else:
            self.se = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, skeleton: torch.Tensor = None) -> tuple:
        """
        Forward pass for Kalman 7-channel input.

        Args:
            x: (B, T, 7) - [smv, ax, ay, az, roll, pitch, yaw]
        """
        # Split channels
        acc_data = x[:, :, :self.acc_channels]  # (B, T, 4)
        ori_data = x[:, :, self.acc_channels:]  # (B, T, 3)

        # Patch embedding for each stream
        acc_patches = self.acc_patch_embed(acc_data)  # (B, N, acc_dim)
        ori_patches = self.ori_patch_embed(ori_data)  # (B, N, ori_dim)

        # Concatenate streams
        x = torch.cat([acc_patches, ori_patches], dim=-1)  # (B, N, embed_dim)
        x = self.fusion_norm(x)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)

        # SE attention
        x = self.se(x)

        # Global pooling and classification
        features = x.mean(dim=1)
        logits = self.classifier(features)

        return logits, features
