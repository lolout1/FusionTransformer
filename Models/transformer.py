import torch 
from torch import nn
from typing import Dict, Tuple
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer, ModuleList
import torch.nn.functional as F
from einops import rearrange
import itertools
import numpy as np
#from util.graph import Graph
import math


class TransformerEncoderWAttention(nn.TransformerEncoder):
    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = src
        #self.attention_weights = []
        for layer in self.layers :
            output, attn = layer.self_attn(output, output, output, attn_mask = mask,
                                            key_padding_mask = src_key_padding_mask, need_weights = True)
            #self.attention_weights.append(attn)
            output = layer(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
        return output


class TransModel(nn.Module):
    """
    Accelerometer-only Transformer for Activity Recognition

    Supports flexible channel configurations:
      - 3 channels (default): ax, ay, az (accelerometer only)
      - 4 channels (with SMV): ax, ay, az, smv

    This is the baseline/standard model for comparison with IMU models.

    Args:
        acc_frames: Number of time steps (default: 128)
        acc_coords: Number of input channels (default: 3 for ax,ay,az; use 4 with SMV)
        num_classes: Number of output classes (default: 2 for smartfallmm)
        num_heads: Number of attention heads (default: 4 per config)
        num_layer: Number of transformer layers (default: 2 per config)
        embed_dim: Embedding dimension (default: 64 per config)
        dropout: Dropout rate (default: 0.5 for regularization)
        activation: Activation function (default: 'relu')
        norm_first: Whether to apply normalization first (default: True)
    """
    def __init__(self,
                mocap_frames = 128,
                num_joints = 32,
                acc_frames = 128,
                num_classes:int = 2,  # Default from config/smartfallmm/transformer.yaml
                num_heads = 4,         # Standard from config
                acc_coords = 3,        # 3 channels default: ax, ay, az (use 4 for SMV)
                av = False,
                num_layer = 2,
                norm_first = True,
                embed_dim = 64,        # Standard from config
                dropout = 0.5,
                activation = 'relu',
                **kwargs) :
        super().__init__()

        ##### uncomment if want to test embedding network ##########
        # self.ts2vec_model = TSEncoder(input_dims = acc_coords, output_dims= 64)
        # self.embeding_model = torch.optim.swa_utils.AveragedModel(self.ts2vec_model)
        # self.embeding_model.load_state_dict(torch.load('/home/bgu9/LightHART/Models/model.pkl'))
        # self.freeze_pretrained()

        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]
        self.input_channels = acc_coords  # Use parameter, not hardcoded

        # Input projection with standardized dropout
        # Supports 3 channels (ax, ay, az) or 4 channels (ax, ay, az, smv)
        self.input_proj = nn.Sequential(
            nn.Conv1d(acc_coords, embed_dim, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout * 0.5)  # Light dropout on input
        )

        # Transformer encoder with standardized dropout
        self.encoder_layer = TransformerEncoderLayer(
            d_model = embed_dim,
            activation = activation,
            dim_feedforward = embed_dim*2,
            nhead = num_heads,
            dropout = dropout,
            norm_first = norm_first,
            batch_first = False
        )

        self.encoder = TransformerEncoderWAttention(
            encoder_layer = self.encoder_layer,
            num_layers = num_layer,
            norm = nn.LayerNorm(embed_dim)
        )

        # Output layers with dropout
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes)

        # Initialize output layer with small weights
        nn.init.normal_(self.output.weight, 0, math.sqrt(2. / num_classes))
        if self.output.bias is not None:
            nn.init.constant_(self.output.bias, 0)
        # self.ln1 = nn.Conv1d(64,32, kernel_size=3, stride=1, padding = 1)
        # self.bn1 = nn.BatchNorm1d(32)
        # self.drop1 = nn.Dropout(p = 0.5)
        # self.ln2 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm1d(16)
        # self.drop2 = nn.Dropout(p = 0.5)
        
        # self.output = Linear(16, num_classes)
        #nn.init.normal_(self.output.weight, 0, math.sqrt(2. / num_classes))
    
    def freeze_pretrained(self):
         for params in self.embeding_model.parameters(): 
              params.requires_grad = False
    def unfreeze_pretrained(self, num_epochs): 
        layers = list(self.embeding_model.children())
        unfrozen_layers = layers[-200//num_epochs:]
        #  for params in self.embeding_model.parameters():
        #       params.requires_grad = True
        for layer in unfrozen_layers: 
             for param in layer.parameters():
                  param.requires_grad = True

    def forward(self, acc_data, skl_data=None, **kwargs):
        """
        Forward pass

        Args:
            acc_data: Accelerometer data tensor of shape (batch, time, channels)
                      Expected: (B, 128, 4) where channels = [ax, ay, az, smv]
            skl_data: Skeleton data (not used, for compatibility)

        Returns:
            logits: Output predictions (B, num_classes)
            features: Intermediate features (B, time, embed_dim) for distillation
        """
        epochs = kwargs.get('epoch')
        # if isinstance(epochs,int) and ( epochs+1) % 25 == 0:
        #       self.unfreeze_pretrained(epochs)
        # b, l, c = acc_data.shape
        # x = rearrange(acc_data, 'b l c -> b c l')
        # x = self.embeding_model(acc_data)


        # Input shape: (batch, time, channels)
        # Rearrange to (batch, channels, time) for Conv1d
        x = rearrange(acc_data, 'b l c -> b c l')

        # Project to embedding space: (B, embed_dim, time)
        x = self.input_proj(x)

        # Rearrange for transformer: (time, batch, embed_dim)
        x = rearrange(x, 'b c l -> l b c')

        # Transformer encoding
        x = self.encoder(x)

        # Rearrange back: (batch, time, embed_dim)
        x = rearrange(x, 'l b c -> b l c')

        # Normalize features
        x = self.temporal_norm(x)

        # Store features for knowledge distillation
        features = x

        # Global average pooling over time dimension
        # (batch, time, embed_dim) -> (batch, embed_dim)
        x = rearrange(x, 'b f c -> b c f')
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride=1)
        x = rearrange(x, 'b c f -> b (c f)')

        # Apply dropout before final layer
        x = self.dropout(x)

        # Final classification
        logits = self.output(x)

        return logits, features

if __name__ == "__main__":
    batch_size = 16
    seq_len = 128

    # Test with 3 channels (ax, ay, az) - default
    print("=" * 50)
    print("Test 1: 3 channels (acc only, no SMV)")
    print("=" * 50)
    data_3ch = torch.randn(size=(batch_size, seq_len, 3))
    skl_data = torch.randn(size=(batch_size, seq_len, 32, 3))

    model_3ch = TransModel(acc_coords=3)
    logits, features = model_3ch(data_3ch, skl_data, epoch=20)
    print(f"Input shape: {data_3ch.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_3ch.parameters()):,}")

    # Test with 4 channels (ax, ay, az, smv) - with SMV
    print("\n" + "=" * 50)
    print("Test 2: 4 channels (acc + SMV)")
    print("=" * 50)
    data_4ch = torch.randn(size=(batch_size, seq_len, 4))

    model_4ch = TransModel(acc_coords=4)
    logits, features = model_4ch(data_4ch, skl_data, epoch=20)
    print(f"Input shape: {data_4ch.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_4ch.parameters()):,}")
