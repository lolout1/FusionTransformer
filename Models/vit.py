import torch
from torch import nn
import math
import torch.nn.functional as F
class AccelerometerTransformer(nn.Module):
    def __init__(self,acc_frames=150, mocap_frames = 128,num_joints=29, in_chans=3, acc_coords=3, acc_embed=32, acc_features=18, adepth=4, num_heads=4, mlp_ratio=2., qkv_bias=True,
                 qk_scale=None, op_type='cls', embed_type='lin', fuse_acc_features=False,has_features = False,
                 drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2,  norm_layer=None, num_classes=6, spatial_embed = 32):
        super(AccelerometerTransformer, self).__init__()
        self.embed_dim = acc_embed
        self.num_heads = num_heads
        self.num_layers = adepth
        
        # Input Embedding
        self.input_embedding = nn.Linear(in_chans, self.embed_dim)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(self.embed_dim)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(self.embed_dim, num_heads, dim_feedforward=128, dropout=drop_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_layers)
        
        # Output Layer
        self.output_layer = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, x, skl_data):
        # Input Embedding
        x = self.input_embedding(x)
        
        # Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        encoder = self.transformer_encoder(x)
        # Output Layer
        x = self.output_layer(encoder[:, 0])
        
        return encoder, x, F.log_softmax(x, dim = 1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]