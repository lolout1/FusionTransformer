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
        self.attention_weights = []
        for layer in self.layers :
            output, attn = layer.self_attn(output, output, output, attn_mask = mask,
                                            key_padding_mask = src_key_padding_mask, need_weights = True)
            self.attention_weights.append(attn)
            output = layer(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
        return output


class TransModel(nn.Module):
    def __init__(self, data_shape:Dict[str, Tuple[int, int]] = {'inertial':(128, 3)},
                mocap_frames = 128,
                num_joints = 32,
                acc_frames = 128,
                num_classes:int = 8, 
                num_heads = 4, 
                acc_coords = 3, 
                av = False,
                adepth = 2, norm_first = True, 
                acc_embed= 8, activation = 'relu',
                **kwargs) :
        super().__init__()
        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]
        size = self.data_shape[1]
        self.av = av
        if av : 
            size = size + 3
        
        print(size)

        self.input_proj = nn.Linear(size, acc_embed)
        self.encoder_layer = TransformerEncoderLayer(d_model = acc_embed, activation = activation, 
                                                     dim_feedforward = 32, nhead = num_heads,dropout=0.5)
        
        self.encoder = TransformerEncoderWAttention(encoder_layer = self.encoder_layer, num_layers = adepth, 
                                          norm=nn.LayerNorm(acc_embed))

        pooled = self.length//2 + 1  
        self.ln1 = nn.Linear(pooled*acc_embed, 64)
        self.output = Linear(64, num_classes)
        nn.init.normal_(self.output.weight, 0, math.sqrt(2. / num_classes))
    
    def forward(self, acc_data, skl_data):

        b, l, c = acc_data.shape


        x = self.input_proj(acc_data) # [ 8, 64, 3]
        x = rearrange(x,'b l c ->  l b c') #[8, 64, 3]
        x = self.encoder(x)
        x = rearrange(x, 'c b l -> b l c')

        # x = self.feature_transform(x)
        #x = rearrange(x, 'b l c -> b c l')
        # for i, l in enumerate(self.reduciton):
        #     x = l(x)

        x = F.max_pool1d(x, kernel_size = x.shape[-1]//2, stride = 1)
        x = rearrange(x, 'b c f -> b (c f)')
        x = self.ln1(x)        
        
        x = self.output(x)
        return x

if __name__ == "__main__":
        data = torch.randn(size = (16,128,3))
        skl_data = torch.randn(size = (16,128,32,3))
        model = TransModel()
        output = model(data, skl_data)
