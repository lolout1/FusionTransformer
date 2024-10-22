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
    def __init__(self,
                mocap_frames = 128,
                num_joints = 32,
                acc_frames = 128,
                num_classes:int = 8, 
                num_heads = 2, 
                acc_coords = 4, 
                av = False,
                num_layer = 2, norm_first = True, 
                embed_dim= 8, activation = 'relu',
                **kwargs) :
        super().__init__()
        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]
        size = self.data_shape[1]
        #self.channel_embed_dim = embed_dim // 2
        self.input_proj = nn.Sequential(nn.Conv1d(size, embed_dim, kernel_size=3, stride=1, padding='same'),
                                         nn.Conv1d(embed_dim, embed_dim*2, kernel_size = 3, stride=1, padding='same'),
                                         nn.Conv1d(embed_dim*2, embed_dim, kernel_size=3, stride=1, padding='same'))


        self.encoder_layer = TransformerEncoderLayer(d_model = self.length,  activation = activation, 
                                                     dim_feedforward = 32, nhead = num_heads,dropout=0.5)
        
        self.encoder = TransformerEncoderWAttention(encoder_layer = self.encoder_layer, num_layers = num_layer, 
                                          norm=nn.LayerNorm(embed_dim))

        self.ln1 = nn.Linear(self.length, 32)
        # self.drop1 = nn.Dropout(p = 0.5)
        self.ln2 = nn.Linear(32, 16)
        self.drop2 = nn.Dropout(p = 0.5)
        self.output = Linear(16, num_classes)
        nn.init.normal_(self.output.weight, 0, math.sqrt(2. / num_classes))
    
    def forward(self, acc_data, skl_data):

        b, l, c = acc_data.shape
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_proj(x) # [ 8, 64, 3]
        x = rearrange(x,'b c l ->  c b l') #[8, 64, 3]
        x = self.encoder(x)
        x = rearrange(x, 'c b l -> b l c')


        x = F.avg_pool1d(x, kernel_size = x.shape[-1], stride = 1)
        x = rearrange(x, 'b c f -> b (c f)')
        # x= self.drop1(x)
        x = F.relu(self.ln1(x))
        # x = self.drop2(x)
        x = F.relu(self.ln2(x))
        x = self.output(x)
        return x

if __name__ == "__main__":
        data = torch.randn(size = (16,128,4))
        skl_data = torch.randn(size = (16,128,32,3))
        model = TransModel()
        output = model(data, skl_data)
