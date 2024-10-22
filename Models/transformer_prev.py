import torch 
from torch import nn
from typing import Dict, Tuple
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer, ModuleList
import torch.nn.functional as F
from einops import rearrange
#from util.graph import Graph
import math

class TransModel(nn.Module):
    def __init__(self,
                acc_frames = 128,
                acc_coords = 6, 
                num_classes:int = 9, 
                num_heads = 4, 
                num_layers = 2, norm_first = True, 
                acc_embed= 256, activation = 'relu',
                **kwargs) :
        super().__init__()
        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]
        size = self.data_shape[1]
        self.input_proj = nn.Sequential(nn.Conv1d(self.length, acc_embed, 1), nn.GELU(),
                                        nn.Conv1d(acc_embed, acc_embed, 1), nn.GELU(),
                                        nn.Conv1d(acc_embed, acc_embed, 1), nn.GELU(),
                                        nn.Conv1d(acc_embed, acc_embed, 1), nn.GELU())
        self.transform_layer = Linear(self.data_shape[-2], acc_embed)
        self.encoder_layer = TransformerEncoderLayer(d_model = acc_embed, activation = activation, 
                                                     dim_feedforward = 256, nhead = num_heads,dropout=0.5)
        
        self.encoder = TransformerEncoder(encoder_layer = self.encoder_layer, num_layers = num_layers, 
                                          norm=nn.LayerNorm(256))
        self.reduciton =  ModuleList([Linear(acc_embed, int(acc_embed/2)),
                                   Linear(int(acc_embed/2), acc_embed//4)])
        self.feature_transform = nn.Linear(3, 16)
        pooled = acc_embed//8 + 1 
        self.ln1 = nn.Linear(pooled*16, 64)
        self.output = Linear(64, num_classes)
        nn.init.normal_(self.output.weight, 0, math.sqrt(2. / num_classes))
                                                    
    
    def forward(self, acc_data, skl_data):
        b, l, c = acc_data.shape

        x = self.input_proj(acc_data) # [ 8, 64, 3]
        x = self.feature_transform(x)
        x = rearrange(x,'b l c -> c b l') #[8, 64, 3]
        x = self.encoder(x)
        x = rearrange(x, 'c b l -> b c l')
        for i, l in enumerate(self.reduciton):
            x = l(x)
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
