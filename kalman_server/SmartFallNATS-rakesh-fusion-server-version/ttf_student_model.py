import torch 
from torch import nn
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer, ModuleList
import torch.nn.functional as F
from einops import rearrange


class TransformerEncoderWAttention(nn.TransformerEncoder):
    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = src
        for layer in self.layers :
            output, attn = layer.self_attn(output, output, output, attn_mask = mask,
                                            key_padding_mask = src_key_padding_mask, need_weights = True)
            output = layer(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
        return output


class TransModel(nn.Module):
    def __init__(self,
                mocap_frames = 128,
                num_joints = 32,
                acc_frames = 128,
                num_classes:int = 8, 
                num_heads = 2, 
                acc_coords = 3, 
                av = False,
                num_layer = 2, norm_first = True, 
                embed_dim= 16, activation = 'relu',
                **kwargs) :
        super().__init__()

        ##### uncomment if want to test embedding network ##########
        # self.ts2vec_model = TSEncoder(input_dims = acc_coords, output_dims= 64)
        # self.embeding_model = torch.optim.swa_utils.AveragedModel(self.ts2vec_model)
        # self.embeding_model.load_state_dict(torch.load('/home/bgu9/LightHART/Models/model.pkl'))
        # self.freeze_pretrained()

        self.data_shape = (acc_frames, acc_coords)
        self.length = self.data_shape[0]
        size = self.data_shape[1]
        self.input_proj = nn.Sequential(nn.Conv1d(4, embed_dim, kernel_size=8, stride=1, padding='same'), 
                                        nn.BatchNorm1d(embed_dim))



        #dropout = 0.3 best
        self.encoder_layer = TransformerEncoderLayer(d_model = embed_dim,  activation = activation, 
                                                     dim_feedforward =embed_dim*2, nhead = num_heads,dropout=0.5)
        
        self.encoder = TransformerEncoderWAttention(encoder_layer = self.encoder_layer, num_layers = num_layer, 
                                          norm=nn.LayerNorm(embed_dim))

        self.output = nn.Linear(embed_dim, num_classes)
        self.temporal_norm = nn.LayerNorm(embed_dim)
    
    def freeze_pretrained(self):
         for params in self.embeding_model.parameters(): 
              params.requires_grad = False
    def unfreeze_pretrained(self, num_epochs): 
        layers = list(self.embeding_model.children())
        unfrozen_layers = layers[-200//num_epochs:]

        for layer in unfrozen_layers: 
             for param in layer.parameters():
                  param.requires_grad = True

    def forward(self, acc_data, **kwargs):
        epochs = kwargs.get('epoch')
       
        
        ###### for transformer ############
        x = rearrange(acc_data, 'b l c -> b c l')
        x = self.input_proj(x) # [ 8, 64, 3]
        x = rearrange(x, 'b c l -> l b c ')
        x = self.encoder(x)
        x = rearrange(x ,'l b c -> b l c')

        x = self.temporal_norm(x)

        feature = x

        x = rearrange(x, 'b f c -> b c f')
        x = F.avg_pool1d(x, kernel_size=x.shape[-1], stride = 1)
        x = rearrange(x, 'b c f -> b (c f)')
        x = self.output(x)
        return x , feature

if __name__ == "__main__":
        data = torch.randn(size = (16,128,4))
        skl_data = torch.randn(size = (16,128,32,3))
        model = TransModel()
        output = model(data, skl_data, epochs = 20)
