from functools import partial
from einops import rearrange
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from Models.model_utils import Block

class MMTransformer(nn.Module):
    def __init__(self, device = 'cpu', mocap_frames= 600, acc_frames = 256, num_joints = 31, in_chans = 3, num_patch = 10 ,  acc_coords = 3, spatial_embed = 32, sdepth = 4, adepth = 4, tdepth = 4, num_heads = 8, mlp_ratio = 2, qkv_bias = True, qk_scale = None, op_type = 'all', embed_type = 'lin', drop_rate =0.2, attn_drop_rate = 0.2, drop_path_rate = 0.2, norm_layer = None, num_classes =11):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps = 1e-6)

        ##### I might change temp_embed later to 512
        temp_embed = spatial_embed
        acc_embed = temp_embed
        self.num_patch = num_patch
        self.mocap_frames = mocap_frames
        self.skl_patch_size = mocap_frames // num_patch
        self.acc_patch_size = acc_frames // num_patch
        self.skl_patch = self.skl_patch_size * (num_joints-8-8)//2
        self.temp_frames = mocap_frames
        self.op_type = op_type
        self.embed_type = embed_type
        self.sdepth = sdepth
        self.adepth = adepth 
        self.tdepth = sdepth 
        self.num_joints = num_joints
        self.joint_coords = in_chans
        self.acc_frames = acc_frames
        self.acc_coords = acc_coords
        #self.skl_encode_size = (self.skl_patch//(temp_embed//8))* (temp_embed)
        self.skl_encoder_size = temp_embed
        #Spatial postional embedding
        # self.Spatial_pos_embed = nn.Parameter(torch.zeros((1, num_patch+1, spatial_embed)))
        self.temp_token = nn.Parameter(torch.zeros(1, 1, mocap_frames))
        # self.proj_up_clstoken = nn.Linear(mocap_frames*spatial_embed, self.num_joints* spatial_embed)

        #Temporal Embedding  
        #adds postion info to every element
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, 1,spatial_embed)) 

        #accelerometer positional embedding
        self.Acc_pos_embed = nn.Parameter(torch.zeros(1, 1,acc_embed))

        #acceleromter 
        #a token to add class info to all the patches
        #if patch size is larger than one change the acc_embed to patch_number
        self.acc_token = nn.Parameter(torch.zeros((1, 1, acc_embed))) 
        
        #linear transformer of the raw skeleton and accelerometer data
        if self.embed_type == 'lin':
            self.Spatial_patch_to_embedding = nn.Linear(in_chans, spatial_embed)

            self.Acc_coords_to_embedding = nn.Linear(acc_coords, acc_embed)
        else:
            ## have confusion about Conv1D
            self.Spatial_patch_to_embedding= nn.Conv1d(in_chans , spatial_embed, 1, 1)

            self.Acc_coords_to_embedding = nn.Conv1d(acc_coords, acc_embed, 1, 1)
        
        #
        sdpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.sdepth)]
        adpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.adepth)]
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.tdepth)]

        self.Spatial_encoder = nn.Sequential(
            nn.Conv1d(16, self.skl_encoder_size, 3, 1, 1,), 
            nn.BatchNorm1d((self.skl_encoder_size)), 
            nn.ReLU(), 
            # nn.Conv1d(self.skl_encoder_size ,self.skl_encoder_size , 3, 1, 1),
            # nn.BatchNorm1d((self.skl_encoder_size)),
            # nn.ReLU(),
            # nn.Conv1d(self.skl_encoder_size, temp_embed, 3, 1 , 1),
            # nn.BatchNorm1d(temp_embed),
            # nn.ReLU()
        )

        #temporal encoder block 
        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim = temp_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop= drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[i], norm_layer=norm_layer
            )
            for i in range(self.tdepth)
        ])
        
        #joint relation block 
        self.joint_block = nn.ModuleList([
             Block(
               dim = self.skl_encoder_size  , num_heads = num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
              drop= drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[0], norm_layer=norm_layer

             )
         ])
        #accelerometer encoder block
        # self.Accelerometer_blocks = nn.ModuleList([
        #    Block(
        #     dim = acc_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop= drop_rate, attn_drop=attn_drop_rate, drop_path=adpr[i], norm_layer=norm_layer, blocktype='Sensor'
        #    )
        #    for i in range(self.adepth)
        # ])



        self.Spatial_norm = norm_layer(spatial_embed)
        self.Acc_norm = norm_layer(acc_embed)
        self.   Temporal_norm = norm_layer(temp_embed)

        self.pos_drop = nn.Dropout(p = drop_rate)


        self.class_head = nn.Sequential(
            nn.LayerNorm(temp_embed),
            nn.Linear(temp_embed, num_classes)
        )
        

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_chans, in_chans, (1, 9), 1 ),
                                          nn.BatchNorm2d(in_chans),
                                          nn.ReLU(), 
                                          nn.Conv2d(in_chans, 1, (1, 9), 1), 
                                          nn.BatchNorm2d(1),
                                          nn.ReLU() )
        
        self.transform = nn.Sequential(
                            nn.Linear(16, 32),
                            nn.ReLU())
        
    
    def Temp_forward_features(self, x):

        b,f,St = x.shape
        cv_idx = 0 
        #class_token = torch.tile(self.temp_token, (b, 1, 1))
        #x = torch.cat((x, class_token), dim = 1)
        x += self.Temporal_pos_embed
        for idx, blk in enumerate(self.Temporal_blocks):
            x = blk(x) #output 3

            #x= x + acc_data #merged both 3
            # x = self.intermediate_norm(x)
        x = self.Temporal_norm(x)

        ###Extract Class token head from the outputs
        if self.op_type=='cls':
            cls_token = x[:,1,:]
            cls_token = cls_token.view(b, -1) # (Batch_size, temp_embed)
            return cls_token

        else:
            x = x[:,:f,:]
            feature = self.Temporal_norm(x)
            x = rearrange(x, 'b f St -> b St f')
            x = F.avg_pool1d(x,x.shape[-1],stride=x.shape[-1]) #b x St x 1
            x = torch.reshape(x, (b,St))
            return x , feature #b x St 

    def forward(self, acc_data, skl_data, **kwargs):
        #Input: B X Mocap_frames X Num_joints X in_channs
        b, f, j, c = skl_data.shape
        j = j + 1

   
        acc_data = acc_data.unsqueeze(2)
        #Extract skeletal signal from input 
        #x = inputs[:,:, :self.num_joints , :self.joint_coords]
        # acc_data = rearrange(acc_data, 'b f c -> b c f')
        # acc_data = F.avg_pool1d(acc_data,(acc_data.shape[-1]//4)-1,stride=1,padding = 3, count_include_pad =True)
        # acc_data = rearrange(acc_data, 'b c f -> b f c')
        #acc_data = torch.reshape(acc_data, [b,f,2,3])
        #combined_data = torch.cat((skl_data, acc_data), dim = 2) #[8, 128 , 3] [8, 128, 32, 3]

        #Spatial Block 
        #channels,timestamps,joints  
        x = rearrange(skl_data, 'b f j c -> b c f j')
        x = self.spatial_conv(x)
        x = rearrange(x, 'b c f j -> b (j c) f')
        x = self.Spatial_encoder(x)
        #x = self.transform(x)
        x = rearrange(x, 'b e f -> b f e')

        for idx, block in enumerate(self.joint_block):
            x = block(x)
            # if idx == 0: 
            #     feature = x[:,:f,:]
            #     #feature = rearrange(feature, 'b j f -> b f j')
            #     feature = self.Temporal_norm(feature)
                # feature = F.avg_pool1d(feature, kernel_size=feature.shape[-1], stride=1)
                # feature = torch.flatten(feature, 1)
        #x = rearrange(x, 'b f c -> b c f')

        #Extract acc_signal from input 
        # b, f, c = acc_data.shape
        # sx = acc_data
        # sx = sx.view(b, self.num_patch, -1)
        # sx = self.Acc_encoder(sx)

        #Get acceleration features 
        # sx, cv_signals = self.Acc_forward_features(sx)
        #Get skeletal features
        #x, cls_token = self.Spatial_forward_features(x) # in: B x mocap_frames x num_joints x in_chann  out: x = b x mocap_frame x (num_joints*Se) cls_token b x mocap_frames*Se     
        #Pass cls  token to temporal transformer
        # print(f'Class token {cls_token.shape}')
        # temp_cls_token = self.proj_up_clstoken(cls_token) # in b x mocap_frames * se -> #out: b x num_joints*Se
        # temp_cls_token = torch.unsqueeze(temp_cls_token, dim = 1) #in: B x 1 x num_joints*Se)
        x , feature = self.Temp_forward_features(x) #in: B x mocap_frames x ()
        #feature = rearrange(feature, 'b c f -> b f c')
        #x = x + sx 
        #x = sx
        logits = self.class_head(x)
        return logits, feature





if __name__ == "__main__" :
    skl_data = torch.randn(size=(1,128, 32, 3))
    # layer = nn.Sequential(nn.Conv2d(3, 3, (1, 9), 1), 
    #                                       nn.Conv2d(3, 1, (1, 9), 1))
    # transformed = layer(skl_data)
    # print(transformed.shape)
    acc_data = torch.randn(size = (1, 128, 3))
    model = MMTransformer(device = 'cpu', mocap_frames= 128, num_patch=4, acc_frames = 128, num_joints =32, in_chans = 3, acc_coords = 3,
                           spatial_embed = 16, sdepth = 4, adepth = 4, tdepth = 2, num_heads = 2 , mlp_ratio = 2, qkv_bias = True, qk_scale = None,
                          op_type = 'pool', embed_type = 'lin', drop_rate =0.2, attn_drop_rate = 0.2, drop_path_rate = 0.2, norm_layer = None, num_classes = 8)
    model(acc_data, skl_data)
    # model(acc_data, skl_data)
