from functools import partial
import math
from einops import rearrange
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Block ,PredictorLG, TokenExchange

class MMTransformer(nn.Module):
    def __init__(self, device = 'cpu', mocap_frames= 600, acc_frames = 256, num_joints = 31, in_chans = 3, num_patch = 10 ,  acc_coords = 3, spatial_embed = 32, acc_embed = 32, sdepth = 4, adepth = 4, tdepth = 4, num_heads = 8, mlp_ratio = 2, qkv_bias = True, qk_scale = None, op_type = 'all', embed_type = 'lin', drop_rate =0.2, attn_drop_rate = 0.1, drop_path_rate = 0.1, norm_layer = None, num_classes =11):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps = 1e-6)

        ##### I might change temp_embed later to 512
        temp_embed = spatial_embed
        #acc_embed = temp_embed
        self.acc_embed = acc_embed
        self.num_patch = num_patch
        self.mocap_frames = mocap_frames
        self.skl_patch_size = mocap_frames // num_patch
        self.skl_patch = self.skl_patch_size * (num_joints-8-8)        
        #print(self.skl_patch_size)
        #print(self.skl_patch)
        self.acc_patch_size = acc_frames // num_patch
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
        self.skl_encode_size = (self.skl_patch//(temp_embed//2))* (temp_embed)
        print(self.skl_encode_size)
        #print(self.skl_encode_size)
        
        #Spatial postional embedding
        # self.Spatial_pos_embed = nn.Parameter(torch.zeros((1, num_patch+1, spatial_embed)))
        self.temp_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        # self.proj_up_clstoken = nn.Linear(mocap_frames*spatial_embed, self.num_joints* spatial_embed)

        #Temporal Embedding  
        #adds postion info to every elementa
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_patch + 1,spatial_embed)) 

        #accelerometer positional embedding
        self.Acc_pos_embed = nn.Parameter(torch.zeros(1, num_patch+1 ,acc_embed))

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

        #spatial encoder block 
        # self.Spatial_blocks = nn.ModuleList([
        #     Block(
        #         dim = spatial_embed, num_heads=num_heads, mlp_ratio= mlp_ratio, qkv_bias  = qkv_bias, qk_scale= qk_scale, 
        #         drop = drop_rate, attn_drop=attn_drop_rate, drop_path=sdpr[i], norm_layer=norm_layer
        #     ) 
        #     for i in range(self.sdepth)
        # ])

        # self.Spatial_encoder = nn.Sequential(
        #         nn.Linear(240, 256),
        #         nn.ReLU(), 
        #         # nn.Linear(512,256),
        #         # nn.ReLU(),
        #         nn.Linear(256, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 64),
        #         nn.ReLU(),
        #         #best for utd
        #         # nn.Linear(18,16),
        #         #experiment for berkley
        #         nn.Linear(64, 32),
        #         nn.ReLU(), )
        #         #nn.Linear(32, 16), 
        #         #nn.ReLU())
        # self.Spatial_encoder = nn.Sequential(
        #     nn.Conv1d(120, 64, 3, 1 , 1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 32, 3, 1, 1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 16, 3, 1, 1),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),

        # )
        self.Spatial_encoder = nn.Sequential(
            nn.Conv1d(self.skl_patch, self.skl_encode_size, 3, 1, 1), 
            nn.BatchNorm1d((self.skl_encode_size)), 
            nn.ReLU(), 
            nn.Conv1d(self.skl_encode_size ,self.skl_encode_size//2 , 3, 1, 1),
            nn.BatchNorm1d((self.skl_encode_size//2)),
            nn.ReLU(),
            nn.Conv1d(self.skl_encode_size//2, temp_embed, 3, 1 , 1),
            nn.BatchNorm1d(temp_embed),
            nn.ReLU()
        )
        #temporal encoder block 
        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim = temp_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop= drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[i], norm_layer=norm_layer
            )
            for i in range(self.tdepth)
        ])

        #accelerometer encoder block
        self.Accelerometer_blocks = nn.ModuleList([
           Block(
            dim = acc_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop= drop_rate, attn_drop=attn_drop_rate, drop_path=adpr[i], norm_layer=norm_layer
           )
           for i in range(self.adepth)
        ])

        self.Acc_encoder = nn.Sequential(
                nn.Linear(self.acc_patch_size * self.acc_coords, self.acc_embed),
                nn.ReLU(),
                # nn.Linear(32, 16),
                # nn.ReLU(),
                # nn.Linear(64, 128),
                # nn.ReLU()
        )
        #norm layer 
        self.Spatial_norm = norm_layer(spatial_embed)
        self.Acc_norm = norm_layer(acc_embed)
        self.Temporal_norm = norm_layer(temp_embed)

        #positional dropout 
        self.pos_drop = nn.Dropout(p = drop_rate)


        self.class_head = nn.Sequential(
            nn.LayerNorm(self.acc_embed*2),
            nn.Linear(self.acc_embed*2, num_classes)
        )
        
        # self.spatial_frame_reduce = nn.Conv1d(self.mocap_frames, self.acc_frames, 1,1)
        # self.frame_reduce_mf = nn.Conv1d(self.mocap_frames, self.acc_frames, 1,1)
        # self.frame_reduce_acc = nn.Conv1d(self.acc_frames+1, self.mocap_frames+1, 1, 1)
        self.down_samp = nn.ModuleList([nn.Linear(temp_embed, self.acc_embed) for i in range(self.adepth)])
        self.up_samp = nn.ModuleList([nn.Linear(self.acc_embed, temp_embed) for i in range(self.adepth)])

        self.acc_conv = nn.Sequential(nn.Conv2d(acc_coords, acc_coords, (1, acc_coords), 1), 
                                        nn.BatchNorm2d(acc_coords),
                                        nn.ReLU())


        self.spatial_conv = nn.Sequential(nn.Conv2d(in_chans, in_chans, (1, 9), 1 ),
                                          nn.BatchNorm2d(in_chans),
                                          nn.ReLU(), 
                                          nn.Conv2d(in_chans, 1, (1, 9), 1), 
                                          nn.BatchNorm2d(1),
                                          nn.ReLU() )

        #score net 
        self.score_net = nn.ModuleList([PredictorLG(embed_dim= acc_embed) for i in range(self.adepth)])
        self.exchange_net = nn.ModuleList([TokenExchange() for i in range(self.adepth)])

    def Acc_forward_features(self, x):
        b,f,e = x.shape

        #x = rearrange(x, 'b f p c -> b f (p c)') # 16 , 150, 1, 3
        
        # if self.embed_type == 'conv':
        #     x = rearrange(x, '(b f) p c  -> (b f) c p',b=b ) # b x 3 x Fa  - Conv k liye channels first
        #     x = self.Acc_coords_to_embedding(x) # B x c x p ->  B x Sa x p
        #     x = rearrange(x, '(b f) Sa p  -> (b f) p Sa', b=b)
        # else: 
        #     x = self.Acc_coords_to_embedding(x)
        class_token = torch.tile(self.acc_token, (b,1,1))
        x = torch.cat((x, class_token), dim = 1)
        _,_,Sa = x.shape

        x += self.Acc_pos_embed
        x = self.pos_drop(x)
        ##get cross fusion indexes
        cv_signals = []
        for _, blk in enumerate(self.Accelerometer_blocks):
            cv_sig, x = blk(x)
            cv_signals.append(x)
        
        x = self.Acc_norm(x)
        cls_token = x[:,-1,:]    

        if self.op_type == 'cls':
            return cls_token , cv_signals

        else:
            x = x[:,:f,:]
            x = rearrange(x, 'b f Sa -> b Sa f')
            x = F.avg_pool1d(x,x.shape[-1],stride=x.shape[-1]) #b x Sa x 1
            x = torch.reshape(x, (b,Sa))
            return x,cv_signals #b x Sa
    
    # def Spatial_forward_features(self, x):

    #     b, f, p , c = x.shape 
    #     x = rearrange(x, 'b f p c -> (b f) p c') # B  = b x f


    #     if self.embed_type == 'conv':
    #         x = rearrange(x, '(b f) p c  -> (b f) c p',b=b ) # b x 3 x Fa  - Conv k liye channels first
    #         x = self.Spatial_patch_to_embedding(x) # B x c x p ->  B x Se x p
    #         x = rearrange(x, '(b f) Se p  -> (b f) p Se', b=b)
    #     else: 
    #         x = self.Spatial_patch_to_embedding(x) # B x p x c ->  B x p x Se
        
    #     class_token = torch.tile(self.spatial_token, (b*f, 1 , 1))
    #     x = torch.cat((x, class_token), dim = 1)

    #     x += self.Spatial_pos_embed 
    #     x = self.pos_drop(x)

    #     # for blk in self.Spatial_blocks:
    #     #     x = blk(x)
        
    #     # x = self.Spatial_norm(x)

    #     #extract class token 
    #     Se = x.shape[-1]
    #     cls_token = x[:,-1, :]
    #     cls_token = torch.reshape(cls_token, (b, f*Se))

    #     #reshape input 
    #     x = x[:, :p, :]
    #     x = rearrange(x, '(b f) p Se -> b f (p Se)', f = f)
        
    #     return x, cls_token
    
    def Temp_forward_features(self, x, cv_signals):

        b,f,St = x.shape
        cv_idx = 0 
        class_token = torch.tile(self.temp_token, (b, 1, 1))
        x = torch.cat((class_token , x), dim = 1)
        x += self.Temporal_pos_embed
        for idx, blk in enumerate(self.Temporal_blocks):
            # print(f' In temporal {x.shape}')
            # skl_data = self.frame_reduce(x)
            # print(idx)
            acc_data = cv_signals[idx]
            # if x.shape[1] > cv_signals[1].shape[1]-1:
            #     x = self.frame_reduce_mf(x)
            # elif x.shape[1] < cv_signals[1].shape[1] -1:
            # else: 
            #     x = x
           
            x = blk(x) #output 3
            x= x + acc_data #merged both 3
        x = self.Temporal_norm(x)

        ###Extract Class token head from the output
        if self.op_type=='cls':
            cls_token = x[:,1,:]
            cls_token = cls_token.view(b, -1) # (Batch_size, temp_embed)
            return cls_token

        else:
            x = x[:,:f,:]
            x = rearrange(x, 'b f St -> b St f')
            x = F.avg_pool1d(x,x.shape[-1],stride=x.shape[-1]) #b x St x 1
            x = torch.reshape(x, (b,St))
            return x #b x St 
    #need two scores for 2 channels
    def joint_forward(self, skl_data, acc_data): 
        b,f,St = acc_data.shape

        acc_class_token = torch.tile(self.acc_token, (b, 1, 1))
        skl_class_token = torch.tile(self.temp_token, (b, 1, 1))

        #adding class token 
        acc_data = torch.cat((acc_class_token, acc_data), dim = 1)
        skl_data = torch.cat((skl_class_token, skl_data ), dim = 1)

        #adding positional embedding
        acc_data += self.Acc_pos_embed
        skl_data += self.Temporal_pos_embed
        masks = []
        for idx in range(self.adepth):
            acc_data = self.Accelerometer_blocks[idx](acc_data)
            #print(acc_data[0])
            skl_data = self.Temporal_blocks[idx](skl_data)

            #always keep skl_embed larger
            down_skl = self.down_samp[idx](skl_data)
            
            
            scores = self.score_net[idx]([down_skl, acc_data])
            mask = [F.softmax(score_.reshape(b, -1, 2), dim=2)[:, :, 0] for score_ in scores]

            masks.append([mask_.flatten() for mask_ in mask])
            acc_data, skl_data = self.exchange_net[idx]([acc_data, down_skl], mask, 0.04)
            if idx != self.adepth-1:
                skl_data = self.up_samp[idx](skl_data)

        x = torch.cat((acc_data , skl_data), dim = -1)

        if self.op_type=='cls':
            cls_token = x[:,0,:]
            cls_token = cls_token.view(b, -1) # (Batch_size, temp_embed)
            return masks, cls_token

        else:
            x = x[:,:f,:]
            x = rearrange(x, 'b f St -> b St f')
            x = F.avg_pool1d(x,x.shape[-1],stride=x.shape[-1]) #b x St x 1
            x = torch.reshape(x, (b,St*2))
            return masks,x #b x St 

    def forward(self, acc_data, skl_data):

        #Input: B X Mocap_frames X Num_joints X in_channs
        b, f, j, c = skl_data.shape
        skl_data = skl_data
        #Extract skeletal signal from input 
        #x = inputs[:,:, :self.num_joints , :self.joint_coords]
        x = rearrange(skl_data, 'b f j c -> b c f j')
        x = self.spatial_conv(x)
        x = rearrange(x, 'b c f j -> b f j c')
        x = x.view(b, self.num_patch ,-1)
        #x = rearrange(x, 'b f j c -> b np (j pl c)' , np = 10, pl = 5)
        x = rearrange(x, 'b t c -> b c t')
        x = self.Spatial_encoder(x)
        x = rearrange(x, 'b c t -> b t c ')
        #Extract acc_signal from input 
        #sx = inputs[:, 0, self.num_joints:, :self.acc_coords]
        sx = acc_data
        # sx = rearrange(acc_data,'b f c -> b c f' )
        # sx = self.acc_conv(sx)
        # sx = rearrange(sx,'b c f -> b f c' )
        sx = sx.view(b, self.num_patch, -1)
        sx = self.Acc_encoder(sx)
        b,f,St = sx.shape
        masks,ex_out  = self.joint_forward(x, sx)

        #sx = torch.reshape(sx, (b,-1,1,self.acc_coords)) #batch X acc_frames X 1 X acc_channel


        logits = self.class_head(ex_out)

        return masks, logits, F.log_softmax(logits,dim =1)





if __name__ == "__main__" :
    skl_data = torch.randn(size=(2, 512, 32, 3))
    acc_data = torch.randn(size = (2, 32, 3))

    model = MMTransformer(device = 'cpu', mocap_frames= 512, acc_frames = 32, num_joints = 32, num_patch=16, in_chans = 3, acc_coords = 3, spatial_embed = 32, acc_embed= 16, sdepth = 4, adepth = 4, tdepth = 4, num_heads = 8, mlp_ratio = 2, qkv_bias = True, qk_scale = None, op_type = 'cls', embed_type = 'lin', drop_rate =0.2, attn_drop_rate = 0.2, drop_path_rate = 0.2, norm_layer = None, num_classes =27)
    model(acc_data, skl_data)

    # model(acc_data, skl_data)


