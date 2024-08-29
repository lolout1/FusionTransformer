from functools import partial
from einops import rearrange
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from model_utils import Block

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
        self.skl_patch = self.skl_patch_size * (num_joints-8-8)
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
        self.skl_encode_size = (self.skl_patch//(temp_embed//8))* (temp_embed)
        #Spatial postional embedding
        # self.Spatial_pos_embed = nn.Parameter(torch.zeros((1, num_patch+1, spatial_embed)))
        self.temp_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        # self.proj_up_clstoken = nn.Linear(mocap_frames*spatial_embed, self.num_joints* spatial_embed)

        #Temporal Embedding  
        #adds postion info to every elementa
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_patch + 1,spatial_embed)) 

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
        #     #nn.Conv1d(72, 64,3,1,1),
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
            drop= drop_rate, attn_drop=attn_drop_rate, drop_path=adpr[i], norm_layer=norm_layer, blocktype='Sensor'
           )
           for i in range(self.adepth)
        ])

        self.Acc_encoder = nn.Sequential(
                nn.Linear(self.acc_patch_size * self.acc_coords, acc_embed),
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
        #self.intermediate_norm = nn.BatchNorm1d(9)

        #positional dropout 
        self.pos_drop = nn.Dropout(p = drop_rate)


        self.class_head = nn.Sequential(
            nn.LayerNorm(temp_embed),
            nn.Linear(temp_embed, num_classes)
        )
        
        # self.spatial_frame_reduce = nn.Conv1d(self.mocap_frames, self.acc_frames, 1,1)
        self.frame_reduce_mf = nn.Conv1d(self.mocap_frames, self.acc_frames, 1,1)
        self.frame_reduce_acc = nn.Conv1d(self.acc_frames+1, self.mocap_frames+1, 1, 1)

        self.acc_conv = nn.Sequential(nn.Conv2d(acc_coords, acc_coords, (1, acc_coords), 1), 
                                        nn.BatchNorm2d(acc_coords),
                                        nn.ReLU())

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_chans, in_chans, (1, 9), 1 ),
                                          nn.BatchNorm2d(in_chans),
                                          nn.ReLU(), 
                                          nn.Conv2d(in_chans, 1, (1, 9), 1), 
                                          nn.BatchNorm2d(1),
                                          nn.ReLU() )

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
        # ##get cross fusion indexe s
        cv_signals = []
        # for _, blk in enumerate(self.Accelerometer_blocks):
        #     cv_sig, x = blk(x)
        #     cv_signals.append(x)
        
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
    
    def Spatial_forward_features(self, x):

        b, f, p , c = x.shape 
        x = rearrange(x, 'b f p c -> (b f) p c') # B  = b x f


        if self.embed_type == 'conv':
            x = rearrange(x, '(b f) p c  -> (b f) c p',b=b ) # b x 3 x Fa  - Conv k liye channels first
            x = self.Spatial_patch_to_embedding(x) # B x c x p ->  B x Se x p
            x = rearrange(x, '(b f) Se p  -> (b f) p Se', b=b)
        else: 
            x = self.Spatial_patch_to_embedding(x) # B x p x c ->  B x p x Se
        
        class_token = torch.tile(self.spatial_token, (b*f, 1 , 1))
        x = torch.cat((x, class_token), dim = 1)

        x += self.Spatial_pos_embed 
        x = self.pos_drop(x)

        # for blk in self.Spatial_blocks:
        #     x = blk(x)
        
        # x = self.Spatial_norm(x)

        #extract class token 
        Se = x.shape[-1]
        cls_token = x[:,-1, :]
        cls_token = torch.reshape(cls_token, (b, f*Se))

        #reshape input 
        x = x[:, :p, :]
        x = rearrange(x, '(b f) p Se -> b f (p Se)', f = f)
        
        return x, cls_token
    
    def Temp_forward_features(self, x, cv_signals):

        b,f,St = x.shape
        cv_idx = 0 
        class_token = torch.tile(self.temp_token, (b, 1, 1))
        x = torch.cat((class_token , x), dim = 1)
        x += self.Temporal_pos_embed
        #for idx, blk in enumerate(self.Temporal_blocks):

            # skl_data = self.frame_reduce(x)
            #acc_data = cv_signals[idx]
            # if x.shape[1] > cv_signals[1].shape[1]-1:
            #     x = self.frame_reduce_mf(x)
            # elif x.shape[1] < cv_signals[1].shape[1] -1:
            # else: 
            #     x = x
           
            # x = blk(x) #output 3
            #x= x + acc_data #merged both 3
            #x = self.intermediate_norm(x)
        x = x + cv_signals
        x = self.Temporal_norm(x)

        ###Extract Class token head from the outputs
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

    def forward(self, acc_data, skl_data):

        #Input: B X Mocap_frames X Num_joints X in_channs
        b, f, j, c = skl_data.shape
        skl_data = skl_data
        #Extract skeletal signal from input 
        #x = inputs[:,:, :self.num_joints , :self.joint_coords]
        # acc_data = rearrange(acc_data, 'b f c -> b c f')
        # acc_data = F.avg_pool1d(acc_data,(acc_data.shape[-1]//4)-1,stride=1,padding = 3, count_include_pad =True)
        # acc_data = rearrange(acc_data, 'b c f -> b f c')

        #
        x = rearrange(skl_data, 'b f j c -> b c f j')
        x = self.spatial_conv(x)
        x = rearrange(x, 'b c f j -> b f j c')
        x = x.view(b, self.num_patch ,-1)
        #x = rearrange(x, 'b f j c -> b np (j pl c)' , np = 10, pl = 5)
        x = rearrange(x, 'b t c -> b c t')
        x = self.Spatial_encoder(x)
        spatial_out = F.avg_pool1d(x,x.shape[-1],stride=x.shape[-1]) #b x St x 1
        b, St, _ = x.shape
        spatial_out = torch.reshape(spatial_out, (b,St))
        x = rearrange(x, 'b c t -> b t c ')


        #Extract acc_signal from input 
        sx = acc_data
        sx = sx.view(b, self.num_patch, -1)
        sx = self.Acc_encoder(sx)

        #Get acceleration features 
        sx, cv_signals = self.Acc_forward_features(sx)
        #Get skeletal features
        #x, cls_token = self.Spatial_forward_features(x) # in: B x mocap_frames x num_joints x in_chann  out: x = b x mocap_frame x (num_joints*Se) cls_token b x mocap_frames*Se     
        #Pass cls  token to temporal transformer
        # print(f'Class token {cls_token.shape}')
        # temp_cls_token = self.proj_up_clstoken(cls_token) # in b x mocap_frames * se -> #out: b x num_joints*Se
        # temp_cls_token = torch.unsqueeze(temp_cls_token, dim = 1) #in: B x 1 x num_joints*Se)
        x = self.Temp_forward_features(x, sx) #in: B x mocap_frames x ()
        x = x + sx 
        logits = self.class_head(x)
        return logits





if __name__ == "__main__" :
    skl_data = torch.randn(size=(1, 128, 25, 3))
    # layer = nn.Sequential(nn.Conv2d(3, 3, (1, 9), 1), 
    #                                       nn.Conv2d(3, 1, (1, 9), 1))
    # transformed = layer(skl_data)
    # print(transformed.shape)
    acc_data = torch.randn(size = (1, 128, 3))
    model = MMTransformer(device = 'cpu', op_type='pool', mocap_frames= 128, num_patch=16, acc_frames = 128, num_joints = 25, in_chans = 3, acc_coords = 3, spatial_embed = 16, sdepth = 4, adepth = 4, tdepth = 4, num_heads = 8, mlp_ratio = 2, qkv_bias = True, qk_scale = None, embed_type = 'lin', drop_rate =0.2, attn_drop_rate = 0.2, drop_path_rate = 0.2, norm_layer = None, num_classes =27)
    logits = model(acc_data, skl_data)

    # model(acc_data, skl_data)