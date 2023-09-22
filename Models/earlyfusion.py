from functools import partial
from einops import rearrange
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Block

class MMTransformer(nn.Module):
    def __init__(self, device = 'cpu', mocap_frames= 600, acc_frames = 256, num_joints = 31, in_chans = 3, acc_coords = 3, spatial_embed = 32, sdepth = 4, adepth = 4, tdepth = 4, num_heads = 8, mlp_ratio = 2, qkv_bias = True, qk_scale = None, op_type = 'cls', embed_type = 'lin', drop_rate =0.2, attn_drop_rate = 0.2, drop_path_rate = 0.2, norm_layer = None, num_classes =11):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps = 1e-6)

        ##### I might change temp_embed later to 512
        temp_embed = spatial_embed * num_joints # 31 * 32
        acc_embed = temp_embed
        
        self.mocap_frames = mocap_frames
        self.temp_frames = mocap_frames
        self.op_type = op_type
        self.embed_type = embed_type
        self.sdepth = sdepth
        self.adepth = adepth 
        self.tdepth = tdepth
        self.num_joints = num_joints
        self.joint_coords = in_chans
        self.acc_frames = acc_frames
        self.acc_coords = acc_coords
        
        #Spatial postional embedding
        self.Spatial_pos_embed = nn.Parameter(torch.zeros((1, self.num_joints+1, spatial_embed)))
        self.spatial_token = nn.Parameter(torch.zeros(1, 1, spatial_embed))
        self.proj_up_clstoken = nn.Linear(mocap_frames*spatial_embed, self.num_joints* spatial_embed)

        #Temporal Embedding  
        #adds postion info to every elementa
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.acc_frames+1, temp_embed)) 

        #accelerometer positional embedding
        self.Acc_pos_embed = nn.Parameter(torch.zeros(1, self.acc_frames+1, acc_embed))

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
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim = spatial_embed, num_heads=num_heads, mlp_ratio= mlp_ratio, qkv_bias  = qkv_bias, qk_scale= qk_scale, 
                drop = drop_rate, attn_drop=attn_drop_rate, drop_path=sdpr[i], norm_layer=norm_layer
            ) 
            for i in range(self.sdepth)
        ])

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

        #norm layer 
        self.Spatial_norm = norm_layer(spatial_embed)
        self.Acc_norm = norm_layer(acc_embed)
        self.Temporal_norm = norm_layer(temp_embed)

        #positional dropout 
        self.pos_drop = nn.Dropout(p = drop_rate)


        self.class_head = nn.Sequential(
            nn.LayerNorm(temp_embed),
            nn.Linear(temp_embed, num_classes)
        )
        
        # self.spatial_frame_reduce = nn.Conv1d(self.mocap_frames, self.acc_frames, 1,1)
        self.frame_reduce_mf = nn.Conv1d(self.mocap_frames, self.acc_frames, 1,1)
        self.frame_reduce_acc = nn.Conv1d(self.acc_frames+1, self.mocap_frames+1, 1, 1)

    def Acc_forward_features(self, x):
        b,f,p,c = x.shape

        x = rearrange(x, 'b f p c -> b f (p c)')
        
        if self.embed_type == 'conv':
            x = rearrange(x, '(b f) p c  -> (b f) c p',b=b ) # b x 3 x Fa  - Conv k liye channels first
            x = self.Acc_coords_to_embedding(x) # B x c x p ->  B x Sa x p
            x = rearrange(x, '(b f) Sa p  -> (b f) p Sa', b=b)
        else: 
            x = self.Acc_coords_to_embedding(x)
        
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
    
    def Temp_forward_features(self, x, cls_token, cv_signals):

        b,f,St = x.shape
        
        cv_idx = 0 
        x = torch.cat((x, cls_token), dim = 1)
        for idx, blk in enumerate(self.Temporal_blocks):
            # print(f' In temporal {x.shape}')
            # skl_data = self.frame_reduce(x)
            
            acc_data = cv_signals[cv_idx]
            if x.shape[1] > cv_signals[1].shape[1]-1:
                x = self.frame_reduce_mf(x)
            elif x.shape[1] < cv_signals[1].shape[1] -1:
                acc_data = self.frame_reduce_acc(acc_data)
            else: 
                x = x
            fused_data = x + acc_data
            x = blk(fused_data)
        x = self.Temporal_norm(x)

        ###Extract Class token head from the output
        if self.op_type=='cls':
            cls_token = x[:,-1,:]
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
        b, _, _, c = skl_data.shape

        #Extract skeletal signal from input 
        #x = inputs[:,:, :self.num_joints , :self.joint_coords]
        x = skl_data
        #Extract acc_signal from input 
        #sx = inputs[:, 0, self.num_joints:, :self.acc_coords]
        sx = acc_data
        sx = torch.reshape(sx, (b,-1,1,self.acc_coords)) #batch X acc_frames X 1 X acc_channel

        #Get acceleration features 
        sx, cv_signals = self.Acc_forward_features(sx)
        print(f'Output from accelerometed {sx.shape}')
        #Get skeletal features
        x, cls_token = self.Spatial_forward_features(x) # in: B x mocap_frames x num_joints x in_chann  out: x = b x mocap_frame x (num_joints*Se) cls_token b x mocap_frames*Se

        print(f'Output from spatial transformer {x.shape}')       

        #Pass cls  token to temporal transformer
        # print(f'Class token {cls_token.shape}')
        temp_cls_token = self.proj_up_clstoken(cls_token) # in b x mocap_frames * se -> #out: b x num_joints*Se
        temp_cls_token = torch.unsqueeze(temp_cls_token, dim = 1) #in: B x 1 x num_joints*Se

        x = self.Temp_forward_features(x, temp_cls_token, cv_signals) #in: B x mocap_frames x ()

        logits = self.class_head(x)

        return x , logits, F.log_softmax(logits,dim =1)








