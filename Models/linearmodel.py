import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LinearModel(nn.Module):
    def __init__(self, acc_frames = 100, channel = 3, num_classes = 27, mocap_frames= 50, num_joints = 25):
        super().__init__()
        #32 , 150, 3
        self.acc_ln = nn.Linear(acc_frames*channel , 64)
        self.drop1 = nn.Dropout(p = 0.3)

        self.sk_ln1 = nn.Linear(mocap_frames*num_joints*channel,1024)
        self.sk_ln2 = nn.Linear(1024, 512)
        self.sk_ln3 = nn.Linear(512, 128)
        self.sk_ln4 = nn.Linear(128,64)
        self.drop2 = nn.Dropout(p = 0.3)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.3)
        self.ln5 = nn.Linear(64, 32)
        self.ln6 = nn.Linear(32, num_classes)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()
        self.act5 = nn.ReLU()

    def forward(self, acc_data, skl_data):
        acc_data = rearrange(acc_data, 'b f c -> b (f c)')
        skl_data = rearrange(skl_data,'b f j c -> b (f j c)' )
        acc_embed  = self.acc_ln(acc_data)
        acc_embed = self.act1(acc_embed)
        acc_embed = self.drop1(acc_embed)
        skl_embed = self.sk_ln1(skl_data)
        skl_embed = self.act2(skl_embed)
        x = self.drop2(skl_embed)
        x = self.sk_ln2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.sk_ln3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x = self.sk_ln4(x)
        x  = x + acc_embed
        x = self.act4(x)
        x = self.drop4(x)
        x = self.ln5(x)
        x = self.act5(x)
        x = self.ln6(x)
        return x