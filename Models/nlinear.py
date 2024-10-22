import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, acc_frames = 128, num_class = 9, channels = 3, individual = True, mocap_frames = 128):
        super(NLinear, self).__init__()
        self.acc_frames = acc_frames
        self.num_class = num_class
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.acc_frames)*torch.ones([self.num_class,self.seq_len]))
        self.channels = channels
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.acc_frames,self.num_class))
        else:
            self.Linear = nn.Linear(self.acc_frames, self.num_class)
        
        self.final_layer = nn.Linear(num_class*channels, num_class)

    def forward(self, x, skl_data):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        #x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.num_class,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        
        #x = x + seq_last
        x = x.view(x.size(0), -1)
        x = self.final_layer(x)
        #x = F.avg_pool1d(x , kernel_size=self.channels, stride=1)
        return x# [Batch, Output length, Channel]


if __name__ == "__main__":
    data = torch.randn((8,128,3))
    model = Model()
    output = model(data)
