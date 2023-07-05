import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LinearModel(nn.Module):
    def __init__(self, acc_frame = 150, channel = 3, num_classes = 27):
        super().__init__()
        #32 , 150, 3
        self.ln1 = nn.Linear(acc_frame*channel , 512)
        self.drop1 = nn.Dropout(p = 0.3)
        self.ln2 = nn.Linear(512, 1024)
        self.drop2 = nn.Dropout(p = 0.3)
        self.ln3 = nn.Linear(1024, 512)
        self.drop3 = nn.Dropout(p = 0.3)
        self.ln4 = nn.Linear(512, 128)
        self.drop4 = nn.Dropout(p = 0.3)
        self.ln5 = nn.Linear(128, num_classes)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()

    def forward(self, inputs):
        inputs = rearrange(inputs, 'b f c -> b (f c)')
        x = self.ln1(inputs)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.ln3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x = self.ln4(x)
        x = self.act4(x)
        x = self.drop4(x)
        x = self.ln5(x)
        x = F.log_softmax(x , dim = 1)
        return x

