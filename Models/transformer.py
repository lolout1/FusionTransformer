import torch 
from torch import nn
from typing import Dict, Tuple
from torch.nn import Linear, LayerNorm, TransformerEncoder, TransformerEncoderLayer, ModuleList
import torch.nn.functional as F
from einops import rearrange
#from util.graph import Graph
import math

def gumbel_softmax(logits, tau=1, hard=False, dim=-1):
    gumbel_noise = torch.rand_like(logits)
    y = logits + (-torch.log(-torch.log(gumbel_noise))).detach()
    y_soft = F.softmax(y / tau, dim=dim)
    if hard:
        y_hard = torch.max(y_soft, dim=dim, keepdim=True)[1]
        y_hard = y_hard.squeeze(dim)
        y = (y_soft == y_hard).float()
    else:
        y = y_soft
    return y

class PeriodicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PeriodicLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Apply a linear transformation
        output = F.linear(input, self.weight, self.bias)
        # Apply a periodic activation function, e.g., sine or cosine
        output = torch.sin(output)  # You can also use torch.cos() or other periodic functions
        return output

class TransModel(nn.Module):
    def __init__(self, data_shape:Dict[str, Tuple[int, int]] = {'inertial':(128, 3)},
                mocap_frames = 128,
                num_joints = 32,
                acc_frames = 128,
                num_classes:int = 8, 
                num_heads = 4, 
                adepth = 2, norm_first = True, 
                acc_embed= 256, activation = 'relu',
                **kwargs) :
        super().__init__()
        self.data_shape = (acc_frames, 3)
        self.length = self.data_shape[0]
        size = self.data_shape[1]

        self.input_proj = nn.Sequential(nn.Conv1d(self.length, acc_embed, 1), nn.GELU(),
                                        nn.Conv1d(acc_embed, acc_embed, 1), nn.GELU(),
                                        nn.Conv1d(acc_embed, acc_embed, 1), nn.GELU(),
                                        nn.Conv1d(acc_embed, acc_embed, 1), nn.GELU())
        self.transform_layer = Linear(self.data_shape[-2], acc_embed)
        self.encoder_layer = TransformerEncoderLayer(d_model = acc_embed, activation = activation, 
                                                     dim_feedforward = 256, nhead = num_heads,dropout=0.5)
        
        self.encoder = TransformerEncoder(encoder_layer = self.encoder_layer, num_layers = adepth, 
                                          norm=nn.LayerNorm(acc_embed))

        self.reduciton =  ModuleList([Linear(acc_embed, int(acc_embed/2)),
                                   Linear(int(acc_embed/2), acc_embed//4)])
        self.feature_transform = nn.Linear(3, 16)
        pooled = acc_embed//8 + 1 
        self.ln1 = nn.Linear(pooled*16, 64)
        self.output = Linear(64, num_classes)
        nn.init.normal_(self.output.weight, 0, math.sqrt(2. / num_classes))
        #self.pool_layer = nn.AvgPool1D(kernel_size = 100, stride = (100, 1))  
        # self.hidden_size = acc_embed
        #self.lstm = nn.LSTM(size, acc_embed, batch_first = True)
        # nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))
                                                    
    
    def forward(self, acc_data, skl_data):

        b, l, c = acc_data.shape

        #x = torch.reshape(acc_data, [b, w, ((l//w)*c)])
        #mask = (acc_data != 0)
        #print(mask.shape)
        # gyrodata = acc_data[:,-3:, :]
        # accdata = acc_data[:, :3, :]
        #x = self.transform_layer(acc_data)
        x = self.input_proj(acc_data) # [ 8, 64, 3]
        x = rearrange(x,'b l c -> b c l') #[8, 64, 3]
        # gyro = self.periodic_transform(gyrodata)
        # gyro = self.encoder(gyro)
        x = self.encoder(x)
        x = rearrange(x, 'b c l -> b l c')
        x = self.feature_transform(x)
        x = rearrange(x, 'b l c -> b c l')
        #print(f'transformer {x.shape}')
        for i, l in enumerate(self.reduciton):
            x = l(x)

        # for i, l in enumerate(self.reduciton):
        #     gyro = l(gyro)
        # gyro = F.max_pool1d(gyro, kernel_size = x.shape[-1]//2, stride = 1)
        # gyro = rearrange(gyro, 'b c f -> b (c f)')

        # gyro = self.ln1(gyro)
        x = F.max_pool1d(x, kernel_size = x.shape[-1]//2, stride = 1)
        x = rearrange(x, 'b c f -> b (c f)')
        x = self.ln1(x)


        # lstm_x = self.lstm(acc_data)
        # for i, l in enumerate(self.reduciton):
        #     lstm_x = l(x)
        # x = torch.cat((gyro, x), dim = 1)

        
        
        x = self.output(x)
        #x = gumbel_softmax(x, dim = -2)
        #print(x)
        # x = 
        # h0 = torch.zeros(1, acc_data.size(1), self.hidden_size).to(acc_data.device)
        # # Initialize cell state with zeros
        # c0 = torch.zeros(1, acc_data.size(1), self.hidden_size).to(acc_data.device)
        
        # # Forward pass through LSTM layer
        # lstm_out, _ = self.lstm(acc_data)

        # # Only take the output from the final time step
        # output = self.fc(lstm_out[:, -1, :])
        # output = self.sigmoid(output)
        return x