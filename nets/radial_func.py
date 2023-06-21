import torch
import torch.nn as nn
from torch.nn import init
import math
from e3nn import o3
from .fast_activation import Activation
   

class RadialProfile(nn.Module):
    def __init__(self, ch_list, use_layer_norm=True, use_offset=True):
        super().__init__()
        modules = []
        input_channels = ch_list[0]
        for i in range(len(ch_list)):
            if i == 0:
                continue
            if (i == len(ch_list) - 1) and use_offset:
                use_biases = False
            else:
                use_biases = True
            modules.append(nn.Linear(input_channels, ch_list[i], bias=use_biases))
            input_channels = ch_list[i]
            
            if i == len(ch_list) - 1:
                break
            
            if use_layer_norm:
                modules.append(nn.LayerNorm(ch_list[i]))
            #modules.append(nn.ReLU())
            #modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])), 
            #    acts=[torch.nn.functional.silu]))
            #modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])), 
            #    acts=[ShiftedSoftplus()]))
            modules.append(torch.nn.SiLU())
        
        self.net = nn.Sequential(*modules)
        
        self.offset = None
        if use_offset:
            self.offset = nn.Parameter(torch.zeros(ch_list[-1]))
            fan_in = ch_list[-2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.offset, -bound, bound)
            
        
    def forward(self, f_in):
        f_out = self.net(f_in)
        if self.offset is not None:
            f_out = f_out + self.offset.reshape(1, -1) 
        return f_out