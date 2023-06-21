import math
import torch


def sin_pos_embedding(x: torch.Tensor, start, end, number, cutoff=True):
    
    x = (x[..., None] - start) / (end - start)
    i = torch.arange(1, number, dtype=x.dtype, device=x.device)
    if not cutoff:
        temp = torch.cos(math.pi * i * x)
    else:
        temp = torch.sin(math.pi * i * x) * (0 < x) * (x < 1)
    x = x*2 - 1
    temp = torch.cat([temp, x], dim=-1)
    return temp
