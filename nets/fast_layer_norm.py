import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
 
    
class EquivariantLayerNormFast(nn.Module):
    
    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"


    def forward(self, node_input, **kwargs):
        '''
            Use torch layer norm for scalar features.
        '''
        
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = node_input.narrow(1, ix, mul*d)
            ix += mul * d
            
            if ir.l == 0 and ir.p == 1:
                weight = self.affine_weight[iw:(iw + mul)]
                bias = self.affine_bias[ib:(ib + mul)] 
                iw += mul
                ib += mul 
                field = F.layer_norm(field, tuple((mul, )), weight, bias, self.eps)
                fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]
                continue
            
            # For non-scalar features, use RMS value for std
            field = field.reshape(-1, mul, d)   # [batch * sample, mul, repr]
            
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)    
            field_norm = 1.0 / ((field_norm + self.eps).sqrt())  # [batch * sample, mul]

            if self.affine:
                weight = self.affine_weight[None, iw:(iw + mul)]  # [1, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch * sample, mul]
            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]
            
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        assert ix == dim
        
        output = torch.cat(fields, dim=-1)
        return output
