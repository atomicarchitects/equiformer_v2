import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode


# Reference:
#   https://github.com/NVIDIA/DeepLearningExamples/blob/master/DGLPyTorch/DrugDiscovery/SE3Transformer/se3_transformer/model/layers/norm.py
#   https://github.com/e3nn/e3nn/blob/main/e3nn/nn/_batchnorm.py
@compile_mode('unsupported')
class EquivariantLayerNorm(torch.nn.Module):
    
    NORM_CLAMP = 2 ** -24  # Minimum positive subnormal for FP16
    
    def __init__(self, irreps_in, eps=1e-5):
        super().__init__()
        self.irreps_in = irreps_in
        self.eps = eps
        self.layer_norms = []
        
        for idx, (mul, ir) in enumerate(self.irreps_in):
            self.layer_norms.append(torch.nn.LayerNorm(mul, eps))
        self.layer_norms = torch.nn.ModuleList(self.layer_norms)
        
        #self.relu = torch.nn.ReLU()
        
    
    def forward(self, f_in, **kwargs):
        '''
            Assume `f_in` is of shape [N, C].
        '''
        f_out = []
        channel_idx = 0
        N = f_in.shape[0]
        for degree_idx, (mul, ir) in enumerate(self.irreps_in):
            feat = f_in[:, channel_idx:(channel_idx+mul*ir.dim)]
            feat = feat.reshape(N, mul, ir.dim)
            norm = feat.norm(dim=-1).clamp(min=self.NORM_CLAMP)
            new_norm = self.layer_norms[degree_idx](norm)
            
            #if not ir.is_scalar():
            #    new_norm = self.relu(new_norm)
            
            norm = norm.reshape(N, mul, 1)
            new_norm = new_norm.reshape(N, mul, 1)
            feat = feat * new_norm / norm
            feat = feat.reshape(N, -1)
            f_out.append(feat)
            
            channel_idx += mul * ir.dim
        
        f_out = torch.cat(f_out, dim=-1)
        return f_out
    
    
    def __repr__(self):
        return '{}({}, eps={})'.format(self.__class__.__name__, 
            self.irreps_in, self.eps)
    

class EquivariantLayerNormV2(nn.Module):
    
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
        return f"{self.__class__.__name__}({self.irreps}, eps={self.eps})"


    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input, **kwargs):
        # batch, *size, dim = node_input.shape  # TODO: deal with batch
        # node_input = node_input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # node_input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the node_input batch slices this into separate graphs
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            #field = node_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            field = node_input.narrow(1, ix, mul*d)
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0 and ir.p == 1:
                # Compute the mean
                field_mean = torch.mean(field, dim=1, keepdim=True) # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean
                
            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)    

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]
            
            if self.affine:
                weight = self.affine_weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]
            
            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]
            
            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output


class EquivariantLayerNormV3(nn.Module):
    '''
        V2 + Centering for vectors of all degrees
    '''
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


    #@torch.autocast(device_type='cuda', enabled=False)
    def forward(self, node_input, **kwargs):
        
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            field = field.reshape(-1, mul, d) # [batch * sample, mul, repr]
            
            field_mean = torch.mean(field, dim=1, keepdim=True) # [batch, 1, repr]
            field = field - field_mean
                
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)    
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output
    

class EquivariantLayerNormV4(nn.Module):
    '''
        V3 + Learnable mean shift
    '''
    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        mean_shift = []
        for mul, ir in self.irreps:
            if ir.l == 0 and ir.p == 1:
                mean_shift.append(torch.ones(1, mul, 1))
            else:
                mean_shift.append(torch.zeros(1, mul, 1))
        mean_shift = torch.cat(mean_shift, dim=1)
        self.mean_shift = nn.Parameter(mean_shift)
        
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


    #@torch.autocast(device_type='cuda', enabled=False)
    def forward(self, node_input, **kwargs):
        
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0
        i_mean_shift = 0

        for mul, ir in self.irreps:  
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            field = field.reshape(-1, mul, d) # [batch * sample, mul, repr]
            
            field_mean = torch.mean(field, dim=1, keepdim=True) # [batch, 1, repr]
            field_mean = field_mean.expand(-1, mul, -1)
            mean_shift = self.mean_shift.narrow(1, i_mean_shift, mul)
            field = field - field_mean * mean_shift
            i_mean_shift += mul
                
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)    
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output
    

if __name__ == '__main__':
    
    torch.manual_seed(10)
    
    irreps_in = o3.Irreps('4x0e+2x1o+1x2e')
    ln = EquivariantLayerNorm(irreps_in, eps=1e-5)
    print(ln)
    
    inputs = irreps_in.randn(10, -1)
    ln.train()
    outputs = ln(inputs)
    
    # Check equivariant
    rot = -o3.rand_matrix()
    D = irreps_in.D_from_matrix(rot)
    
    outputs_before = ln(inputs @ D.T)
    outputs_after = ln(inputs) @ D.T
    
    print(torch.max(torch.abs(outputs_after - outputs_before)))
    
    ln2 = EquivariantLayerNormV4(irreps_in)
    outputs2 = ln2(inputs)
            