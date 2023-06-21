'''
    Rescale output and weights of tensor product.
'''

import torch
import e3nn
from e3nn import o3

from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import collections
from e3nn.math import perm


class TensorProductRescale(torch.nn.Module):
    def __init__(self,
        irreps_in1, irreps_in2, irreps_out,
        instructions,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias
        
        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        self.tp = o3.TensorProduct(irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2, irreps_out=self.irreps_out, 
            instructions=instructions, normalization=normalization,
            internal_weights=internal_weights, shared_weights=shared_weights, 
            path_normalization='none')
        
        self.init_rescale_bias()
    
    
    def calculate_fan_in(self, ins):
        return {
            'uvw': (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            'uvu': self.irreps_in2[ins.i_in2].mul,
            'uvv': self.irreps_in1[ins.i_in1].mul,
            'uuw': self.irreps_in1[ins.i_in1].mul,
            'uuu': 1,
            'uvuv': 1,
            'uvu<v': 1,
            'u<vw': self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
        }[ins.connection_mode]
        
        
    def init_rescale_bias(self) -> None:
        
        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        # Determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()
        
        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split('+')]
        self.irreps_bias_parity = [irrep_str[-1] for irrep_str in str(self.irreps_bias).split('+')]
        self.irreps_bias_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(self.irreps_bias).split('+')]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if self.irreps_bias_orders[slice_idx] == 0 and self.irreps_bias_parity[slice_idx] == 'e':
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype))
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)
       
        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                            fan_in if slice_idx in slices_fan_in.keys() else fan_in)
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)
                
            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    #else:
                    #    sqrt_k = 1.
                    #
                    #if self.rescale:
                        #weight.data.uniform_(-sqrt_k, sqrt_k)
                    #    weight.data.mul_(sqrt_k)
                    #self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            #for (out_slice_idx, out_slice, out_bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
            #    sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
            #    out_bias.uniform_(-sqrt_k, sqrt_k)
                

    def forward_tp_rescale_bias(self, x, y, weight=None):
        
        out = self.tp(x, y, weight)
        
        #if self.rescale and self.tp.internal_weights:
        #    for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
        #        out[:, slice] /= slice_sqrt_k
        if self.use_bias:
            for (_, slice, bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                #out[:, slice] += bias
                out.narrow(1, slice.start, slice.stop - slice.start).add_(bias)
        return out
        

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out
    

class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(self,
        irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        instructions = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(irreps_in1, irreps_in2, irreps_out,
            instructions=instructions,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        
    
class LinearRS(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True):
        super().__init__(irreps_in, o3.Irreps('1x0e'), irreps_out, 
            bias=bias, rescale=rescale, internal_weights=True, 
            shared_weights=True, normalization=None)
    
    def forward(self, x):
        y = torch.ones_like(x[:, 0:1])
        out = self.forward_tp_rescale_bias(x, y)
        return out
    

def irreps2gate(irreps):
    irreps_scalars = []
    irreps_gated = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            irreps_scalars.append((mul, ir))
        else:
            irreps_gated.append((mul, ir))
    irreps_scalars = o3.Irreps(irreps_scalars).simplify()
    irreps_gated = o3.Irreps(irreps_gated).simplify()
    if irreps_gated.dim > 0:
        ir = '0e'
    else:
        ir = None
    irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
    return irreps_scalars, irreps_gates, irreps_gated


class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    def __init__(self,
        irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = e3nn.nn.Activation(irreps_out, acts=[torch.nn.functional.silu])
        else:
            gate = e3nn.nn.Gate(
                irreps_scalars, [torch.nn.functional.silu for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.gate = gate
        
    
    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out
    
    
def sort_irreps_even_first(irreps):
    Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
    out = [(ir.l, -ir.p, i, mul) for i, (mul, ir) in enumerate(irreps)]
    out = sorted(out)
    inv = tuple(i for _, _, i, _ in out)
    p = perm.inverse(inv)
    irreps = o3.Irreps([(mul, (l, -p)) for l, p, _, mul in out])
    return Ret(irreps, p, inv)
        

if __name__ == '__main__':
    

    irreps_1 = o3.Irreps('32x0e+16x1o+8x2e')
    irreps_2 = o3.Irreps('4x0e+4x1o+4x2e')
    irreps_out = o3.Irreps('16x0e+8x1o+4x2e')
    
    irreps_mid = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps_1):
        for j, (_, ir_edge) in enumerate(irreps_2):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_out or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_mid)
                    irreps_mid.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
    irreps_mid = o3.Irreps(irreps_mid)
    irreps_mid, p, _ = irreps_mid.sort()

    instructions = [
        (i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions
    ]
    
    torch.manual_seed(0)
    tp = o3.TensorProduct(irreps_1, irreps_2, irreps_mid, instructions)
    
    torch.manual_seed(0)
    tp_rs = TensorProductRescale(irreps_1, irreps_2, irreps_mid, instructions, 
        bias=False, rescale=False)
    
    inputs_1 = irreps_1.randn(10, -1)
    inputs_2 = irreps_2.randn(10, -1)
    
    out_tp = tp.forward(inputs_1, inputs_2)
    out_tp_rs = tp_rs.forward(inputs_1, inputs_2)
    print('[TP] before rescaling difference: {}'.format(torch.max(torch.abs(out_tp - out_tp_rs))))
    
    tp_rs.rescale = True
    tp_rs.init_rescale_bias()
    out_tp_rs = tp_rs.forward(inputs_1, inputs_2)
    print('[TP] after rescaling difference: {}'.format(torch.max(torch.abs(out_tp - out_tp_rs))))
    
    # FullyConnectedTensorProduct
    torch.manual_seed(0)
    fctp = o3.FullyConnectedTensorProduct(irreps_1, irreps_2, irreps_out)
    torch.manual_seed(0)
    fctp_rs = FullyConnectedTensorProductRescale(irreps_1, irreps_2, irreps_out, 
        bias=False, rescale=False)
    
    out_fctp = fctp.forward(inputs_1, inputs_2)
    out_fctp_rs = fctp_rs.forward(inputs_1, inputs_2)
    print('[FCTP] before rescaling difference: {}'.format(torch.max(torch.abs(out_fctp - out_fctp_rs))))
    
    fctp_rs.rescale = True
    fctp_rs.init_rescale_bias()
    out_fctp_rs = fctp_rs.forward(inputs_1, inputs_2)
    print('[FCTP] after rescaling difference: {}'.format(torch.max(torch.abs(out_fctp - out_fctp_rs))))
    