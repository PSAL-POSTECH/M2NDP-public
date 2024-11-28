from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8, pad16
import configs
import torch

batch_size = 1
cols = 2560
rows = 1

class LayerNormKernel1(NdpKernel):
  def __init__(self):
    super().__init__()
    self.vector_size = cols * rows * batch_size
    self.node_num = cols * rows
    self.batch_size = batch_size
    self.input_info_addr = 0x800000000000
    self.input_data_addr = 0x810000000000
    self.output_pow_mean_addr = 0x820000000000
    self.output_mean_addr = 0x830000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'layernorm_kernel1'
    self.base_addr = self.input_data_addr
    self.bound = self.vector_size * configs.data_size
    self.input_addrs = [self.input_info_addr, self.output_pow_mean_addr, self.output_mean_addr, self.node_num, self.batch_size]

    input_data = np.random.rand(batch_size, cols, rows).astype(np.float32)
    pow_input_data = np.power(input_data, 2)
    input_info = [cols, rows]
    outputs = np.mean(input_data, axis=(1,2))
    pow_outputs = np.mean(pow_input_data, axis=(1,2))
    var_outputs = np.var(input_data, axis=(1,2))
    std_outputs = np.std(input_data, axis=(1,2))

    torch.layernorm = torch.nn.LayerNorm([cols, rows], elementwise_affine=False)
    torch_input_data = torch.tensor(input_data)
    #widen into float32
    torch_outputs = torch.layernorm(torch_input_data.float())
    torch_outputs = torch_outputs.detach().numpy()
    
    self.input_data = input_data #pad8(np.array(input_data, dtype=np.float32), constant_values=-1)
    self.input_info = pad8(np.array(input_info, dtype=np.int32), constant_values=-1)
    
    self.outputs = pad8(np.array(outputs, dtype=np.float32))
    self.pow_outputs = pad8(np.array(pow_outputs, dtype=np.float32))
    self.var_outputs = pad8(np.array(var_outputs, dtype=np.float32))
    self.std_outputs = pad8(np.array(std_outputs, dtype=np.float32))

    self.torch_outputs = torch_outputs

    print(torch_outputs)
  
  def make_kernel(self):
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, 8(x1)\n'
    template += f'ld x3, 16(x1)\n'
    
    template += f'vle32.v v1, (x2)\n' # load E[x^2]
    template += f'vle32.v v2, (x3)\n' # load E[x]
    
    template += f'vfmul.vv v14, v2, v2\n' # E[x]^2
    template += f'vfsub.vv v16, v1, v14\n' # E[x^2] - E[x]^2 : Var[x]
    template += f'vfsqrt.v v18, v16\n' # Std[x]

    template += f'vmv.v.i v0, 1\n'
    template += f'vfcvt.f.x.v v0, v0\n'
    template += f'vfdiv.vv v20, v2, v18\n' # E[x] / Std[x]
    template += f'vfdiv.vv v22, v0, v18\n' # 1 / Std[x]

    template += f'vse32.v v22, 64(x1)\n' # store 1 / Std[x]
    template += f'vse32.v v20, 128(x1)\n' # store E[x] / Std[x]

    
    template += f'KERNELBODY:\n'   

    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'flw f1, 64(x1)\n' # 1 / Std[x]
    template += f'flw f2, 128(x1)\n' # E[x] / Std[x]

    template += f'vle32.v v1, (ADDR)\n' # load x
    
    template += f'vfmv.v.f v2, f2\n'
    template += f'vfmsub.vf v1, f1, v2\n' # vfmsub.vf vd, rs1, vs2, vm => vd[i] = (f[rs1] * vd[i]) - vs2[i]

    template += f'vse32.v v1, (ADDR)\n'
    return template
  
  def make_input_map(self):
   return make_memory_map([
        (self.input_info_addr, self.input_info),
        (self.input_data_addr, self.input_data),
        (self.output_pow_mean_addr, self.pow_outputs),
        (self.output_mean_addr, self.outputs)
    ])
  
  def make_output_map(self):
    return make_memory_map([
        # (self.input_info_addr, self.input_info),
        # (self.input_data_addr, self.input_data),
        (self.input_data_addr, self.torch_outputs),
        # (self.output_addr, self.outputs)
        ])
  
