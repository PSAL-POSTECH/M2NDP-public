from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8, pad16
import configs

batch_size = 1
cols = 7168
rows = 1

# Mean Kernel for LayerNorm

class LayerNormKernel0(NdpKernel):
  def __init__(self):
    super().__init__()
    self.vector_size = cols * rows * batch_size
    self.node_num = cols * rows
    self.batch_size = batch_size
    self.input_info_addr = 0x800000000000
    self.input_data_addr = 0x810000000000
    self.output_pow_mean_addr = 0x880000000000
    self.output_mean_addr = 0x890000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'layernorm_kernel0'
    self.base_addr = self.input_data_addr
    self.bound = int(self.vector_size * configs.data_size / 2)
    self.input_addrs = [self.input_info_addr, self.output_mean_addr, self.output_pow_mean_addr, self.node_num, self.batch_size]

    input_data = np.random.rand(batch_size, cols, rows).astype(np.float16)
    pow_input_data = np.power(input_data, 2)
    input_info = [cols, rows]
    outputs = np.mean(input_data, axis=(1,2))
    pow_outputs = np.mean(pow_input_data, axis=(1,2))
    
    self.input_data = input_data #pad8(np.array(input_data, dtype=np.float32), constant_values=-1)
    self.input_info = pad8(np.array(input_info, dtype=np.int32), constant_values=-1)
    self.outputs = pad16(np.array(outputs, dtype=np.float16))
    self.pow_outputs = pad16(np.array(pow_outputs, dtype=np.float16))

    print(self.input_info)
    print(self.input_data)
    print(self.outputs)
    print(self.pow_outputs)
  
  def make_kernel(self):
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += 'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e32, m2\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'vmv.v.i v1, 0\n'
    template += f'vfcvt.f.x.v v2, v1\n'
    template += f'vse32.v v2, 64(x1)\n'
    template += f'vse32.v v2, 128(x1)\n'






    template += f'KERNELBODY:\n'

    template += f'vsetvli t0, a0, e16, m1\n'

    template += f'li x1, {configs.spad_addr}\n' # scratchpad memory address
    template += f'vle16.v v1, (ADDR)\n' # {x} float16
    template += f'vfwcvt.f.f.v v2, v1, v1\n' # {x} float32
    template += f'vfwmul.vv v4, v1, v1\n' # {x^2}

    template += f'vsetvli t0, a0, e32, m2\n'

    template += f'vmset.m v0\n'
    template += f'vid.v v6, v0\n'
    template += f'vmul.vi v6, v6, 4\n'

    template += f'addi x1, x1, {2 * configs.packet_size}\n' # for E[x]
    template += f'vamoaddei32.v x0, (x1), v6, v2\n'
    template += f'addi x1, x1, {2 * configs.packet_size}\n' # for E[x^2]
    template += f'vamoaddei32.v x0, (x1), v6, v4\n' # for E[x^2]

    template += f'li x1, {configs.spad_addr}\n'
    template += f'vle32.v v30, 64(x1)\n'





    template += f'FINALIZER:\n'

    template += f'vsetvli t0, a0, e32, m2\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'vle32.v v2, 64(x1)\n' # E[x]
    template += f'vle32.v v4, 128(x1)\n'
    template += f'ld x30, 8(x1)\n' # output_mean_addr
    template += f'ld x31, 16(x1)\n' # output_pow_mean_addr
    template += f'ld x2, 24(x1)\n' # element size each batch

    template += f'fmv.w.x f0, x2\n'

    template += f'vmv.v.i v0, 0\n'
    template += f'vfredosum.vs v6, v2, v0\n'
    template += f'vfredosum.vs v8, v4, v0\n' # for E[x^2]
    
    template += f'vfdiv.vf v10, v6, f0\n'
    template += f'vfdiv.vf v12, v8, f0\n'
    template += f'vfncvt.f.f.w v14, v10\n' # v10 (float32) -> v14 (float16)
    template += f'vfncvt.f.f.w v16, v12\n' # v12 (float32) -> v16 (float16)

    template += f'vsetvli t0, a0, e16, m1\n'
    template += f'vfmv.f.s f1, v14\n'
    template += f'vfmv.f.s f2, v16\n' # for E[x^2]

    template += f'vsetvli t0, a0, e32, m1\n'    
    template += f'famoaddh x0, f1, (x30)\n'
    template += f'famoaddh x0, f2, (x31)\n' # for E[x^2]

    return template
  
  def make_input_map(self):
   return make_memory_map([
        (self.input_info_addr, self.input_info),
        (self.input_data_addr, self.input_data),
    ])
  
  def make_output_map(self):
    return make_memory_map([
        # (self.input_info_addr, self.input_info),
        # (self.input_data_addr, self.input_data),
        (self.output_pow_mean_addr, self.pow_outputs),
        (self.output_mean_addr, self.outputs)
      ])
  
