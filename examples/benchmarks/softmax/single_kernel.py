from typing import Any
import numpy as np
import os
import math
from utils.utils import NdpKernel, make_memory_map, pad8
import configs


class SingleKenrnelSoftMax(NdpKernel):
  def __init__(self):
    super().__init__()
    self.tensor_shape = [16, 1024] # number of heads, sequence length
    self.input_data_addr = 0x800000000000
    self.output_data_addr = 0x810000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = 'single_kernel_softmax'
    self.base_addr = self.input_data_addr
    self.bound = self.tensor_shape[0] * self.tensor_shape[1] * configs.data_size
    self.input_addrs = [self.output_data_addr, self.tensor_shape[0], self.tensor_shape[1]]
    self.stride = self.tensor_shape[1] * configs.data_size
    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")
    self.input_data = np.random.rand(self.tensor_shape[0], self.tensor_shape[1]).astype(data_type)
    max_vals = np.max(self.input_data, axis=-1, keepdims=True)
    inputs = (self.input_addrs - max_vals.astype(np.float32))
    outputs = np.exp(inputs)
    reduce_outputs = np.sum(outputs, axis=-1, keepdims=True).astype(np.float32)
    self.output_data = (outputs / reduce_outputs).astype(data_type)
  
  def make_kernel(self):
    data_size = configs.data_size
    num_ndp = configs.num_ndp
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    # Initializer
    # 1. Initialize accmulators
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'li x3, {configs.spad_addr}\n'
    template += f'ld x4, 8(x3)\n' # number of heads
    template += f'li x5, {num_ndp}\n' # x5 = num_ndp
    template += f'div x4, x4, x5\n' # x4 = number of heads / num_ndp 
    template += f'li x5, 0\n' # x5 = 0 iter
    template += f'li x6, 0\n' # x6 = 0
    template += f'addi x7, x3, {configs.packet_size}\n' # x7 = spad_addr + packet_size
    template += f'.LOOP1\n'
    template += f'sw x5, (x7)\n' # max value = 0
    template += f'sw x5, 4(x7)\n' # acc value = 0
    template += f'addi x5, x5, 1\n'
    template += f'addi x7, x7, 8\n'
    template += f'blt x5, x4, .LOOP1\n'
    template += f'muli x8, x4, {num_ndp//2}\n'  # x8 = number of heads * num_ndp /2
    template += f'sw x8, 24(x3)\n' # x3 = spad_addr + 24
    template += 'KERNELBODY:\n'
    # 2. Reduce max
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'li x3, {configs.spad_addr}\n'
    template += f'ld x4, 8(x3)\n' # number of heads'
    template += f'ld x5, 24(x3)\n' # x5 = number of heads * num_ndp /2
    template += f'div x5, OFFSET, x5\n' # x5 = OFFSET / (number of heads * num_ndp) * 2
    template += f'vle{data_size*8}.v v1, (ADDR)\n' # data
    template += f'vredmax.vs v2, v1\n' # v2 = max value
    template += f'vmv.x.s x4, v2\n' # x4 = max value
    template += f'add x3, x3, x5\n'
    template += f'sw x4, (x3)\n'
    template += f'KERNELBODY:\n'
    # 3. Reduce sum
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'li x3, {configs.spad_addr}\n'
    template += f'ld x4, 8(x3)\n'
    template += f'ld x5, 24(x3)\n'
    template += f'div x5, OFFSET, x5\n'
    template += f'vle{data_size*8}.v v1, (ADDR)\n'
    template += f'add x5, x3, x5\n'
    template += f'lw x6, (x6)\n'
    template += f'vsub.vx v2, v1, x6\n'
    template += f'vfexp.v v2, v2\n'
    template += f'vredsum.vs v2, v2\n'
    template += f'vmv.x.s x4, v2\n'
    template += f'sw x4, 4(x6)\n'
    template += f'KERNELBODY:\n'
    # 4. Calc softmax
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'li x3, {configs.spad_addr}\n'
    template += f'ld x4, 8(x3)\n'
    template += f'ld x5, 24(x3)\n'
    template += f'div x5, OFFSET, x5\n'
    template += f'vle{data_size*8}.v v1, (ADDR)\n'
    template += f'add x5, x3, x5\n'
    template += f'lw x6, (x6)\n'
    template += f'lw x7, 4(x6)\n'
    template += f'vsub.vx v2, v1, x7\n'
    template += f'vfexp.v v2, v2\n'
    template += f'vdiv.vv v3, v2, v1\n'
    template += f'vse{data_size*8}.v v3, (ADDR)\n'
    return template
  

  def make_input_map(self):
    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")
    return make_memory_map([(self.input_data_addr, self.input_data), 
                            (self.output_data_addr, np.zeros(self.tensor_shape, dtype=data_type))])
  
  def make_output_map(self):
    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")
    return make_memory_map([(self.input_data_addr, self.input_data), 
                            (self.output_data_addr, self.output_data)])
  
  
    

