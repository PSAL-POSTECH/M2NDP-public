from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad
import configs

class AggregationKernel(NdpKernel):
  def __init__(self, num_m2ndps=1):
    super().__init__()
    self.num_m2ndp = num_m2ndps
    self.vector_size = 7168
    self.batch_size = 1
    self.input_addr = 0x730000000000
    self.output_addr = 0x030000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'aggregation_kernel'
    self.base_addr = self.input_addr
    self.bound = self.vector_size * configs.data_size * self.batch_size
    self.input_addrs = ([self.output_addr])
    self.input = np.random.randn(self.batch_size, 1, self.vector_size).astype(np.float32)
    self.initial_output = np.random.randn(self.batch_size, 1, self.vector_size).astype(np.float32)
    self.output = self.input + self.initial_output
  
  def make_kernel(self):
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, (x1)\n' # output_addr
    template += f'vle32.v v1, (ADDR)\n' # v1 = input
    template += f'add x2, x2, OFFSET\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vsll.vi v2, v2, 2\n' # 4byte
    template += f'vamoaddei32.v x0, (x2), v2, v1\n' # output += input
    return template
  
  def make_input_map(self, i=0):
   return make_memory_map([
        (self.input_addr, self.input.reshape(-1)),
        (self.output_addr, self.initial_output.reshape(-1))
    ])
  
  def make_output_map(self, i=0):
    return make_memory_map([
        (self.input_addr, self.input.reshape(-1)),
        (self.output_addr, self.output.reshape(-1))])
