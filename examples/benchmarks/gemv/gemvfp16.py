from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad
import configs

class GemvFP16Kernel(NdpKernel):
  def __init__(self):
    super().__init__()
    self.vector_size = 256
    self.weight_shape = (self.vector_size, 128)
    self.batch_size = 3
    self.output_size = self.weight_shape[1] # 1x256 * 256x128 = 1x128
    self.input_addr = 0x800000000000
    self.weight_addr = 0x810000000000
    self.output_addr = 0x820000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'gemv_kernel'
    self.base_addr = self.output_addr
    self.bound = self.output_size * configs.data_size * self.batch_size
    self.input_addrs = [self.input_addr, self.weight_addr, self.weight_shape[0], self.weight_shape[1], self.batch_size]
    self.input = np.random.randn(self.batch_size, 1, self.vector_size).astype(np.float16)
    self.weight = np.random.randn(self.batch_size, self.weight_shape[0], self.weight_shape[1]).astype(np.float16)
    self.output = (self.input @ self.weight)

  
  def make_kernel(self):
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'vsetvli t0, a0, e16, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, (x1)\n' # input_addr
    template += f'ld x3, 16(x1)\n' # vector_size
    template += f'ld x4, 32(x1)\n' # batch_size
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # int64 = 8 bytes, 1 for next address
    template += f'li x5, 0\n' # batch index
    template += f'.LOOP0\n'
    template += f'li x6, 0\n' # vector index
    template += f'.LOOP1\n'
    template += f'vle16.v v1, (x2)\n'
    template += f'vse16.v v1, (x1)\n'
    template += f'addi x1, x1, {configs.packet_size}\n'
    template += f'addi x2, x2, {configs.packet_size}\n'
    template += f'addi x6, x6, 16\n' # 256 / float 16 = 16 elements in a row
    template += f'bne x6, x3, .LOOP1\n'
    template += f'addi x5, x5, 1\n'
    template += f'bne x5, x4, .LOOP0\n'
    
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e16, m1\n'
    template += f'vmv.v.i v5, 0\n' # initialize with 0
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, 8(x1)\n' # weight_addr
    template += f'ld x3, 16(x1)\n' # vector_size
    template += f'ld x9, 24(x1)\n' # output_size
    template += f'vmv.v.i v4, -1\n'
    template += f'vfcvt.f.x.v v4, v4\n'
    template += f'li x8, 16\n' # 256 / float 16 = 16 elements in a row
    template += f'muli x4, OFFSET, x3\n' # OFFSET / 32 * (16 elements in a row * 256(vectorsize) elements for 1 output element) * 2byte -> weight offset
    template += f'add x4, x4, x2\n' # weight_addr + weight offset
    template += f'addi x1, x1, {configs.packet_size * 2}\n'
    template += f'srli x11, OFFSET, 1\n' # OFFSET / 2Byte
    template += f'div x11, x11, x9\n' # current batch index
    template += f'mul x10, x11, x3\n' # current batch index * vector_size
    template += f'slli x10, x10, 1\n' # vector_size * 2byte
    template += f'li x5, 0\n' # output index
    template += f'.LOOP2\n'
    template += f'addi x6, x1, 0\n'
    template += f'addi x7, x3, 0\n'
    template += f'add x6, x6, x10\n' # input_addr + current batch index * vector_size
    
    template += f'vmv.v.i v3, 0\n' # initialize output with 0
    template += f'vfcvt.f.x.v v3, v3\n'
    template += f'.LOOP3\n'
    template += f'vle16.v v1, (x6)\n' # input
    template += f'vle16.v v2, (x4)\n' # weight
    template += f'vfmacc.vv v3, v1, v2\n' # output += input * weight
    template += f'addi x6, x6, {configs.packet_size}\n'
    template += f'addi x4, x4, {configs.packet_size}\n'
    template += f'addi x7, x7, -16\n'
    template += f'bgtz x7, .LOOP3\n'
    template += f'vmv.v.i v5, 0\n' # initialize with 0
    template += f'vfredosum.vs v3, v3, v5\n' # reduce sum
    template += f'vfmv.f.s f0, v3\n' # get float value from vector
    template += f'csrw vstart, x5\n'
    template += f'vfmv.s.f v4, f0\n' # v4[i] = f0
    template += f'addi x5, x5, 1\n'
    template += f'bne x5, x8, .LOOP2\n'
    template += f'vse16.v v4, (ADDR)\n'
    return template
  
  def make_input_map(self):
   return make_memory_map([
        (self.input_addr, self.input),
        (self.weight_addr, np.transpose(self.weight, (0, 2, 1)))
    ])
  
  def make_output_map(self):
    return make_memory_map([
        (self.input_addr, self.input),
        (self.weight_addr, np.transpose(self.weight, (0, 2, 1))),
        (self.output_addr, self.output)])
