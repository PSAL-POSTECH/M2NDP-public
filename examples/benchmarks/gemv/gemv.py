from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad
import configs

class GemvKernel(NdpKernel):
  def __init__(self, num_m2ndps=1):
    super().__init__()
    self.num_m2ndp = num_m2ndps
    self.vector_size = 512
    self.output_size = 512
    self.vector_per_m2ndp = int(self.vector_size / self.num_m2ndp)
    self.batch_size = 1
    self.unroll_level = 4
    self.weight_size = self.vector_per_m2ndp * self.output_size
    self.addr_offset = 0x100000000000
    self.input_base_addr = 0x010000000000
    self.weight_base_addr = 0x020000000000
    self.input_addr = []
    self.weight_addr = []
    for i in range(num_m2ndps):
      self.input_addr.append(self.input_base_addr + i * self.addr_offset)
      self.weight_addr.append(self.weight_base_addr + i * self.addr_offset)
    # self.input_addr = self.input_base_addr + num_m2ndps * self.addr_offset
    # self.weight_addr = self.weight_base_addr + num_m2ndps * self.addr_offset
    self.output_addr = 0x030000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'gemv_kernel'
    self.base_addrs = self.weight_addr
    self.bound = int(self.vector_per_m2ndp * self.output_size * (configs.data_size / 2) * self.batch_size / 16)
    self.input_addrs_list = []
    for i in range(num_m2ndps):
      self.input_addrs_list.append([self.input_addr[i], self.output_addr, self.weight_addr[i], self.vector_per_m2ndp, self.output_size, self.batch_size])
    self.input = np.random.randn(self.batch_size, 1, self.vector_size).astype(np.float16)
    self.weight = np.random.randn(self.batch_size, self.vector_size, self.output_size).astype(np.float16)
    self.partial_acc_output = []
    self.partial_output = np.zeros((self.batch_size, 1, self.output_size)).astype(np.float16)
    self.partial_acc_output.append(self.partial_output.copy())
    for i in range(num_m2ndps):
      self.partial_output += (self.input[:, :, i * self.vector_per_m2ndp : (i + 1) * self.vector_per_m2ndp].astype(np.float32) @ self.weight[:, i * self.vector_per_m2ndp : (i + 1) * self.vector_per_m2ndp, :].astype(np.float32)).astype(np.float16)
      self.partial_acc_output.append(self.partial_output.copy())
    self.output = (self.input.astype(np.float32) @ self.weight.astype(np.float32)).astype(np.float16)

  
  def make_kernel(self):
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e16, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, (x1)\n' # input_addr
    template += f'ld x3, 24(x1)\n' # vector_length
    template += f'ld x9, 32(x1)\n' # output_length
    # template += f'ld x4, 40(x1)\n' # batch_size ## ucnomment if consider batch
    template += f'slli x7, x3, 1\n' # vector_length * 2byte = vector_size
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # next to argument region
    template += f'add x7, x1, x7\n' # spad_addr + vector_size
    # template += f'li x5, 0\n' # batch index
    # template += f'.LOOP0\n'
    template += f'li x6, 0\n' # vector index
    template += f'.LOOP1\n'
    for i in range(1, self.unroll_level + 1):
      template += f'vle16.v v{i}, (x2)\n'
      template += f'vse16.v v{i}, (x1)\n'
      template += f'addi x2, x2, {configs.packet_size}\n'
      template += f'addi x1, x1, {configs.packet_size}\n'
    template += f'addi x6, x6, {16 * self.unroll_level}\n' # 256 / float 16 = 16 elements in a row
    template += f'blt x6, x3, .LOOP1\n'
    template += f'li x8, 0\n' # output index
    template += f'vsetvli t0, a0, e32, m2\n'
    template += f'vmv.v.i v{self.unroll_level + 1}, 0\n' # initialize with 0
    template += f'vfcvt.f.x.v v{self.unroll_level + 1}, v{self.unroll_level + 1}\n'
    template += f'.LOOP2\n'
    for i in range(self.unroll_level):
      template += f'vse32.v v{self.unroll_level + 1}, (x7)\n'
      template += f'addi x7, x7, {configs.packet_size * 2}\n' # double register
    template += f'addi x8, x8, {16 * self.unroll_level}\n' # 256 / float 16 = 16 elements in a row
    template += f'blt x8, x9, .LOOP2\n'
    # template += f'addi x5, x5, 1\n'
    # template += f'bne x5, x4, .LOOP0\n'

    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e16, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, 16(x1)\n' # weight_addr
    template += f'ld x4, 32(x1)\n' # output_length
    template += f'ld x12, 24(x1)\n' # vector_length
    template += f'srli x3, OFFSET, 5\n' # OFFSET / 32 = thread idx
    template += f'srli x5, x4, 4\n'
    template += f'div x6, x3, x5\n' # thread.y
    template += f'slli x8, x4, 1\n' # output_length * 2byte = output_size
    template += f'slli x9, x8, 4\n'
    template += f'rem x7, x3, x5\n' # thread.x
    template += f'slli x10, x7, 5\n' # thread.x * 32(packet_size)
    template += f'mul x9, x6, x9\n' # thread.y * output_size
    template += f'add x11, x9, x10\n' # weight_addr_offset
    template += f'add x2, x2, x11\n' # weight_addr + weight_addr_offset
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # next to argument region
    template += f'slli x14, x6, 5\n' # thread.y * 32(packet_size)
    template += f'add x14, x1, x14\n' # spad_addr + thread.y * 32(packet_size)
    template += f'vle16.v v1, (x14)\n' # load input
    template += f'vsetvli t0, a0, e32, m2\n'
    template += f'vmv.v.i v3, 0\n' # initialize with 0
    template += f'vfcvt.f.x.v v3, v3\n'
    template += f'vsetvli t0, a0, e16, m1\n'
    for i in range(5, 21):
      template += f'vle16.v v{i}, (x2)\n' # load weight
      template += f'add x2, x2, x8\n'
    template += f'slli x4, x12, 1\n' # vector_length * 2byte = vector_size
    template += f'add x12, x1, x4\n' # spad_addr + vector_size
    template += f'slli x10, x10, 1\n' # thread.x * 64(packet_size * double register)
    template += f'add x12, x12, x10\n' # spad_addr + vector_size + thread.x * 32(packet_size)
    for i in range(1, 17):
      template += f'vfmv.f.s f{i}, v1\n' # input[i]
      template += f'vfwmacc.vf v3, v{i+4}, f{i}\n' # output += input[i] * weight
      template += f'vslide1down.vx v1, v1, x0\n'
    template += f'vsetvli t0, a0, e32, m2\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v4, v0\n'
    template += f'vsll.vi v4, v4, 2\n' # 4byte
    template += f'vamoaddei32.v x0, (x12), v4, v3\n'

    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e32, m2\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x3, 24(x1)\n' # vector_length
    template += f'ld x2, 8(x1)\n' # output_addr
    template += f'ld x4, 32(x1)\n' # output_length
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # next to argument region
    template += f'slli x5, x3, 1\n' # vector_length * 2byte
    template += f'add x5, x1, x5\n' # spad_addr + vector_size = output_base_addr
    template += f'li x6, 0\n' # output index
    template += f'vsetvli t0, a0, e16, m1\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v31, v0\n'
    template += f'vsll.vi v31, v31, 1\n' # 4byte
    template += f'.LOOP4\n'
    template += f'vsetvli t0, a0, e32, m2\n'
    for i in range(self.unroll_level):
      template += f'vle32.v v{i + 1}, (x5)\n'
      template += f'addi x5, x5, {configs.packet_size * 2}\n'
    for i in range(self.unroll_level):
      template += f'vfncvt.f.f.w v{i + self.unroll_level + 1}, v{i + 1}\n' # narrow
    template += f'vsetvli t0, a0, e16, m1\n'
    for i in range(self.unroll_level):
      template += f'vamoaddei16.v x0, (x2), v31, v{i + self.unroll_level + 1}\n'
      template += f'addi x2, x2, {configs.packet_size}\n'
    template += f'addi x6, x6, {16 * self.unroll_level}\n' # 256 / float 16 = 16 elements in a row
    template += f'blt x6, x4, .LOOP4\n'
    return template
  
  def make_input_map(self, i=0):
   return make_memory_map([
        (self.input_addr[i], self.input.reshape(-1)[i * self.vector_per_m2ndp : (i + 1) * self.vector_per_m2ndp]),
        (self.weight_addr[i], self.weight.reshape(-1)[i * self.weight_size : (i + 1) * self.weight_size]),
        (self.output_addr, self.partial_acc_output[i].reshape(-1))
    ])
  
  def make_output_map(self, i=0):
    return make_memory_map([
        (self.input_addr[i], self.input.reshape(-1)[i * self.vector_per_m2ndp : (i + 1) * self.vector_per_m2ndp]),
        (self.weight_addr[i], self.weight.reshape(-1)[i * self.weight_size : (i + 1) * self.weight_size]),
        (self.output_addr, self.partial_acc_output[i+1].reshape(-1))])
