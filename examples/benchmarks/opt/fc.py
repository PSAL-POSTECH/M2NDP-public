from typing import Any
import numpy as np
import os
import json
from utils.utils import *
import configs
import math
class OptGemv(NdpKernel):
  def __init__(self, input_size, weight_size, fuse_bias=False):
    super().__init__()
    self.batch_size = 1
    self.unroll_level = 16
    self.vector_size = input_size
    self.output_size = weight_size
    self.input_addr = 0x800000000000
    self.weight_addr = 0x810000000000
    self.bias_addr = 0x820000000000
    self.output_addr = 0x830000000000
    self.sync = 0
    self.kernel_id = 0
    self.base_addr = self.weight_addr
    self.bound = self.vector_size * self.output_size * configs.data_size * self.batch_size // 8
    self.input_addrs = [self.input_addr, self.output_addr, self.weight_addr, self.vector_size, self.output_size, self.batch_size]
    self.input = np.random.randn(self.batch_size, 1, self.vector_size).astype(np.float32)
    self.weight = np.random.randn(self.batch_size, self.vector_size, self.output_size).astype(np.float32)
    self.bias = np.random.randn(self.batch_size, self.output_size).astype(np.float32)
    self.initialized_output = np.zeros((self.batch_size, 1, self.output_size)).astype(np.float32)
    self.output = (self.input @ self.weight) #+ self.bias
    self.smem_size = input_size * configs.data_size + weight_size * configs.data_size

  def make_kernel(self):
    ndp_unit_log2 = int(math.log2(configs.ndp_units))
    stride_log2 = int(math.log2(configs.stride))
    packet_size_log2 = int(math.log2(configs.packet_size))
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x13, (x1)\n' # input_addr
    template += f'ld x3, 24(x1)\n' # vector_length
    template += f'ld x9, 32(x1)\n' # output_length
    # template += f'ld x4, 40(x1)\n' # batch_size ## ucnomment if consider batch
    template += f'slli x3, x3, 2\n' # vector_length * 4byte = vector_size
    template += f'slli x9, x9, 2\n' # output_length * 4byte = output_size
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # next to argument region
    template += f'add x7, x1, x3\n' # spad_addr + vector_size
    # template += f'li x5, 0\n' # batch index
    # template += f'.LOOP0\n'
    template += f'andi x4, x2, {configs.stride -1}\n' # OFFSET & (stride - 1)
    template += f'srli x5, x2, {stride_log2 + ndp_unit_log2}\n' # OFFSET / stride / ndp_units
    template += f'slli x5, x5, {stride_log2 }\n' # OFFSET / stride / ndp_units * stride
    template += f'or x4, x5, x4\n' # OFFSET / stride / ndp_units * stride + OFFSET & (stride - 1)
    template += f'bge x4, x3, .SKIP0\n'
    template += f'add x13, x13, x4\n'
    template += f'add x1, x1, x4\n'
    template += f'vle32.v v1, (x13)\n'
    template += f'vse32.v v1, (x1)\n'
    template += f'.SKIP0\n'
    template += f'li x8, 0\n' # output index
    template += f'vmv.v.i v2, 0\n' # initialize with 0
    template += f'vfcvt.f.x.v v2, v2\n'
    template += f'bge x4, x9, .SKIP1\n'
    template += f'add x7, x7, x4\n'
    template += f'vse32.v v2, (x7)\n'
    template += f'.SKIP1\n'

    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x13, 16(x1)\n' # weight_addr
    template += f'ld x4, 32(x1)\n' # output_length
    template += f'ld x12, 24(x1)\n' # vector_length
    template += f'srli x3, x2, 5\n' # OFFSET / 32 = thread idx
    template += f'srli x5, x4, 3\n'
    template += f'div x6, x3, x5\n' # thread.y
    template += f'slli x8, x4, 2\n' # output_length * 4byte = output_size
    template += f'slli x9, x8, 3\n'
    template += f'rem x7, x3, x5\n' # thread.x
    template += f'slli x10, x7, 5\n' # thread.x * 32(packet_size)
    template += f'mul x9, x6, x9\n' # thread.y * output_size
    template += f'add x11, x9, x10\n' # weight_addr_offset
    template += f'add x13, x13, x11\n' # weight_addr + weight_addr_offset
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # next to argument region
    template += f'slli x14, x6, 5\n' # thread.y * 32(packet_size)
    template += f'add x14, x1, x14\n' # spad_addr + thread.y * 32(packet_size)
    template += f'vle32.v v1, (x14)\n' # load input
    template += f'vmv.v.i v3, 0\n' # initialize with 0
    template += f'vfcvt.f.x.v v3, v3\n'
    for i in range(5, 5 + 8):
      template += f'vle32.v v{i}, (x13)\n' # load weight
      template += f'add x13, x13, x8\n'
    template += f'slli x4, x12, 2\n' # vector_length * 4byte = vector_size
    template += f'add x12, x1, x4\n' # spad_addr + vector_size
    template += f'add x12, x12, x10\n' # spad_addr + vector_size + thread.x * 32(packet_size)
    for i in range(1, 1 + 8):
      template += f'vfmv.f.s f{i}, v1\n' # input[i]
      template += f'vfmacc.vf v3, v{i+4}, f{i}\n' # output += input[i] * weight
      template += f'vslide1down.vx v1, v1, x0\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v4, v0\n'
    template += f'vsll.vi v4, v4, 2\n' # 4byte
    template += f'vamoaddei32.v x0, (x12), v4, v3\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x4, 32(x1)\n' # output_length
    template += f'slli x4, x4, 2\n' # output_length * 4byte = output_size
    template += f'andi x6, x2, {configs.stride -1}\n' # OFFSET & (stride - 1)
    template += f'srli x7, x2, {stride_log2 + ndp_unit_log2}\n' # OFFSET / stride / ndp_units
    template += f'slli x7, x7, {stride_log2 }\n' # OFFSET / stride / ndp_units * stride
    template += f'or x6, x7, x6\n' # OFFSET / stride / ndp_units * stride + OFFSET & (stride - 1)
    template += f'bge x6, x4, .SKIP2\n'
    template += f'ld x3, 24(x1)\n' # vector_length
    template += f'ld x13, 8(x1)\n' # output_addr
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # next to argument region
    template += f'slli x5, x3, 2\n' # vector_length * 4byte
    template += f'add x5, x1, x5\n' # spad_addr + vector_size = output_base_addr
    template += f'add x5, x5, x6\n'
    template += f'add x13, x13, x6\n'
    template += f'vle32.v v1, (x5)\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v31, v0\n'
    template += f'vsll.vi v31, v31, 2\n' # 4byte
    template += f'vamoaddei32.v x0, (x13), v31, v1\n'
    template += f'.SKIP2\n'

    return template

  def make_input_map(self):
    return make_memory_map([
          (self.input_addr, self.input.reshape(-1)),
          (self.weight_addr, self.weight.reshape(-1)),
          (self.output_addr, self.initialized_output.reshape(-1))
      ])

  def make_output_map(self):
    return make_memory_map([
        (self.input_addr, self.input.reshape(-1)),
        (self.weight_addr, self.weight.reshape(-1)),
        (self.output_addr, self.output.reshape(-1))])


class OptGemvOutputMap(NdpKernel):
  def __init__(self, input_size, weight_size, fuse_bias=False):
    super().__init__()
    self.batch_size = 1
    self.unroll_level = 8
    self.vector_size = input_size
    self.output_size = weight_size
    self.input_addr = 0x800000000000
    self.weight_addr = 0x810000000000
    self.bias_addr = 0x820000000000
    self.output_addr = 0x830000000000
    self.sync = 0
    self.kernel_id = 0
    self.base_addr = self.output_addr
    self.bound = self.output_size * configs.data_size
    # self.bound = self.output_size * configs.packet_size
    # self.input_addrs = [self.input_addr, self.weight_addr, self.vector_size, self.output_size, self.batch_size]
    self.input_addrs = [self.input_addr, self.weight_addr, self.output_addr,  self.vector_size, self.batch_size]
    self.input = np.random.randn(self.batch_size, 1, self.vector_size).astype(np.float32)
    # self.input = np.ones((self.batch_size, 1, self.vector_size)).astype(np.float32)
    self.weight = np.random.randn(self.batch_size, self.vector_size, self.output_size).astype(np.float32)
    self.weight = np.random.randn(self.batch_size, self.output_size, self.vector_size).astype(np.float32)
    # self.weight = np.ones((self.batch_size, self.output_size, self.vector_size)).astype(np.float32)

    self.weight_t = self.weight.transpose(0, 2, 1)
    # self.weight = np.ones((self.batch_size, self.vector_size, self.output_size)).astype(np.float32)
    self.bias = np.random.randn(self.batch_size, self.output_size).astype(np.float32)
    self.initialized_output = np.zeros((self.batch_size, 1, self.output_size)).astype(np.float32)
    self.output = (self.input @ self.weight_t) #+ self.bias
    self.smem_size = input_size * configs.data_size

  def make_input_map(self):
    return make_memory_map([
          (self.input_addr, self.input.reshape(-1)),
          (self.weight_addr, self.weight.reshape(-1)),
          (self.output_addr, self.initialized_output.reshape(-1))
      ])

  def make_output_map(self):
    return make_memory_map([
        (self.input_addr, self.input.reshape(-1)),
        (self.weight_addr, self.weight.reshape(-1)),
        (self.output_addr, self.output.reshape(-1))])

  def make_kernel(self):
    ndp_unit_log2 = int(math.log2(configs.ndp_units))
    stride_log2 = int(math.log2(configs.stride))
    packet_size_log2 = int(math.log2(configs.packet_size))
    elems_per_packet = configs.packet_size // configs.data_size
    elems_per_packet_log2 = int(math.log2(elems_per_packet))
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x8, {configs.spad_addr}\n'
    template += f'ld x9, (x8)\n' # input_addr
    template += f'ld x3, 24(x8)\n' # vector_length
    template += f'slli x3, x3, 2\n' # vector_length * 4byte = vector_size
    template += f'andi x4, x2, {configs.stride -1}\n' # OFFSET & (stride - 1)
    template += f'srli x5, x2, {stride_log2 + ndp_unit_log2}\n' # OFFSET / stride / ndp_units
    template += f'slli x5, x5, {stride_log2 }\n' # OFFSET / stride / ndp_units * stride
    template += f'or x4, x5, x4\n' # OFFSET / stride / ndp_units * stride + OFFSET & (stride - 1)
    template += f'slli x4, x4, {ndp_unit_log2}\n'
    template += f'addi x8, x8, {configs.packet_size * 2}\n' # next to argument region
    template += f'bge x4, x3, .SKIP0\n'
    template += f'add x9, x9, x4\n'
    template += f'add x8, x8, x4\n'
    for i in range(configs.ndp_units):
      template += f'vle32.v v{i}, (x9)\n'
      template += f'addi x9, x9, {configs.packet_size}\n'
    for i in range(configs.ndp_units):
      template += f'vse32.v v{i}, (x8)\n'
      template += f'addi x8, x8, {configs.packet_size}\n'
    template += f'.SKIP0\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x8, {configs.spad_addr}\n'
    template += f'ld x9, 8(x8)\n' # weight_addr
    template += f'ld x3, 24(x8)\n' # vector_length
    template += f'ld x7, 16(x8)\n' # output_addr
    template += f'vmv.v.i v1, 0\n' # output initialize with 0
    # template += f'srli x4, x2, {elems_per_packet_log2}\n' # OFFSET / 8
    template += f'vfcvt.f.x.v v1, v1\n'
    for unroll in range(1, self.unroll_level):
      template += f'vmv.v.v v{unroll + 1}, v1\n'
    template += f'mul x6, x3, x2\n' # vector_length * OFFSET / 8
    template += f'add x12, x9, x6\n' # weight_addr + OFFSET
    template += f'addi x8, x8, {configs.packet_size * 2}\n' # next to argument region
    template += f'slli x3, x3, 2\n' # vector_length * 4byte
    # template += f'add x13, x12, x3\n'
    template += f'add x4, x8, x3\n'
    template += f'.LOOP1\n'
    template += f'addi x13, x12, 0\n'
    for unroll in range(self.unroll_level):
      template += f'vle32.v v1{unroll+1}, {unroll * configs.packet_size}(x8)\n' #load input
    for i in range(elems_per_packet):
      if i > 0:
        template += f'add x13, x13, x3\n'
      for unroll in range(self.unroll_level):
        template += f'vle32.v v2{unroll+1}, {unroll * configs.packet_size}(x13)\n' #load weight
      for unroll in range(self.unroll_level):
        template += f'vfmacc.vv v{i+1}, v1{unroll+1}, v2{unroll+1}\n'
    template += f'addi x8, x8, {configs.packet_size * self.unroll_level}\n' # next input addr
    template += f'addi x12, x12, {configs.packet_size * self.unroll_level}\n' # next input addr
    template += f'blt x8, x4, .LOOP1\n'
    template += f'vmv.v.i v0, 0\n'
    for unroll in range(self.unroll_level):
      template += f'vfredosum.vs v{self.unroll_level - unroll}, v{self.unroll_level - unroll}, v0\n'
    for unroll in range(self.unroll_level - 1):
      template += f'vslide1up.vx v{self.unroll_level - unroll}, v{self.unroll_level - unroll}, x0\n'
      template += f'vadd.vv v{self.unroll_level - unroll - 1}, v{self.unroll_level - unroll -1}, v{self.unroll_level - unroll}\n'
    template += f'vse32.v v1, (ADDR)\n' # Store output to output_addr

    return template

class OptGemvOutputMapSplit(MultipleKernel):
  def __init__(self, input_size, weight_size, fuse_bias=False, repeat=1, split=4, unroll_level=8, input_unroll=32):
    per_ndp_count = get_min_per_ndp_unit_count(configs, weight_size * configs.data_size)
    per_ndp_count = max(1, per_ndp_count)
    elems_per_packet = configs.packet_size // configs.data_size
    if input_unroll == -1:
      for count in [1, 2, 4, 8 ,16, 32, 64, 128, 256]:
        input_repeat = (input_size // repeat) // (count * elems_per_packet)
        if input_repeat <= per_ndp_count:
          input_unroll = count * 2
          break
    print(input_unroll)
    super().__init__(repeat=repeat)
    self.batch_size = 1
    self.unroll_level = unroll_level
    self.split = split
    self.input_unroll = input_unroll
    self.vector_size = input_size
    self.output_size = weight_size
    self.input_addr = 0x800000000000
    self.weight_addr = 0x810000000000
    self.bias_addr = 0x820000000000
    self.output_addr = 0x830000000000
    self.sync = 0
    self.kernel_id = 0
    self.base_addr = self.output_addr
    self.base_offset = 0
    self.bound = self.output_size * configs.data_size * self.split
    # self.bound = self.output_size * configs.packet_size
    # self.input_addrs = [self.input_addr, self.weight_addr, self.vector_size, self.output_size, self.batch_size]
    self.input_addrs = [self.input_addr, self.weight_addr, self.output_addr,  self.vector_size, self.batch_size]
    self.input_addrs_offsets = [self.vector_size // self.repeat * configs.data_size,
                                self.vector_size // self.repeat * configs.data_size,
                                  0, 0, 0]
    self.input = np.random.randn(self.batch_size, 1, self.vector_size).astype(np.float32)
    # self.input = np.ones((self.batch_size, 1, self.vector_size)).astype(np.float32)
    self.weight = np.random.randn(self.batch_size, self.vector_size, self.output_size).astype(np.float32)
    self.weight = np.random.randn(self.batch_size, self.output_size, self.vector_size).astype(np.float32)
    # self.weight = np.ones((self.batch_size, self.output_size, self.vector_size)).astype(np.float32)

    self.weight_t = self.weight.transpose(0, 2, 1)
    # self.weight = np.ones((self.batch_size, self.vector_size, self.output_size)).astype(np.float32)
    self.bias = np.random.randn(self.batch_size, self.output_size).astype(np.float32)
    self.initialized_output = np.zeros((self.batch_size, 1, self.output_size)).astype(np.float32)
    self.output = (self.input @ self.weight_t) #+ self.bias
    self.smem_size = input_size * configs.data_size // repeat

  def make_input_map(self):
    return make_memory_map([
          (self.input_addr, self.input.reshape(-1)),
          (self.weight_addr, self.weight.reshape(-1)),
          (self.output_addr, self.initialized_output.reshape(-1))
      ])

  def make_output_map(self):
    return make_memory_map([
        (self.input_addr, self.input.reshape(-1)),
        (self.weight_addr, self.weight.reshape(-1)),
        (self.output_addr, self.output.reshape(-1))])

  def make_kernel(self):
    ndp_unit_log2 = int(math.log2(configs.ndp_units))
    stride_log2 = int(math.log2(configs.stride))
    packet_size_log2 = int(math.log2(configs.packet_size))
    elems_per_packet = configs.packet_size // configs.data_size
    elems_per_packet_log2 = int(math.log2(elems_per_packet))
    split_log2 = int(math.log2(self.split))
    repeat_log2 = int(math.log2(self.repeat))
    unroll_level_log2 = int(math.log2(self.unroll_level))
    multiple_load = 1
    multiple_load_log = 0
    input_unroll_log2 = int(math.log2(self.input_unroll))
    while self.input_unroll > 32:
      multiple_load_log +=1
      self.input_unroll = self.input_unroll // 2
      multiple_load = 2 ** multiple_load_log
      input_unroll_log2 -=1
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x8, (x1)\n' # input_addr
    template += f'ld x3, 24(x1)\n' # vector_length
    template += f'slli x3, x3, 2\n' # vector_length * 4byte = vector_size
    template += f'andi x4, x2, {configs.stride -1}\n' # OFFSET & (stride - 1)
    template += f'srli x5, x2, {stride_log2 + ndp_unit_log2}\n' # OFFSET / stride / ndp_units
    template += f'slli x5, x5, {stride_log2 }\n' # OFFSET / stride / ndp_units * stride
    template += f'or x4, x5, x4\n' # OFFSET / stride / ndp_units * stride + OFFSET & (stride - 1)
    template += f'slli x4, x4, {input_unroll_log2 + multiple_load_log}\n'
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # next to argument region
    if self.repeat != 0:
      template += f'srli x3, x3, {repeat_log2}\n' # vector_length / repeat
    template += f'bge x4, x3, .SKIP0\n'
    template += f'add x8, x8, x4\n'
    template += f'add x1, x1, x4\n'
    for m in range(multiple_load):
      for i in range(self.input_unroll):
        template += f'vle32.v v{i}, (x8)\n'
        template += f'addi x8, x8, {configs.packet_size}\n'
      for i in range(self.input_unroll):
        template += f'vse32.v v{i}, (x1)\n'
        template += f'addi x1, x1, {configs.packet_size}\n'
    template += f'.SKIP0\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x8, 8(x1)\n' # weight_addr
    template += f'ld x3, 24(x1)\n' # vector_length
    template += f'ld x7, 16(x1)\n' # output_addr
    template += f'vmv.v.i v1, 0\n' # output initialize with 0
    template += f'andi x4, x2, {configs.packet_size * self.split - 1}\n'
    template += f'srli x5, x2, {packet_size_log2 + split_log2 }\n' # OFFSET / 8
    template += f'slli x5, x5, {packet_size_log2}\n'
    template += f'vfcvt.f.x.v v1, v1\n'
    for unroll in range(1, elems_per_packet):
      template += f'vmv.v.v v{unroll + 1}, v1\n'
    # template += f'mul x6, x3, x2\n' # vector_length * OFFSET / 8
    template += f'mul x6, x3, x5\n' # vector_length * OFFSET / 4
    template += f'slli x4, x4, {unroll_level_log2}\n'
    template += f'add x6, x6, x4\n'
    template += f'add x12, x8, x6\n' # weight_addr + OFFSET
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # next to argument region
    template += f'add x1, x1, x4\n'
    template += f'slli x3, x3, 2\n' # vector_length * 4byte
    # template += f'add x13, x12, x3\n'
    if self.repeat != 0:
      template += f'srli x4, x3, {repeat_log2}\n' # vector_length / repeat
    template += f'add x4, x1, x4\n'
    template += f'.LOOP1\n'
    # template += f'TEST.v.x x1, x1, x1\n'
    template += f'addi x13, x12, 0\n'
    for unroll in range(self.unroll_level):
      template += f'vle32.v v1{unroll+1}, {unroll * configs.packet_size}(x1)\n' #load input
    for i in range(elems_per_packet):
      if i > 0:
        template += f'add x13, x13, x3\n'
      for unroll in range(self.unroll_level):
        template += f'vle32.v v2{unroll+1}, {unroll * configs.packet_size }(x13)\n' #load weight
      for unroll in range(self.unroll_level):
        template += f'vfmacc.vv v{i+1}, v1{unroll+1}, v2{unroll+1}\n'
    template += f'addi x1, x1, {configs.packet_size * self.unroll_level * self.split}\n' # next input addr
    template += f'addi x12, x12, {configs.packet_size * self.unroll_level * self.split}\n' # next input addr
    template += f'blt x1, x4, .LOOP1\n'
    template += f'vmv.v.i v0, 0\n'
    for unroll in range(elems_per_packet):
      template += f'vfredosum.vs v{elems_per_packet - unroll}, v{elems_per_packet - unroll}, v0\n'
    for unroll in range(elems_per_packet - 1):
      template += f'vslide1up.vx v{elems_per_packet - unroll}, v{elems_per_packet - unroll}, x0\n'
      template += f'vadd.vv v{elems_per_packet- unroll - 1}, v{elems_per_packet - unroll -1}, v{elems_per_packet - unroll}\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n' # output initialize with 0
    template += f'vsll.vi v2, v2, 2\n' # 4byte
    template += f'add x7, x7, x5\n'
    template += f'vamoaddei32.v x0, (x7), v2, v1\n'

    return template

class OptGemvInputMap(NdpKernel):
  def __init__(self, input_size, weight_size, fuse_bias=False):
    super().__init__()
    self.batch_size = 1
    self.unroll_level = 8
    self.vector_size = input_size
    self.output_size = weight_size
    self.input_addr = 0x800000000000
    self.weight_addr = 0x810000000000
    self.bias_addr = 0x820000000000
    self.output_addr = 0x830000000000
    self.sync = 0
    self.kernel_id = 0
    self.base_addr = self.input_addr
    self.bound = self.vector_size * configs.data_size
    self.input_addrs = [self.output_addr, self.weight_addr, self.vector_size, self.output_size, self.batch_size]
    self.input = np.random.randn(self.batch_size, 1, self.vector_size).astype(np.float32)
    self.input = np.ones((self.batch_size, 1, self.vector_size)).astype(np.float32)
    self.weight = np.random.randn(self.batch_size, self.vector_size, self.output_size).astype(np.float32)
    self.weight = np.ones((self.batch_size, self.vector_size, self.output_size)).astype(np.float32)
    self.bias = np.random.randn(self.batch_size, self.output_size).astype(np.float32)
    self.initialized_output = np.zeros((self.batch_size, 1, self.output_size)).astype(np.float32)
    self.output = (self.input @ self.weight) #+ self.bias
    self.smem_size = weight_size * configs.data_size

  def make_kernel(self):
    ndp_unit_log2 = int(math.log2(configs.ndp_units))
    stride_log2 = int(math.log2(configs.stride))
    packet_size_log2 = int(math.log2(configs.packet_size))
    elems_per_packet = configs.packet_size // configs.data_size
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x6, {configs.spad_addr}\n'
    template += f'ld x7, (x6)\n' # output_addr
    template += f'ld x3, 24(x6)\n' # output_length
    template += f'slli x3, x3, 2\n' # output_length * 4byte = output_size
    template += f'andi x4, x2, {configs.stride -1}\n' # OFFSET & (stride - 1)
    template += f'srli x5, x2, {stride_log2 + ndp_unit_log2}\n' # OFFSET / stride / ndp_units
    template += f'slli x5, x5, {stride_log2 }\n' # OFFSET / stride / ndp_units * stride
    template += f'or x4, x5, x4\n' # OFFSET / stride / ndp_units * stride + OFFSET & (stride - 1)
    template += f'slli x4, x4, {ndp_unit_log2 + 3}\n'
    template += f'addi x6, x6, {configs.packet_size * 2}\n' # next to argument region
    template += f'bge x4, x3, .SKIP0\n'
    template += f'vmv.v.i v1, 0\n' # output initialize with 0
    template += f'vfcvt.f.x.v v1, v1\n'
    template += f'add x6, x6, x4\n'
    for i in range(configs.ndp_units  * 8):
      template += f'vse32.v v1, (x6)\n'
      template += f'addi x6, x6, {configs.packet_size}\n'
    template += f'.SKIP0\n'

    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x6, {configs.spad_addr}\n'
    template += f'ld x7, 8(x6)\n' # weight_addr
    template += f'ld x4, 24(x6)\n' # output_length
    template += f'vmset.m v0\n'
    template += f'vid.v v1, v0\n' # output initialize with 0
    template += f'vsll.vi v1, v1, 2\n' # 4byte
    template += f'li x5, 0\n' # output index
    template += f'mul x3, x2, x4\n' # OFFSET * output_length
    template += f'add x7, x7, x3\n' # weight_addr + OFFSET * output_length
    template += f'slli x4, x4, 2\n' # output_length * 4byte = output_size
    template += f'addi x6, x6, {configs.packet_size * 2}\n' # next to argument region
    template += f'vle32.v v2, (x1)\n' #load input
    template += f'vmv.v.i v3, 0\n' # initialize with 0
    template += f'vfcvt.f.x.v v3, v3\n'
    for i in range(elems_per_packet):
      template += f'csrwi vstart, {i}\n'
      template += f'vfmv.f.s f{i}, v2\n' # input[i]
    template += f'.LOOP1\n'
    template += f'add x3, x7, x5\n'
    for unroll in range(self.unroll_level):
        template += f'vmv.v.v v{unroll + 4}, v3\n'
    for i in range(elems_per_packet):
      for unroll in range(self.unroll_level):
        template += f'vle32.v v{unroll + self.unroll_level + 4}, {unroll * configs.packet_size}(x3)\n' # load weight
      for unroll in range(self.unroll_level):
        template += f'vfmacc.vf v{unroll + 4}, v{unroll + self.unroll_level + 4}, f{i}\n'
      template += f'add x3, x3, x4\n'
    for unroll in range(self.unroll_level):
      template += f'vamoaddei32.v x0, (x6), v1, v{unroll + 4}\n'
      template += f'addi x6, x6, {configs.packet_size}\n' # next input addr
    # template += f'TEST.v.x x5, x5, x5\n'
    # template += f'TEST.v.x x4, x4, x4\n'

    template += f'addi x5, x5, {configs.packet_size * self.unroll_level}\n' # next weight addr'
    template += f'blt x5, x4, .LOOP1\n'

    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x6, {configs.spad_addr}\n'
    template += f'ld x7, (x6)\n' # output_addr
    template += f'ld x3, 24(x6)\n' # output_length
    template += f'slli x3, x3, 2\n' # output_length * 4byte = output_size
    template += f'andi x4, x2, {configs.stride -1}\n' # OFFSET & (stride - 1)
    template += f'srli x5, x2, {stride_log2 + ndp_unit_log2}\n' # OFFSET / stride / ndp_units
    template += f'slli x5, x5, {stride_log2 }\n' # OFFSET / stride / ndp_units * stride
    template += f'or x4, x5, x4\n' # OFFSET / stride / ndp_units * stride + OFFSET & (stride - 1)
    template += f'slli x4, x4, {ndp_unit_log2 + 3}\n'
    template += f'addi x6, x6, {configs.packet_size * 2}\n' # next to argument region
    template += f'bge x4, x3, .SKIP1\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v1, v0\n' # output initialize with 0
    template += f'vsll.vi v1, v1, 2\n' # 4byte
    template += f'add x7, x7, x4\n'
    template += f'add x6, x6, x4\n'
    for j in range(8):
      for i in range(configs.ndp_units):
        template += f'vle32.v v{i+2}, (x6)\n'
        template += f'addi x6, x6, {configs.packet_size}\n'
      for i in range(configs.ndp_units):
        # template += f'TEST.v.x x7, x7, x7\n'
        template += f'vamoaddei32.v x0, (x7), v1, v{i+2}\n'
        template += f'addi x7, x7, {configs.packet_size}\n'
    template += f'.SKIP1\n'

    return template

  def make_input_map(self):
    return make_memory_map([
          (self.input_addr, self.input.reshape(-1)),
          (self.weight_addr, self.weight.reshape(-1)),
          (self.output_addr, self.initialized_output.reshape(-1))
      ])

  def make_output_map(self):
    return make_memory_map([
        (self.input_addr, self.input.reshape(-1)),
        (self.weight_addr, self.weight.reshape(-1)),
        (self.output_addr, self.output.reshape(-1))])

class OptFc1(OptGemvOutputMapSplit):
  def __init__(self, config : str = 'opt-125m', model_parallelism=1):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path))
    input_size = config_file['hidden_size']
    output_size = config_file['ffn_dim'] // model_parallelism
    super().__init__(input_size, output_size)
    self.kernel_name = f'{config}-fc1'

class OptFc2(OptGemvOutputMapSplit):
  def __init__(self, config : str = 'opt-125m', model_parallelism=1):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path))
    input_size = config_file['ffn_dim'] // model_parallelism
    output_size = config_file['hidden_size']
    repeat = 1
    split = max(input_size // output_size, 1)
    if config == 'opt-30b':
      split *= 2
    input_unroll = 32
    if config == 'opt-125m':
      input_unroll = -1
    super().__init__(input_size, output_size, repeat=repeat, split=split, input_unroll=input_unroll)
    self.kernel_name = f'{config}-fc2'

class OptQKVProj(OptGemvOutputMapSplit):
  def __init__(self, config : str = 'opt-125m', model_parallelism=1):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path))
    input_size = config_file['hidden_size']
    output_size = config_file['hidden_size'] * 3 // model_parallelism
    split = max(input_size // output_size, 1)
    if config == 'opt-2.7b':
      split = 1
    super().__init__(input_size, output_size, split=split, repeat=1)
    self.kernel_name = f'{config}-qkv-proj'

class OptOutProj(OptGemvOutputMapSplit):
  def __init__(self, config : str = 'opt-125m', model_parallelism=1):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path))
    input_size = config_file['hidden_size'] // model_parallelism
    output_size = config_file['hidden_size']
    split = max(input_size // output_size, 1)
    # if config == 'opt-2.7b':
    split = 4
    super().__init__(input_size, output_size, repeat=1, split=split, unroll_level=8)
    self.kernel_name = f'{config}-out-proj'