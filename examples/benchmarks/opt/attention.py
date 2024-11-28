from typing import Any
import numpy as np
import os
from utils.utils import MultipleKernel, NdpKernel, make_memory_map, pad8, pad16
import configs
from .fc import OptGemv
import json
import math

class OptAttentionGemv(MultipleKernel):
  def __init__(self, repeat, input_dim, output_dim):
    super().__init__(repeat=repeat)
    self.repeat = repeat
    self.batch_size = 1
    self.unroll_level = 4
    self.vector_size = input_dim
    self.output_size = output_dim
    self.input_addr = 0x800000000000
    self.weight_addr = 0x810000000000
    self.bias_addr = 0x820000000000
    self.output_addr = 0x830000000000
    self.sync = 0
    self.kernel_id = 0
    self.base_addr = self.weight_addr
    self.bound =  self.vector_size * self.output_size * configs.data_size * self.batch_size // 8
    self.base_offset = self.vector_size * self.output_size * configs.data_size * self.batch_size
    self.input_addrs = [self.input_addr, self.output_addr, self.weight_addr, self.vector_size, self.output_size, self.batch_size]
    self.input_addrs_offsets = [input_dim * configs.data_size,
                                self.output_size * configs.data_size,
                                self.vector_size * self.output_size * configs.data_size, 0, 0, 0]
    self.input = np.random.randn(self.batch_size, self.repeat, self.vector_size).astype(np.float32)
    self.weight = np.random.randn(self.repeat, self.vector_size, self.output_size).astype(np.float32)
    self.bias = np.random.randn(self.batch_size, self.output_size).astype(np.float32)
    self.initialized_output = np.zeros((self.batch_size, self.repeat, self.output_size)).astype(np.float32)
    self.output = np.zeros((self.batch_size, self.repeat, self.output_size)).astype(np.float32)
    #iterate over repeat
    for i in range(self.repeat):
      self.output[:,i,:] = (self.input[:,i,:] @ self.weight[i,:,:])
    # print(self.output.shape)
    self.smem_size = input_dim * configs.data_size + output_dim * configs.data_size

  def make_kernel(self):
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x13, (x1)\n' # input_addr
    template += f'ld x3, 24(x1)\n' # vector_length
    template += f'ld x9, 32(x1)\n' # output_length
    # template += f'ld x4, 40(x1)\n' # batch_size ## ucnomment if consider batch
    template += f'slli x7, x3, 2\n' # vector_length * 4byte = vector_size
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # next to argument region
    template += f'add x7, x1, x7\n' # spad_addr + vector_size
    # template += f'li x5, 0\n' # batch index
    # template += f'.LOOP0\n'
    template += f'li x6, 0\n' # vector index
    template += f'.LOOP1\n'
    for i in range(1, self.unroll_level + 1):
      template += f'vle32.v v{i}, (x13)\n'
      template += f'vse32.v v{i}, (x1)\n'
      template += f'addi x13, x13, {configs.packet_size}\n'
      template += f'addi x1, x1, {configs.packet_size}\n'
    template += f'addi x6, x6, {8 * self.unroll_level}\n' # 256 / float 32 = 8 elements in a row
    template += f'blt x6, x3, .LOOP1\n'
    template += f'li x8, 0\n' # output index
    # template += f'vsetvli t0, a0, e32, m2\n'
    template += f'vmv.v.i v{self.unroll_level + 1}, 0\n' # initialize with 0
    template += f'vfcvt.f.x.v v{self.unroll_level + 1}, v{self.unroll_level + 1}\n'
    template += f'.LOOP2\n'
    for i in range(self.unroll_level):
      template += f'vse32.v v{self.unroll_level + 1}, (x7)\n'
      template += f'addi x7, x7, {configs.packet_size}\n'
    template += f'addi x8, x8, {8 * self.unroll_level}\n' # 256 / float 32 = 8 elements in a row
    template += f'blt x8, x9, .LOOP2\n'
    # template += f'addi x5, x5, 1\n'
    # template += f'bne x5, x4, .LOOP0\n'

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

    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x3, 24(x1)\n' # vector_length
    template += f'ld x13, 8(x1)\n' # output_addr
    template += f'ld x4, 32(x1)\n' # output_length
    template += f'addi x1, x1, {configs.packet_size * 2}\n' # next to argument region
    template += f'slli x5, x3, 2\n' # vector_length * 4byte
    template += f'add x5, x1, x5\n' # spad_addr + vector_size = output_base_addr
    template += f'li x6, 0\n' # output index
    template += f'vmset.m v0\n'
    template += f'vid.v v31, v0\n'
    template += f'vsll.vi v31, v31, 2\n' # 4byte
    template += f'.LOOP4\n'
    for i in range(self.unroll_level):
      template += f'vle32.v v{i + 1}, (x5)\n'
      template += f'addi x5, x5, {configs.packet_size}\n'
    for i in range(self.unroll_level):
      template += f'vamoaddei32.v x0, (x13), v31, v{i + 1}\n'
      template += f'addi x13, x13, {configs.packet_size}\n'
    template += f'addi x6, x6, {8 * self.unroll_level}\n' # 256 / float 32 = 8 elements in a row
    template += f'blt x6, x4, .LOOP4\n'
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



class OptQK(OptAttentionGemv):
  def __init__(self, config : str = 'opt-125m', context_len : int = 1024):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path, 'r'))
    num_heads = config_file['num_attention_heads']
    hidden_dim = config_file['hidden_size']
    input_dim = hidden_dim // num_heads
    super().__init__(repeat=num_heads, input_dim=input_dim, output_dim=context_len)
    self.kernel_name = f'{config}-qk'

class OptSV(OptAttentionGemv):
  def __init__(self, config : str = 'opt-125m', context_len : int = 1024):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path, 'r'))
    num_heads = config_file['num_attention_heads']
    hidden_dim = config_file['hidden_size']
    output_dim = hidden_dim // num_heads
    super().__init__(repeat=num_heads, input_dim=context_len, output_dim=output_dim)
    self.kernel_name = f'{config}-sv'


class OptSoftmax0(MultipleKernel):
  def __init__(self, config : str = 'opt-125m', context_len : int = 1024):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path, 'r'))
    num_heads = config_file['num_attention_heads']
    super().__init__(repeat=num_heads)
    self.vector_size = context_len
    self.input_data_addr = 0x810000000000
    self.output_max_addr = 0x890000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'{config}-softmax_kernel0'
    self.base_addr = self.input_data_addr
    self.bound = self.vector_size * configs.data_size
    self.input_addrs = [self.output_max_addr]
    self.input_addrs_offsets = [4]
    self.smem_size = configs.packet_size

    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")

    input_data = np.random.rand(num_heads, self.vector_size).astype(data_type)#np.random.rand(self.vector_size).astype(np.float32)
    outputs = np.max(input_data, axis=-1, keepdims=True).flatten()
    # print(outputs.shape)
    self.input_data = input_data
    self.outputs = pad8(np.array(outputs, dtype=data_type))

  def make_kernel(self):
    data_size = configs.data_size
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    # Initializer
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'li x5, {configs.spad_addr + configs.packet_size}\n'
    template += f'vle{data_size*8}.v v1, (x1)\n'
    template += f'vse{data_size*8}.v v1, (x5)\n'                # Initialize with first packet

    # Body
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    # Shared atomic operation
    template += f'li x5, {configs.spad_addr + configs.packet_size}\n'
    template += f'vle{data_size*8}.v v1, (x1)\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vmul.vi v2, v2, {data_size}\n'
    template += f'vamomaxei{data_size*8}.v x0, (x5), v2, v1\n'

    # Finalizer
    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'li x5, {configs.spad_addr}\n'
    template += f'ld x6, (x5)\n'                                # x6 = output_max_addr
    template += f'addi x5, x5, {configs.packet_size}\n'
    template += f'vle{data_size*8}.v v1, (x5)\n'                           # v1 = spad_addr + packet_size[]
    template += f'vfredomax.vs v1, v1, v1\n'                    # v1 = reduce(v1)

    # Global atomic operation
    """
    For debug code below
    template += f'muli x3, NDPID, {configs.packet_size}\n'
    template += f'addi x4, x6, {configs.packet_size}\n'
    template += f'add x4, x4, x3\n'
    template += f'vse{data_size*8}.v v1, (x4)\n'
    """

    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vmul.vi v2, v2, {data_size}\n'
    # FIXME. it should be sacalar amomaxei
    template += f'vamomaxei{data_size*8}.v x0, (x6), v2, v1\n'
    return template

  def make_input_map(self):
   return make_memory_map([
        (self.input_data_addr, self.input_data),
    ])

  def make_output_map(self):
    return make_memory_map([
        (self.input_data_addr, self.input_data),
        (self.output_max_addr, self.outputs)])

class OptSoftmax1(MultipleKernel):
  def __init__(self, config : str = 'opt-125m', context_len : int = 1024):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path, 'r'))
    num_heads = config_file['num_attention_heads']
    super().__init__(repeat=num_heads)
    self.vector_size = context_len
    self.input_data_addr = 0x810000000000
    self.input_max_addr = 0x820000000000
    self.output_addr = 0x890000000000
    self.reduce_output_addr = 0x8a0000000000
    self.sync = 0
    self.kernel_id = 1
    self.kernel_name = f'{config}-softmax_kernel1'
    self.base_addr = self.input_data_addr
    self.bound = self.vector_size * configs.data_size
    self.smem_size = configs.packet_size * 4

    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")

    input_data = np.random.rand(num_heads, self.vector_size).astype(data_type)#np.random.rand(self.vector_size).astype(np.float32)
    input_max_data = np.max(input_data, axis=-1, keepdims=True)
    outputs = np.exp(input_data-input_max_data)
    reduce_outputs = np.sum(outputs, axis=-1, keepdims=True).astype(np.float32)

    self.input_data = input_data
    self.input_max_data = input_max_data
    self.outputs = outputs
    self.reduce_outputs = reduce_outputs

    self.input_addrs = [self.output_addr, self.reduce_output_addr, self.input_max_addr]
    self.input_addrs_offsets = [self.bound, 4, 4]
    self.smem_size = configs.packet_size * 2

  def make_kernel(self):
    data_size = configs.data_size
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    # Initializer
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e{32}, m1\n'
    template += f'li x5, {configs.spad_addr + configs.packet_size}\n'
    template += f'vmv.v.i v1, 0\n'
    template += f'vfcvt.f.x.v v1, v1\n'
    template += f'vse{32}.v v1, (x5)\n'                # Scratch pad Initialize with 0

    # Body
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    # Shared atomic operation
    template += f'li x5, {configs.spad_addr + configs.packet_size}\n' # Scratch pad addr
    template += f'li x6, {configs.spad_addr}\n'                       #
    template += f'ld x3, (x6)\n'
    template += f'add x3, x3, x2\n'
    template += f'ld x4, 16(x6)\n'                                    # x3 = input_max_addr
    template += f'flw f3, (x4)\n'

    template += f'vle{data_size*8}.v v1, (x1)\n'
    template += f'vfsub.vf v1, v1, f3\n'
    template += f'vfexp.v v1, v1\n'
    template += f'vse{data_size*8}.v v1, (x3)\n'

    template += f'vmv.v.i v2, 0\n'                              # v1 = 0
    template += f'vfcvt.f.x.v v2, v2\n'
    if configs.data_size == 2:
      template += f'vwredosum.vs v3, v1, v2\n'
      template += f'vsetvli t0, a0, e32, m1\n'
    elif configs.data_size == 4:
      template += f'vfredosum.vs v3, v1, v2\n'
    else:
       raise Exception("Unsupported data type")
    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vmul.vi v2, v2, 4\n'
    template += f'vamoaddei32.v x0, (x5), v2, v3\n'

    # Finalizer
    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x5, {configs.spad_addr}\n'
    template += f'ld x6, 8(x5)\n'                                # x6 = reduce_output
    template += f'addi x5, x5, {configs.packet_size}\n'
    template += f'vle32.v v1, (x5)\n'

    # Global atomic operation
    """
    For debug code below
    template += f'muli x3, NDPID, {configs.packet_size}\n'
    template += f'addi x4, x6, {configs.packet_size}\n'
    template += f'add x4, x4, x3\n'
    template += f'vse{data_size*8}.v v1, (x4)\n'
    """

    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vmul.vi v2, v2, 4\n'
    template += f'vamoaddei32.v x0, (x6), v2, v1\n'
    return template

  def make_input_map(self):
   return make_memory_map([
        (self.input_data_addr, self.input_data), (self.input_max_addr, self.input_max_data)
    ])

  def make_output_map(self):
    return make_memory_map([
        (self.input_data_addr, self.input_data), (self.input_max_addr, self.input_max_data),
        (self.output_addr, self.outputs), (self.reduce_output_addr, self.reduce_outputs)])

class OptSoftmax2(MultipleKernel):
  def __init__(self, config : str = 'opt-125m', context_len : int = 1024):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path, 'r'))
    num_heads = config_file['num_attention_heads']
    super().__init__(repeat=num_heads)
    self.vector_size = context_len
    self.input_addr =  0x800000000000
    self.output_addr = 0x820000000000
    self.coefficient_addr = 0x830000000000
    self.sync = 0
    self.kernel_id = 2
    self.kernel_name = f'{config}-softmax_kernel2'
    self.base_addr = self.input_addr
    self.bound = self.vector_size * configs.data_size
    self.smem_size = configs.packet_size
    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")
    input_data = np.random.rand(num_heads, self.vector_size).astype(data_type)
    input_max_data = np.max(input_data, axis=-1, keepdims=True)
    outputs = np.exp(input_data-input_max_data)
    self.input_data = outputs
    self.coefficient = np.sum(outputs, axis=-1, keepdims=True).astype(np.float32)
    self.output = (self.input_data / self.coefficient).astype(data_type)
    self.input_addrs = [self.output_addr, self.coefficient_addr]
    self.input_addrs_offsets = [self.bound, 4]

  def make_kernel(self):
    packet_size = configs.packet_size
    data_size = configs.data_size
    spad_addr = configs.spad_addr
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'li x4, {configs.spad_addr}\n'
    template += f'ld x5, 8(x4)\n'
    template += f'li x3, {1}\n'
    template += f'flw f1, (x5)\n'
    template += f'fmv.w.x f2, x3\n'
    template += f'fdiv f1, f2, f1\n'            # 1 / sum(E(tensor))
    template += f'fsw f1, {packet_size}(x4)\n'

    template += f'KERNELBODY:\n'
    template += f'vsetvli 0, 0, e32, m1, 0\n'
    template += f'li x4, {spad_addr}\n'
    template += f'flw f1, {packet_size}(x4)\n'
    template += f'ld x4, (x4)\n'
    template += f'add x4, x4, x2\n'
    template += f'vle32.v v1, (x1)\n'
    template += f'vfmul.vf v3, v1, f1\n'
    template += f'vse32.v v3, (x4)\n'
    return template

  def make_input_map(self):
   return make_memory_map([
        (self.input_addr, self.input_data), (self.coefficient_addr, self.coefficient)
    ])

  def make_output_map(self):
    return make_memory_map([
        (self.input_addr, self.input_data), (self.coefficient_addr, self.coefficient),
        (self.output_addr, self.output)])

class OptAttention1Kernel(MultipleKernel):
  def __init__(self, config : str = 'opt-125m', context_len : int = 1024, max_context_len : int = 2048, model_parallelism=1):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path, 'r'))
    num_heads = config_file['num_attention_heads'] // model_parallelism
    # num_heads = 2
    # num_heads = 56
    super().__init__(repeat=num_heads)
    self.kernel_name = f'{config}-attention1'
    self.head_dim = config_file['hidden_size'] // num_heads // model_parallelism
    # self.head_dim = 64
    # self.head_dim = config_file['hidden_size'] // 12
    self.context_len = context_len
    self.max_context = max_context_len
    self.query_addr = 0x800000000000
    self.key_addr = 0x810000000000
    self.value_addr = 0x820000000000
    self.coefficient_addr = 0x830000000000
    self.output_addr = 0x840000000000
    self.reduced_output_addr = 0x850000000000
    if self.head_dim % 10 == 0:
      self.unroll_level = 10
    elif self.head_dim % 8 == 0:
      self.unroll_level = 8
    elif self.head_dim % 4 == 0:
      self.unroll_level = 4
    elif self.head_dim % 2 == 0:
      self.unroll_level = 2
    else:
      self.unroll_level = 1
    self.bound = context_len * configs.data_size
    self.scale = 1.0 / math.sqrt(self.head_dim)
    self.input_addrs = [self.query_addr, # 0
                        self.key_addr, # 8
                        self.value_addr, # 16
                        self.output_addr, # 24
                        self.coefficient_addr, # 32
                        self.head_dim, # 40
                        self.context_len, #48
                        np.float32(self.scale)] # 56
    self.input_addrs_offsets = [self.head_dim * configs.data_size, #query_addr
                                self.head_dim * self.max_context * configs.data_size, #key_addr
                                self.head_dim * self.max_context * configs.data_size, #value_addr
                                self.head_dim * configs.data_size * configs.ndp_units, #output_addr
                                configs.ndp_units * configs.data_size * 2, #coefficient_addr
                                0, 0, 0]
    self.query = np.random.randn(self.repeat, self.head_dim).astype(np.float32)
    # self.query = np.ones((self.repeat, self.head_dim)).astype(np.float32)
    # self.key = np.random.randn(self.repeat, self.head_dim, self.context_len).astype(np.float32)
    self.key = np.random.randn(self.repeat, self.context_len, self.head_dim).astype(np.float32)
    self.key_t = np.transpose(self.key, (0, 2, 1))
    # self.key = np.ones((self.repeat, self.head_dim, self.context_len)).astype(np.float32)

    self.value = np.random.randn(self.repeat,  self.context_len, self.head_dim).astype(np.float32)
    # self.value = np.ones((self.repeat, self.context_len, self.head_dim)).astype(np.float32)

    interleave = configs.stride // configs.data_size
    interleave_count = math.ceil(self.context_len / interleave)
    self.qk = np.zeros((self.repeat, self.context_len)).astype(np.float32)
    self.softmax = np.zeros((self.repeat, self.context_len)).astype(np.float32)
    for head in range(self.repeat):
      self.qk[head,:] = (self.query[head,:] @ self.key_t[head,:,:]).astype(np.float32)

    self.head_max = np.max(self.qk, axis=-1, keepdims=True).astype(np.float32)
    self.local_max = np.zeros((self.repeat, configs.ndp_units)).astype(np.float32)
    for head in range(self.repeat):
      for i in range(interleave_count):
        # ndp_id = i % configs.ndp_units
        ndp_id = (i + head * interleave_count) % configs.ndp_units
        self.local_max[head, ndp_id] = max(self.local_max[head, ndp_id], np.max(self.qk[head, i * interleave : (i+1) * interleave]))
    self.interleaved_max = np.zeros((self.repeat, context_len)).astype(np.float32)
    for head in range(self.repeat):
      for i in range(interleave_count):
        ndp_id = (i + head * interleave_count) % configs.ndp_units
        self.interleaved_max[head, i * interleave : (i+1) * interleave] = self.local_max[head,ndp_id]

    self.exp = np.exp(self.qk - self.head_max)
    self.interleaved_exp = np.exp(self.qk - self.interleaved_max)


    self.local_sum = np.zeros((self.repeat, configs.ndp_units)).astype(np.float32)
    for head in range(self.repeat):
      for i in range(interleave_count):
        ndp_id = (i + head * interleave_count) % configs.ndp_units
        self.local_sum[head, ndp_id] += np.sum(self.interleaved_exp[head, i * interleave : (i+1) * interleave])

    self.softmax = self.exp / np.sum(self.exp, axis=-1, keepdims=True)

    self.attention = np.zeros((self.repeat, self.head_dim)).astype(np.float32)
    self.local_attention = np.zeros((self.repeat, configs.ndp_units, self.head_dim)).astype(np.float32)

    for head in range(self.repeat):
      self.attention[head,:] = self.softmax[head,:] @ self.value[head,:,:]

    for head in range(self.repeat):
      for i in range(interleave_count):
        ndp_id = (i + head * interleave_count) % configs.ndp_units
        self.local_attention[head, ndp_id,:] += self.interleaved_exp[head, i * interleave : (i+1) * interleave] @ self.value[head, i * interleave : (i+1) * interleave,:]

    #Scratch pad memory allcation
    self.smem_size = self.head_dim * configs.data_size \
                    + self.head_dim * configs.data_size \
                    + max(self.context_len * configs.data_size // configs.ndp_units, configs.stride)  \
                    + configs.packet_size * 3

    self.base_addr = self.output_addr
    self.sync = 0

  def make_kernel(self):
    ndp_unit_log2 = int(math.log2(configs.ndp_units))
    stride_log2 = int(math.log2(configs.stride))
    packet_size_log2 = int(math.log2(configs.packet_size))

    spad_argument_addr = configs.spad_addr
    spad_query_addr = spad_argument_addr + configs.packet_size * 3
    spad_output_addr = spad_query_addr + configs.data_size * self.head_dim
    spad_max_addr = spad_output_addr + configs.data_size * self.head_dim
    spad_softmax_addr = spad_max_addr + configs.packet_size

    accum_offset = 1
    query_offset = configs.packet_size // configs.data_size + 1
    key_offset = configs.packet_size // configs.data_size + self.unroll_level + 1

    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'INITIALIZER:\n'
    #Initializer - load query to spad
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x10, {spad_argument_addr}\n'
    template += f'ld x9, (x10)\n' # query_addr
    template += f'ld x3, 40(x10)\n' # head_dim
    template += f'li x7, {spad_output_addr}\n'
    template += f'li x10, {spad_query_addr}\n' # query_spadd_addr
    template += f'li x6, 0\n' # vector index
    template += 'vmv.v.i v0, 0\n'
    template += 'vfcvt.f.x.v v0, v0\n'
    template += f'.LOOP1\n'
    for i in range(1, self.unroll_level + 1):
      template += f'vle32.v v{i}, (x9)\n'
      template += f'addi x9, x9, {configs.packet_size}\n'
    for i in range(1, self.unroll_level + 1):
      template += f'vse32.v v{i}, (x10)\n'
      template += f'vse32.v v0, (x7)\n'
      template += f'addi x10, x10, {configs.packet_size}\n'
      template += f'addi x7, x7, {configs.packet_size}\n'
    template += f'addi x6, x6, {8 * self.unroll_level}\n' # 256 / float 32 = 8 elements in a row
    template += f'blt x6, x3, .LOOP1\n'
    template += f'li x10, {spad_max_addr}\n'
    template += f'vse32.v v0, (x10)\n' # initialize with 0
    template += f'KERNELBODY:\n'
    #First kernel body, query * key & max reduction
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x10, {spad_argument_addr}\n'
    template += f'ld x9, 8(x10)\n' # key_addr
    template += f'li x3, 0\n' # query_offset
    template += f'ld x6, 40(x10)\n' # head_dim
    template += f'mul x7, x2, x6\n' # OFFSET * head_dim
    template += f'slli x6, x6, 2\n' # head_dim * 4byte = query_size
    template += f'vmv.v.i v0, 0\n' # initialize with 0
    template += f'vfcvt.f.x.v v0, v0\n'
    template += f'li x8, {spad_query_addr}\n' # query_spadd_addr
    template += f'add x9, x9, x7\n' # key_addr + OFFSET
    for i in range(configs.packet_size//configs.data_size):
      template += f'vmv.v.v v{i + accum_offset}, v0\n'
    template += f'.LOOP2\n'
    template += f'add x4, x8, x3\n' # query_spadd_addr + query_offset
    template += f'add x5, x9, x3\n' # key_addr + OFFSET + query_offset
    for unroll in range(self.unroll_level):
      template += f'vle32.v v{unroll+query_offset}, {unroll * configs.packet_size}(x4)\n' # load query
    for i in range(configs.packet_size // configs.data_size):
      for unroll in range(self.unroll_level):
        template += f'vle32.v v{unroll + key_offset}, {unroll * configs.packet_size}(x5)\n' # load key
      for unroll in range(self.unroll_level):
        template += f'vfmacc.vv v{i + accum_offset}, v{unroll + query_offset}, v{unroll + key_offset}\n'
      template += f'add x5, x5, x6\n'
    template += f'addi x3, x3, {configs.packet_size * self.unroll_level}\n'
    template += f'blt x3, x6, .LOOP2\n'
    for i in range(configs.packet_size // configs.data_size):
      template += f'vfredosum.vs v{i + accum_offset}, v{i + accum_offset}, v0\n' # Reduce sum
    for i in range(configs.packet_size // configs.data_size, accum_offset, -1):
      template += f'vslide1up.vx v{i}, v{i}, x0\n'
      template += f'vfadd.vv v{i-1}, v{i-1}, v{i}\n'
    template += f'andi x9, x2, {configs.stride -1}\n' # OFFSET & (stride - 1)
    template += f'srli x3, x2, {stride_log2 + ndp_unit_log2}\n' # OFFSET / stride / ndp_units
    template += f'slli x3, x3, {stride_log2 }\n' # OFFSET / stride / ndp_units * stride
    template += f'or x9, x3, x9\n' # OFFSET / stride / ndp_units * stride + OFFSET & (stride - 1) -> Score offset
    template += f'li x6, {spad_max_addr}\n' # spad_max_addr
    template += f'li x3, {spad_softmax_addr}\n' # spad_softmax_addr
    template += f'add x10, x9, x3\n' # spad_softmax_addr + score_offset
    template += f'vse32.v v1, (x10)\n' # query_offset + score_offset + packet_size
    template += f'vfredomax.vs v1, v1, v1\n' # Reduce max
    template += f'vmset.m v0\n'
    template += f'vid.v v4, v0\n'
    template += f'vsll.vi v4, v4, 2\n'
    template += f'vamomaxei32.v x0, (x6), v4, v1\n' # Atomic max
    template += f'KERNELBODY:\n'

    #Second kernel body, exp & reduce sum
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'andi x3, x2, {configs.stride -1}\n' # OFFSET & (stride - 1)
    template += f'srli x4, x2, {stride_log2 + ndp_unit_log2}\n' # OFFSET / stride / ndp_units
    template += f'slli x4, x4, {stride_log2 }\n' # OFFSET / stride / ndp_units * stride
    template += f'or x3, x4, x3\n' # OFFSET / stride / ndp_units * stride + OFFSET & (stride - 1)
    template += f'li x6, {spad_max_addr}\n' # next to argument region
    template += f'flw f1, (x6)\n' #load max
    template += f'li x10, {spad_softmax_addr}\n' # spad_softmax_addr
    template += f'add x10, x10, x3\n' # spad_addr + vector_size + score_offset
    template += f'vle32.v v1, (x10)\n'
    template += f'vfsub.vf v1, v1, f1\n' # query_offset + score_offset + packet_size - max
    template += f'vfexp.v v1, v1\n'
    template += f'vse32.v v1, (x10)\n' # query_offset + score_offset + packet_size
    template += f'vmv.v.i v0, 0\n'
    template += f'vfredosum.vs v1, v1, v0\n' # Reduce sum
    template += f'vmset.m v0\n'
    template += f'vid.v v4, v0\n'
    template += f'vsll.vi v4, v4, 2\n'
    template += f'vslide1up.vx v1, v1, x0\n'
    template += f'vamoaddei32.v x0, (x6), v4, v1\n' # Atomic sum
    template += f'KERNELBODY:\n'

    #Fourth kernel body - query * key * 1 / reduced sum * value
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x9, {spad_max_addr}\n'
    template += f'li x10, {spad_softmax_addr}\n' # spad_softmax_addr
    template += f'andi x3, x2, {configs.stride -1}\n' # OFFSET & (stride - 1)
    template += f'srli x4, x2, {stride_log2 + ndp_unit_log2}\n' # OFFSET / stride / ndp_units
    template += f'slli x4, x4, {stride_log2 }\n' # OFFSET / stride / ndp_units * stride
    template += f'or x3, x4, x3\n' # OFFSET / stride / ndp_units * stride + OFFSE
    template += f'add x3, x10, x3\n' # spad_addr + vector_size + score_offset
    template += f'vle32.v v1, (x3)\n' # exp(qk)
    template += f'li x6, {self.head_dim * configs.data_size}\n'
    template += f'li x10, {spad_argument_addr}\n'
    template += f'ld x5, 16(x10)\n' # value_addr
    template += f'srli x4, x2, 2\n'
    template += f'mul x4, x4, x6\n'
    template += f'add x5, x5, x4\n' # value_addr + OFFSET * head_dim * 4
    template += f'li x3, 0\n' # value_offset
    template += f'li x8, {spad_output_addr}\n' # next to argument region
    template += f'vmset.m v0\n'
    for i in range(configs.packet_size // configs.data_size):
      template += f'vfmv.f.s f{i}, v1\n'
      template += f'vslide1down.vx v1, v1, x0\n'
    template += f'vid.v v1, v0\n'
    template += f'vsll.vi v1, v1, 2\n'
    template += f'.LOOP3\n'
    template += f'add x9, x5, x3\n' # value_addr
    template += f'vmv.v.i v2, 0\n' # initialize with 0
    template += f'vfcvt.f.x.v v2, v2\n'
    for unroll in range(1, self.unroll_level):
      template += f'vmv.v.v v{unroll+2}, v2\n'

    for i in range(configs.packet_size // configs.data_size):
      for unroll in range(self.unroll_level):
        template += f'vle32.v v{self.unroll_level + unroll+ 2}, {unroll * configs.packet_size}(x9)\n' # load value
      template += f'add x9, x9, x6\n'
      for unroll in range(self.unroll_level):
        template += f'vfmacc.vf v{unroll+2}, v{self.unroll_level + unroll + 2}, f{i}\n' # value * exp(qk) / reduced sum
    template += f'add x4, x8, x3\n' # spad_addr + query_offset
    for unroll in range(self.unroll_level):
      template += f'vamoaddei32.v x0, (x4), v1, v{unroll + 2}\n' # Atomic sum
      if unroll != self.unroll_level - 1:
        template += f'addi x4, x4, {configs.packet_size}\n'
    template += f'addi x3, x3, {configs.packet_size * self.unroll_level}\n'
    template += f'blt x3, x6, .LOOP3\n'

    #Finalizer - store Atomic sum to output
    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x10, {spad_argument_addr}\n'
    template += f'ld x9, 24(x10)\n' # output_addr
    template += f'ld x3, 40(x10)\n' # head_dim
    template += f'ld x4, 32(x10)\n' # coefficient_addr'
    template += f'li x5, {spad_max_addr}\n' # spad_max_addr
    template += f'flw f1, (x5)\n' #load max
    template += f'flw f2, 4(x5)\n' #load reduced sum'
    template += f'slli x5, x2, 2\n'
    template += f'add x4, x4, x5\n' # coefficient_addr + OFFSET  * 4
    template += f'fsw f1, (x4)\n' # store max
    template += f'fsw f2, {configs.ndp_units * configs.data_size}(x4)\n' # store reduced sum
    template += f'mul x5, x3, x5\n' # head_dim * OFFSET * 4
    template += f'add x9, x9, x5\n' # output_addr + OFFSET * head_dim * 4
    template += f'li x10, {spad_output_addr}\n' # spad_output_addr
    template += f'li x6, 0\n' # vector index
    template += f'.LOOP4\n'
    for i in range(2, self.unroll_level + 2):
      template += f'vle32.v v{i}, (x10)\n'
      template += f'addi x10, x10, {configs.packet_size}\n'
    for i in range(2, self.unroll_level + 2):
      template += f'vse32.v v{i}, (x9)\n'
      template += f'addi x9, x9, {configs.packet_size}\n'
    template += f'addi x6, x6, {8 * self.unroll_level}\n' # 256 / float 32 = 8 elements in a row
    template += f'blt x6, x3, .LOOP4\n'
    return template

  def make_input_map(self):
    self.key_pad = np.zeros((self.repeat, self.max_context, self.head_dim )).astype(np.float32)
    self.key_pad[:,:self.context_len, :self.head_dim] = self.key
    self.value_pad = np.zeros((self.repeat, self.max_context, self.head_dim)).astype(np.float32)
    self.value_pad[:,:self.context_len,:] = self.value
    zero_outputs = np.zeros(self.local_attention.shape).astype(np.float32)
    # zero_outputs = np.zeros((1, self.repeat, self.max_context)).astype(np.float32)
    zero_coefficient = np.zeros((self.repeat * configs.ndp_units * 2)).astype(np.float32)
    return make_memory_map([
      (self.query_addr, self.query.reshape(-1)),
      (self.key_addr, self.key_pad.reshape(-1)),
      (self.value_addr, self.value_pad.reshape(-1)),
      (self.output_addr, zero_outputs.reshape(-1)),
      (self.coefficient_addr, zero_coefficient.reshape(-1))])

  def make_output_map(self):
    self.softmax_pad = np.zeros((self.repeat, self.max_context)).astype(np.float32)
    self.softmax_pad[:,:self.context_len] = self.softmax
    self.coefficient = np.zeros((self.repeat, 2, configs.ndp_units)).astype(np.float32)
    for head in range(self.repeat):
      for i in range(configs.ndp_units):
        self.coefficient[head, 0, i] = self.local_max[head, i]
        self.coefficient[head, 1, i] = self.local_sum[head, i]
    return make_memory_map([
      (self.query_addr, self.query.reshape(-1)),
      (self.key_addr, self.key_pad.reshape(-1)),
      (self.value_addr, self.value_pad.reshape(-1)),
      (self.output_addr, self.local_attention.reshape(-1)),
      (self.coefficient_addr, self.coefficient.reshape(-1))])



class OptAttention2Kernel(MultipleKernel):
  def __init__(self, config : str = 'opt-125m', context_len : int = 1024, max_context_len : int = 2048, model_parallelism=1):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path, 'r'))
    num_heads = config_file['num_attention_heads'] // model_parallelism
    # num_heads = 1
    super().__init__(repeat=num_heads)
    self.kernel_name = f'{config}-attention2'
    self.head_dim = config_file['hidden_size'] // num_heads // model_parallelism
    # self.head_dim = config_file['hidden_size'] // 12
    self.context_len = context_len
    self.max_context = max_context_len

    self.coefficient_addr = 0x830000000000
    self.output_addr = 0x840000000000
    self.reduced_output_addr = 0x850000000000
    self.unroll_level = 4
    self.base_addr = self.reduced_output_addr
    self.bound = self.head_dim * configs.data_size
    self.scale = 1.0 / math.sqrt(self.head_dim)
    self.input_addrs = [self.output_addr, # 0
                        self.coefficient_addr, # 8
                        self.head_dim] # 56
    self.input_addrs_offsets = [self.head_dim * configs.data_size * configs.ndp_units, #output_addr
                                configs.ndp_units * configs.data_size * 2, #coefficient_addr
                                0]
    self.query = np.random.randn(self.repeat, self.head_dim).astype(np.float32)
    # self.query = np.ones((self.repeat, self.head_dim)).astype(np.float32)
    self.key = np.random.randn(self.repeat, self.head_dim, self.context_len).astype(np.float32)
    # self.key = np.ones((self.repeat, self.head_dim, self.context_len)).astype(np.float32)

    self.value = np.random.randn(self.repeat,  self.context_len, self.head_dim).astype(np.float32)
    # self.value = np.ones((self.repeat, self.context_len, self.head_dim)).astype(np.float32)

    interleave = configs.stride // configs.data_size
    interleave_count = math.ceil(self.context_len / interleave)
    self.qk = np.zeros((self.repeat, self.context_len)).astype(np.float32)
    self.softmax = np.zeros((self.repeat, self.context_len)).astype(np.float32)
    for head in range(self.repeat):
      self.qk[head,:] = (self.query[head,:] @ self.key[head,:,:]).astype(np.float32)

    self.head_max = np.max(self.qk, axis=-1, keepdims=True).astype(np.float32)
    self.local_max = np.zeros((self.repeat, configs.ndp_units)).astype(np.float32)
    for head in range(self.repeat):
      for i in range(interleave_count):
        # ndp_id = i % configs.ndp_units
        ndp_id = (i + head * interleave_count) % configs.ndp_units
        self.local_max[head, ndp_id] = max(self.local_max[head, ndp_id], np.max(self.qk[head, i * interleave : (i+1) * interleave]))
    self.interleaved_max = np.zeros((self.repeat, context_len)).astype(np.float32)
    for head in range(self.repeat):
      for i in range(interleave_count):
        ndp_id = (i + head * interleave_count) % configs.ndp_units
        self.interleaved_max[head, i * interleave : (i+1) * interleave] = self.local_max[head,ndp_id]

    self.exp = np.exp(self.qk - self.head_max)
    self.interleaved_exp = np.exp(self.qk - self.interleaved_max)

    self.local_sum = np.zeros((self.repeat, configs.ndp_units)).astype(np.float32)
    for head in range(self.repeat):
      for i in range(interleave_count):
        ndp_id = (i + head * interleave_count) % configs.ndp_units
        self.local_sum[head, ndp_id] += np.sum(self.interleaved_exp[head, i * interleave : (i+1) * interleave])
    self.local_attention = np.zeros((self.repeat, configs.ndp_units, self.head_dim)).astype(np.float32)


    self.softmax = self.exp / np.sum(self.exp, axis=-1, keepdims=True)

    self.attention = np.zeros((self.repeat, self.head_dim)).astype(np.float32)


    for head in range(self.repeat):
      self.attention[head,:] = self.softmax[head,:] @ self.value[head,:,:]
    for head in range(self.repeat):
      for i in range(interleave_count):
        ndp_id = (i + head * interleave_count) % configs.ndp_units
        self.local_attention[head, ndp_id,:] += self.interleaved_exp[head, i * interleave : (i+1) * interleave] @ self.value[head, i * interleave : (i+1) * interleave,:]

    self.coefficient = np.zeros((self.repeat, 2, configs.ndp_units)).astype(np.float32)
    for head in range(self.repeat):
      for i in range(configs.ndp_units):
        self.coefficient[head, 0, i] = self.local_max[head, i]
        self.coefficient[head, 1, i] = self.local_sum[head, i]
    #Rescale local attention
    self.scaled_attention = np.zeros((self.repeat, self.head_dim)).astype(np.float32)
    for head in range(self.repeat):
      scaled_exp_sum = 0
      for ndp_id in range(configs.ndp_units):
        scaled_exp = self.local_max[head, ndp_id] - self.head_max[head,0]
        scaled_exp = np.exp(scaled_exp)
        scaled_exp_sum += scaled_exp * self.local_sum[head, ndp_id]
        self.scaled_attention[head,:] += self.local_attention[head, ndp_id,:] * scaled_exp
      # print(scaled_exp_sum)
      self.scaled_attention[head,:] /= scaled_exp_sum
    # print(np.isclose(self.attention, self.scaled_attention, rtol=1e-05, atol=1e-08, equal_nan=False))
    #Scratch pad memory allcation
    self.smem_size = configs.packet_size * 8

    self.sync = 0

  def make_kernel(self):
    ndp_unit_log2 = int(math.log2(configs.ndp_units))
    stride_log2 = int(math.log2(configs.stride))
    packet_size_log2 = int(math.log2(configs.packet_size))
    loop_count = math.ceil(configs.ndp_units / (configs.packet_size // configs.data_size))

    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'

    #Initializer - load query to spad
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x3, {configs.spad_addr}\n'
    template += f'ld x4, 8(x3)\n' # coefficient_addr
    template += f'vmv.v.i v1, 0\n'
    for i in range(loop_count):
      template += f'vle32.v v{i+2}, (x4)\n'
      template += f'vle32.v v{loop_count+i+2}, {configs.ndp_units * configs.data_size}(x4)\n'
      template += f'addi x4, x4, {configs.packet_size}\n'
    for i in range(loop_count):
      template += f'vfredomax.vs v1, v{i+2}, v1\n'
    template += f'vfmv.f.s f1, v1\n' # global max

    template += f'vmv.v.i v1, 0\n'
    for i in range(loop_count):
      #substract global max
      template += f'vfsub.vf v{i+2}, v{i+2}, f1\n'
      #exp
      template += f'vfexp.v v{i+2}, v{i+2}\n'
      template += f'vfmul.vv v{loop_count+i+2}, v{i+2}, v{loop_count+i+2}\n'
      #reduce sum
      template += f'vfredosum.vs v1, v{loop_count+i+2}, v1\n'
    template += f'vfmv.f.s f2, v1\n' # global sum
    template += f'li x4, 1\n'
    template += f'fmv.w.x f3, x4\n'
    template += f'fdiv f3, f3, f2\n'
    template += f'fsw f3, {configs.packet_size}(x3)\n'
    #store global max_exp
    template += f'addi x4, x3, {configs.packet_size*2}\n'
    for i in range(loop_count):
      template += f'vse32.v v{i+2}, (x4)\n'
      template += f'addi x4, x4, {configs.packet_size}\n'

    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x3, {configs.spad_addr}\n'
    template += f'ld x4, (x3)\n' # output addr
    template += f'flw f1, {configs.packet_size}(x3)\n' # 1 / global sum
    template += f'vmv.v.i v1, 0\n'
    template += f'add x4, x4, x2\n'
    for i in range(loop_count):
      template += f'vle32.v v2, {configs.packet_size*2}(x3)\n' # global max_exp
      template += f'addi x3, x3, {configs.packet_size}\n'
      for ndp in range(configs.packet_size // configs.data_size):
        template += f'vle32.v v{loop_count+2+ndp}, (x4)\n'
        template += f'addi x4, x4, {self.head_dim * configs.data_size}\n'
      for ndp in range(configs.packet_size // configs.data_size):
        template += f'vfmv.f.s f0, v2\n'
        template += f'vfmacc.vf v1, v{loop_count+2+ndp}, f0\n'
        template += f'vslide1down.vx v2, v2, x0\n'
    template += f'vfmul.vf v1, v1, f1\n'
    template += f'vse32.v v1, (x1)\n'
    return template

  def make_input_map(self):
    return make_memory_map([
      (self.output_addr, self.local_attention.reshape(-1)),
      (self.coefficient_addr, self.coefficient.reshape(-1))])

  def make_output_map(self):
    return make_memory_map([
      (self.output_addr, self.local_attention.reshape(-1)),
      (self.reduced_output_addr, self.scaled_attention.reshape(-1)),
      (self.coefficient_addr, self.coefficient.reshape(-1))])

