from typing import Any
import numpy as np
import os
import math
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

# Reduce Max Kernel for SoftMax 
np.random.seed(0)
class NewBatchSoftMaxKernel1(NdpKernel):
  def __init__(self):
    super().__init__()
    self.group_number = 32    # This value is fixed...
    self.vector_size = int(os.getenv("SOFTMAX_SIZE",default=1024))
    self.input_data_addr =    0x800000000000
    self.input_max_addr =     0x810000000000
    self.output_addr =        0x820000000000
    self.reduce_output_addr = 0x830000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'batch_softmax_kernel1'
    self.base_addr = self.input_data_addr
    self.unroll_level = 4   # How many read vector register
    self.mask = (self.group_number * configs.data_size // configs.packet_size) - 1
    self.bound = self.vector_size * self.group_number * configs.data_size // self.unroll_level

    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")

    input_data = np.random.rand(self.vector_size * self.group_number).astype(data_type)
    #input_data = np.arange(self.vector_size * self.group_number).astype(data_type)
    input_data = input_data.reshape(self.group_number, self.vector_size)
    input_max_data = np.max(input_data, axis=-1, keepdims=True)
    outputs = np.exp(input_data-input_max_data).astype(data_type)
    reduce_outputs = np.sum(outputs, axis=-1, keepdims=True).astype(np.float32)

    self.input_data = input_data
    self.input_max_data = input_max_data
    self.outputs = outputs
    self.reduce_outputs = reduce_outputs
  
    self.input_addrs = [self.input_data_addr, self.output_addr, self.reduce_output_addr, self.input_max_addr, self.group_number * configs.data_size, int(math.log2(self.unroll_level)), self.mask]
    self.input_data = np.transpose(self.input_data)
    self.outputs = np.transpose(self.outputs)
  
  def make_kernel(self):
    data_size = configs.data_size
    bit_size = data_size * 8
    self.group_level = int(math.log2(self.vector_size*data_size))
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'

    # Initializer
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e{bit_size}, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'addi x2, x1, {configs.packet_size*2}\n'
    template += f'ld x3, 32(x1)\n'        # y stride
    if configs.data_size == 2:
      template += f'slli x3, x3, 1\n'
    template += f'li x4, 0\n'
    template += f'ld x5, 24(x1)\n'

    template += f'vsetvli t0, a0, e32, m2\n'
    template += f'vmv.v.i v0, 0\n'
    template += f'vfcvt.f.x.v v0, v0\n'                                 # Should add this...

    # Reduce spad addr
    template += f'.LOOP1\n'
    template += f'vse32.v v0, (x2)\n'
    template += f'addi x2, x2, {configs.packet_size*2}\n'
    template += f'addi x4, x4, {configs.packet_size*2}\n'
    template += f'bne x3, x4, .LOOP1\n'

    if configs.data_size == 2:
      template += f'vsetvli t0, a0, e16, m1\n'
      template += f'srli x3, x3, 1\n'
    else:
      template += f'vsetvli t0, a0, e32, m1\n'
    # Load input max addr 
    template += f'li x4, 0\n'
    template += f'.LOOP2\n'
    template += f'vle{bit_size}.v v1, (x5)\n'
    template += f'vse{bit_size}.v v1, (x2)\n'
    template += f'addi x5, x5, {configs.packet_size}\n'
    template += f'addi x2, x2, {configs.packet_size}\n'
    template += f'addi x4, x4, {configs.packet_size}\n'
    template += f'bne x3, x4, .LOOP2\n'

    # Body
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e{bit_size}, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, (x1)\n'          # input
    template += f'ld x3, 8(x1)\n'         # output
    template += f'ld x4, 32(x1)\n'        # y stride
    template += f'ld x6, 48(x1)\n'        # mask

    template += f'srli x7, OFFSET, 5\n'   # thread id
    template += f'and x8, x7, x6\n'       # thread id.x
    template += f'sub x9, x7, x8\n'       # thread id.y
    template += f'srli x9, x9, 1\n'       # TODO. mask shift is hardcoded...

    template += f'slli x10, x9, 8\n'      # TODO. packet_size * (group_size * datasize/packetsize) * unroll_level
    template += f'add x2, x2, x10\n'      # x2 = input_addr + y * stride_size
    template += f'add x3, x3, x10\n'      # x3 = output_addr + y * stride_size

    template += f'slli x8, x8, 5\n'       # thread id.x * 32
    if configs.data_size == 2:
      template += f'slli x9, x8, 1\n'       # thread id.x * 64
    else:
      template += f'addi x9, x8, 0\n'

    template += f'add x9, x9, x1\n'       # x9 = spad_addr + x * 64
    template += f'addi x9, x9, {configs.packet_size*2}\n'

    template += f'add x2, x2, x8\n'       # x2 = input_addr + y * stride_size + x * 32
    template += f'add x3, x3, x8\n'       # x3 = output_addr + y * stride_size + x * 32
    
    if configs.data_size == 2:
      template += f'slli x5, x4, 1\n'       # x5 = stride * 2
    else:
      template += f'addi x5, x4, 0\n'       # x5 = stride

    template += f'addi x8, x8, {configs.packet_size*2}\n'
    template += f"add x8, x1, x8\n"       # shared mem addr + 64 + thread id.x * 32
    
    template += f'add x10, x8, x5\n'      # input max addr
    template += f'vle{bit_size}.v v0, (x10)\n'    # input max

    template += f'vle{bit_size}.v v1, (x2)\n'
    template += f'add x2, x2, x4\n'
    template += f'vle{bit_size}.v v2, (x2)\n'
    template += f'add x2, x2, x4\n'
    template += f'vle{bit_size}.v v3, (x2)\n'
    template += f'add x2, x2, x4\n'
    template += f'vle{bit_size}.v v4, (x2)\n'

    template += f'vfsub.vv v1, v1, v0\n'
    template += f'vfsub.vv v2, v2, v0\n'
    template += f'vfsub.vv v3, v3, v0\n'
    template += f'vfsub.vv v4, v4, v0\n'

    template += f'vfexp.v v1, v1\n'
    template += f'vfexp.v v2, v2\n'
    template += f'vfexp.v v3, v3\n'
    template += f'vfexp.v v4, v4\n'

    template += f'vse{bit_size}.v v1, (x3)\n'
    template += f'add x3, x3, x4\n'
    template += f'vse{bit_size}.v v2, (x3)\n'
    template += f'add x3, x3, x4\n'
    template += f'vse{bit_size}.v v3, (x3)\n'
    template += f'add x3, x3, x4\n'
    template += f'vse{bit_size}.v v4, (x3)\n'

    if configs.data_size == 2:
      template += f'vsetvli t0, a0, e32, m2\n'
      template += f'vfwcvt.f.f.v v5, v1\n'
      template += f'vfwcvt.f.f.v v6, v2\n'
      template += f'vfwcvt.f.f.v v7, v3\n'
      template += f'vfwcvt.f.f.v v8, v4\n'

      template += f'vfadd.vv v5, v5, v6\n'
      template += f'vfadd.vv v7, v7, v8\n'
      template += f'vfadd.vv v5, v5, v7\n'
    else:
      template += f'vfadd.vv v1, v1, v2\n'
      template += f'vfadd.vv v2, v3, v4\n'
      template += f'vfadd.vv v5, v1, v2\n'

    template += f'vmset.m v9\n'
    template += f'vid.v v10, v9\n'
    template += f'vsll.vi v10, v10, 2\n'
    template += f'vamoaddei32.v x0, (x9), v10, v5\n'

    # Finalize
    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e{bit_size}, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'addi x2, x1, {configs.packet_size*2}\n'               # Start share mem addr
    template += f'ld x3, 16(x1)\n'                                      # output addr

    # For debug
    # template += f'muli x7, NDPID, 128\n'
    # template += f'add x3, x3, x7\n'
 
    template += f'ld x4, 32(x1)\n'                                      # y stride
    template += f'slli x4, x4, 1\n'                                     # x4 = y stride * 2
    template += f'li x5, 0\n'
 
    if configs.data_size == 2:
      template += f'vsetvli t0, a0, e32, m2\n'
      template += f'vmset.m v0\n'
      template += f'vid.v v2, v0\n'
      template += f'vsll.vi v2, v2, 2\n'

      template += f'.LOOP3\n'
      template += f'vle32.v v1, (x2)\n'
      template += f'vamoaddei32.v x0, (x3), v2, v1\n'
      template += f'addi x2, x2, {configs.packet_size*2}\n'                 # increase shared mem addr
      template += f'addi x3, x3, {configs.packet_size*2}\n'                 # increase ouput addr
      template += f'addi x5, x5, {configs.packet_size*2}\n'                 # conditions
      template += f'bne x4, x5, .LOOP3\n'
    else:
      template += f'vsetvli t0, a0, e32, m1\n'
      template += f'vmset.m v0\n'
      template += f'vid.v v2, v0\n'
      template += f'vsll.vi v2, v2, 2\n'
 
      template += f'.LOOP3\n'
      template += f'vle32.v v1, (x2)\n'
      template += f'vamoaddei32.v x0, (x3), v2, v1\n'
      template += f'addi x2, x2, {configs.packet_size}\n'                 # increase shared mem addr
      template += f'addi x3, x3, {configs.packet_size}\n'                 # increase ouput addr
      template += f'addi x5, x5, {configs.packet_size}\n'                 # conditions
      template += f'bne x4, x5, .LOOP3\n'
    return template

  def make_input_map(self):
   return make_memory_map([
        (self.input_data_addr, self.input_data), (self.input_max_addr, self.input_max_data)
    ])
  
  def make_output_map(self):
    return make_memory_map([
        (self.input_data_addr, self.input_data), (self.input_max_addr, self.input_max_data),
        (self.output_addr, self.outputs), (self.reduce_output_addr, self.reduce_outputs)])
  