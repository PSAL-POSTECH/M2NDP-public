from typing import Any
import numpy as np
import os
import math
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

# Reduce Max Kernel for SoftMax 
np.random.seed(0)
class NewBatchSoftMaxKernel0(NdpKernel):
  def __init__(self):
    super().__init__()
    self.group_number = 32    # TODO. This value is fixed...
    self.vector_size = int(os.getenv("SOFTMAX_SIZE",default=1024))
    self.input_data_addr = 0x800000000000
    self.output_max_addr = 0x810000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'batch_softmax_kernel0'
    self.base_addr = self.input_data_addr
    self.unroll_level = 4   # TODO. dynamically change, How many read vector register
    self.mask = (self.group_number * configs.data_size // configs.packet_size) - 1
    self.bound = self.vector_size * self.group_number * configs.data_size // self.unroll_level
    self.input_addrs = [self.input_data_addr, self.output_max_addr, self.group_number * configs.data_size, int(math.log2(self.unroll_level)), self.mask]
    self.smem_size = configs.packet_size * 4
    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")

    input_data = np.random.rand(self.vector_size * self.group_number).astype(data_type)
    #input_data = np.arange(self.vector_size * self.group_number).astype(data_type)
    self.input_data = input_data.reshape(self.group_number, self.vector_size)
    self.outputs = np.max(self.input_data, axis=-1)
    self.input_data = np.transpose(self.input_data)
  
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
    template += f'ld x3, 16(x1)\n'        # y stride
    template += f'li x4, 0\n'
    template += f'vmv.v.i v0, 0\n'
    template += f'vfcvt.f.x.v v0, v0\n'                                 # Should add this...

    template += f'.LOOP1\n'
    template += f'vse{bit_size}.v v0, (x2)\n'
    template += f'addi x2, x2, {configs.packet_size}\n'
    template += f'addi x4, x4, {configs.packet_size}\n'
    template += f'bne x3, x4, .LOOP1\n'

    # Body
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e{bit_size}, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, (x1)\n'          # input
    template += f'ld x4, 16(x1)\n'        # y stride
    template += f'ld x6, 32(x1)\n'        # mask

    template += f'srli x7, OFFSET, 5\n'   # thread id
    template += f'and x8, x7, x6\n'       # thread id.x
    template += f'sub x9, x7, x8\n'       # thread id.y
    template += f'srli x9, x9, 1\n'       # TODO. mask shift is hardcoded...

    template += f'slli x10, x9, 8\n'      # TODO. packet_size * (group_size * datasize/packetsize) * unroll_level
    template += f'add x2, x2, x10\n'

    template += f'slli x8, x8, 5\n'
    template += f'add x2, x2, x8\n'
    
    template += f'addi x8, x8, {configs.packet_size*2}\n'
    template += f"add x8, x1, x8\n"       # shared mem addr

    template += f'vle{bit_size}.v v1, (x2)\n'
    template += f'add x2, x2, x4\n'
    template += f'vle{bit_size}.v v2, (x2)\n'
    template += f'add x2, x2, x4\n'
    template += f'vle{bit_size}.v v3, (x2)\n'
    template += f'add x2, x2, x4\n'
    template += f'vle{bit_size}.v v4, (x2)\n'

    template += f'vmax.vv v1, v1, v2\n'
    template += f'vmax.vv v1, v1, v3\n'
    template += f'vmax.vv v1, v1, v4\n'

    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vsll.vi v2, v2, {int(math.log2(data_size))}\n'
    template += f'vamomaxei{bit_size}.v x0, (x8), v2, v1\n'

    # Finalize
    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e{bit_size}, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'addi x2, x1, {configs.packet_size*2}\n'                # Start share mem addr
    template += f'ld x3, 8(x1)\n'                                       # output addr
 
    template += f'ld x4, 16(x1)\n'        # y stride
    template += f'li x5, 0\n'
 
    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vsll.vi v2, v2, {int(math.log2(data_size))}\n'
 
    template += f'.LOOP2\n'
    template += f'vle{bit_size}.v v1, (x2)\n'
    template += f'vamomaxei{bit_size}.v x0, (x3), v2, v1\n'
    template += f'addi x2, x2, {configs.packet_size}\n'                 # increase shared mem addr
    template += f'addi x3, x3, {configs.packet_size}\n'                 # increase ouput addr
    template += f'addi x5, x5, {configs.packet_size}\n'                 # conditions
    template += f'bne x4, x5, .LOOP2\n'
    return template
  
  def make_input_map(self):
   return make_memory_map([
        (self.input_data_addr, self.input_data), 
    ])
  
  def make_output_map(self):
    return make_memory_map([
        (self.input_data_addr, self.input_data),
        (self.output_max_addr, self.outputs)])
  