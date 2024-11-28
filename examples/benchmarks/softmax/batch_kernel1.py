from typing import Any
import numpy as np
import os
import math
from utils.utils import NdpKernel, make_memory_map, pad8
import configs
np.random.seed(0)

# Reduce Max Kernel for SoftMax 
class BatchSoftMaxKernel1(NdpKernel):
  def __init__(self):
    super().__init__()
    self.group_number = 32
    self.vector_size = int(os.getenv("SOFTMAX_SIZE",default=1024))
    self.input_data_addr =    0x800000000000
    self.input_max_addr =     0x810000000000
    self.output_addr =        0x820000000000
    self.reduce_output_addr = 0x830000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'batch_softmax_kernel1'
    self.base_addr = self.input_data_addr
    self.bound = self.vector_size * self.group_number * configs.data_size

    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")

    #input_data = np.random.rand(self.group_number, self.vector_size).astype(data_type)
    input_data = np.arange(self.vector_size * self.group_number).astype(data_type)#np.random.rand(self.vector_size).astype(np.float32)
    input_data = input_data.reshape(self.group_number, self.vector_size)
    input_max_data = np.max(input_data, axis=-1, keepdims=True)
    outputs = np.exp(input_data-input_max_data)
    reduce_outputs = np.sum(outputs, axis=-1, keepdims=True).astype(np.float32)

    self.input_data = input_data
    self.input_max_data = input_max_data.astype(np.float32)
    self.outputs = outputs
    self.reduce_outputs = reduce_outputs
  
    self.input_addrs = [self.output_addr, self.reduce_output_addr, self.group_number, self.input_max_addr]


  def make_kernel(self):
    data_size = configs.data_size
    self.group_level = int(math.log2(self.vector_size*data_size))
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'

    # Initializer
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'li x2, {configs.spad_addr + 2*configs.packet_size}\n' # Start Addr of buffer in shared mem
    template += f'ld x3, 16(x1)\n'                                      # Number of Group
    template += f'li x4, 0\n'
    template += f'ld x5, 24(x1)\n'

    template += f'slli x6, x3, 5\n'                                      # Size of scratch buffer
    template += f'add x6, x2, x6\n'

    template += f'vmv.v.i v1, 0\n'
    template += f'vfcvt.f.x.v v1, v1\n'

    template += f'.LOOP1\n'
    template += f'vse{data_size*8}.v v1, (x2)\n'                # Scratch pad Initialize with 0
    template += f'flw f1, (x5)\n'
    template += f'fsw f1, (x6)\n'
    template += f'addi x4, x4, 1\n'
    template += f'addi x2, x2, {configs.packet_size}\n'
    template += f'addi x5, x5, {4}\n'
    template += f'addi x6, x6, {4}\n'
    template += f'bne x3, x4, .LOOP1\n'

    # Body
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'vle{data_size*8}.v v1, (ADDR)\n'
    # Shared atomic operation
    template += f'li x1, {configs.spad_addr + 2*configs.packet_size}\n' # Scratch pad addr
    template += f'li x2, {configs.spad_addr}\n' 
    template += f'ld x8, 16(x2)\n'
    template += f'ld x2, (x2)\n'

    # Calc group id
    template += f'srli x3, OFFSET, {self.group_level}\n'                # Is it cheating? Maybe..?
    template += f'slli x4, x3, {int(math.log2(configs.packet_size))}\n'
    template += f'slli x5, x3, {2}\n'
    template += f'add x1, x1, x4\n'

    template += f'li x6, {configs.spad_addr + 2*configs.packet_size}\n'
    template += f'slli x8, x8, {5}\n'
    template += f'add x6, x6, x8\n'                                     # += packet_size * group_nr
    template += f'add x6, x6, x5\n'                                     # += group_id * 4
    template += f'flw f3, (x6)\n'

    template += f'vmv.v.i v2, 0\n'
    template += f'vfcvt.f.x.v v2, v2\n'
 
    template += f'vfsub.vf v1, v1, f3\n'
    template += f'add x2, x2, OFFSET\n'
    template += f'vfexp.v v1, v1\n'
    template += f'vse{data_size*8}.v v1, (x2)\n'

    if configs.data_size == 2:
      template += f'vwredosum.vs v3, v1, v2\n'
      template += f'vsetvli t0, a0, e32, m1\n'
    elif configs.data_size == 4:
      template += f'vfredosum.vs v3, v1, v2\n'
    else:
       raise Exception("Unsupported data type")
    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vsll.vi v2, v2, 2\n'
    template += f'vamoaddei32.v x0, (x1), v2, v3\n'

    # Finalizer
    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, 8(x1)\n'                                # x2 = reduce_output
    template += f'ld x3, 16(x1)\n'                               # x3 = group
    template += f'addi x1, x1, {2*configs.packet_size}\n'

    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vsll.vi v2, v2, {2}\n'
    template += f'li x6, 0\n'

    template += f'.LOOP2\n'
    template += f'vle32.v v1, (x1)\n'  
    template += f'vamoaddei32.v x0, (x2), v2, v1\n'
  
    # Increment
    template += f'addi x6, x6, 1\n'
    template += f'addi x1, x1, {configs.packet_size}\n'
    template += f'addi x2, x2, {4}\n'

    template += f'bne x3, x6, .LOOP2\n'
    return template
  
  def make_input_map(self):
   return make_memory_map([
        (self.input_data_addr, self.input_data), (self.input_max_addr, self.input_max_data)
    ])
  
  def make_output_map(self):
    return make_memory_map([
        (self.input_data_addr, self.input_data), (self.input_max_addr, self.input_max_data),
        (self.output_addr, self.outputs), (self.reduce_output_addr, self.reduce_outputs)])
  