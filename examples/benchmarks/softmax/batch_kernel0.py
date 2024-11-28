from typing import Any
import numpy as np
import os
import math
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

# Reduce Max Kernel for SoftMax 

class BatchSoftMaxKernel0(NdpKernel):
  def __init__(self):
    super().__init__()
    self.group_number = 16
    self.vector_size = int(os.getenv("SOFTMAX_SIZE",default=1024))
    self.input_data_addr = 0x800000000000
    self.output_max_addr = 0x810000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'batch_softmax_kernel0'
    self.base_addr = self.input_data_addr
    self.bound = self.vector_size * self.group_number * configs.data_size
    self.input_addrs = [self.output_max_addr, self.group_number]

    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")

    self.smem_size = 1024
    input_data = np.arange(self.vector_size * self.group_number).astype(data_type)#np.random.rand(self.vector_size).astype(np.float32)
    self.input_data = input_data.reshape(self.group_number, self.vector_size)
    self.outputs = np.max(self.input_data, axis=-1)
  
  def make_kernel(self):
    data_size = configs.data_size
    self.group_level = int(math.log2(self.vector_size*data_size))
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'

    # Initializer
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'li x2, {configs.spad_addr + 2*configs.packet_size}\n' # Start Addr of buffer in shared mem
    template += f'ld x3, 8(x1)\n'                                       # Number of Group
    template += f'li x4, 0\n'
    template += f'vmv.v.i v0, 0\n'
    template += f'vfcvt.f.x.v v0, v0\n'                                 # Should add this...

    template += f'.LOOP1\n'
    template += f'vse{data_size*8}.v v0, (x2)\n'
    template += f'addi x4, x4, 1\n'
    template += f'addi x2, x2, {configs.packet_size}\n'
    template += f'bne x3, x4, .LOOP1\n'

    # Body
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    # Shared atomic operation
    template += f'li x1, {configs.spad_addr + 2*configs.packet_size}\n'
    template += f'srli x2, OFFSET, {self.group_level}\n'                # Is it cheating? Maybe..?
    template += f'slli x2, x2, {int(math.log2(configs.packet_size))}\n'
    template += f'add x1, x1, x2\n'

    template += f'vle{data_size*8}.v v1, (ADDR)\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vsll.vi v2, v2, {int(math.log2(data_size))}\n'
    template += f'vfredomax.vs v1, v1, v1\n'                    # v1 = reduce(v1)
    template += f'vamomaxei{data_size*8}.v x0, (x1), v2, v1\n'

    # Finalizer
    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, (x1)\n'                                # x2 = output_max_addr
    template += f'ld x3, 8(x1)\n'                               # x3 = group
    template += f'addi x1, x1, {2*configs.packet_size}\n'

    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vsll.vi v2, v2, {int(math.log2(data_size))}\n'
    template += f'li x6, 0\n'

    template += f'.LOOP2\n'
    template += f'vle{data_size*8}.v v1, (x1)\n'                # v1 = spad_addr + packet_size[]

    # Global atomic operation
    """ 
    For debug code below
    template += f'muli x3, NDPID, {configs.packet_size}\n'
    template += f'addi x4, x2, {configs.packet_size}\n'
    template += f'add x4, x4, x3\n'
    template += f'vse{data_size*8}.v v1, (x4)\n'
    """
    # FIXME. it should be sacalar amomaxei
    template += f'vamomaxei{data_size*8}.v x0, (x2), v2, v1\n'

    # Increment
    template += f'addi x6, x6, 1\n'
    template += f'addi x1, x1, {configs.packet_size}\n'
    template += f'addi x2, x2, {data_size}\n'

    template += f'bne x3, x6, .LOOP2\n'
    return template
  
  def make_input_map(self):
   return make_memory_map([
        (self.input_data_addr, self.input_data), (0x103000000000000,self.outputs.astype(np.float32))
    ])
  
  def make_output_map(self):
    return make_memory_map([
        (self.input_data_addr, self.input_data),
        (self.output_max_addr, self.outputs)])
  