from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

# Reduce Max Kernel for SoftMax 

class SoftMaxKernel0(NdpKernel):
  def __init__(self):
    super().__init__()
    self.vector_size = int(os.getenv("SOFTMAX_SIZE",default=1024))
    self.input_data_addr = 0x810000000000
    self.output_max_addr = 0x890000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'softmax_kernel0'
    self.base_addr = self.input_data_addr
    self.bound = self.vector_size * configs.data_size
    self.input_addrs = [self.output_max_addr]
    self.smem_size = configs.packet_size

    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")

    input_data = np.arange(self.vector_size).astype(data_type)#np.random.rand(self.vector_size).astype(np.float32)
    outputs = np.max(input_data, keepdims=True)
    
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
    template += f'li x1, {configs.spad_addr + configs.packet_size}\n'
    template += f'vle{data_size*8}.v v1, (ADDR)\n'
    template += f'vse{data_size*8}.v v1, (x1)\n'                # Initialize with first packet

    # Body
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    # Shared atomic operation
    template += f'li x1, {configs.spad_addr + configs.packet_size}\n'
    template += f'vle{data_size*8}.v v1, (ADDR)\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vmul.vi v2, v2, {data_size}\n'
    template += f'vamomaxei{data_size*8}.v x0, (x1), v2, v1\n'

    # Finalizer
    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, (x1)\n'                                # x2 = output_max_addr
    template += f'addi x1, x1, {configs.packet_size}\n'
    template += f'vle{data_size*8}.v v1, (x1)\n'                           # v1 = spad_addr + packet_size[]
    template += f'vfredomax.vs v1, v1, v1\n'                    # v1 = reduce(v1)

    # Global atomic operation
    """ 
    For debug code below
    template += f'muli x3, NDPID, {configs.packet_size}\n'
    template += f'addi x4, x2, {configs.packet_size}\n'
    template += f'add x4, x4, x3\n'
    template += f'vse{data_size*8}.v v1, (x4)\n'
    """

    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n'
    template += f'vmul.vi v2, v2, {data_size}\n'
    # FIXME. it should be sacalar amomaxei
    template += f'vamomaxei{data_size*8}.v x0, (x2), v2, v1\n'
    return template
  
  def make_input_map(self):
   return make_memory_map([
        (self.input_data_addr, self.input_data),
    ])
  
  def make_output_map(self):
    return make_memory_map([
        (self.input_data_addr, self.input_data),
        (self.output_max_addr, self.outputs)])
  