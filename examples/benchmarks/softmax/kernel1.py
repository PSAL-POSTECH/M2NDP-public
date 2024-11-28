from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

# Reduce Max Kernel for SoftMax 
class SoftMaxKernel1(NdpKernel):
  def __init__(self):
    super().__init__()
    self.vector_size = int(os.getenv("SOFTMAX_SIZE",default=1024))
    self.input_data_addr = 0x810000000000
    self.input_max_addr = 0x820000000000
    self.output_addr = 0x890000000000
    self.reduce_output_addr = 0x8a0000000000
    self.sync = 0
    self.kernel_id = 1
    self.kernel_name = f'softmax_kernel1'
    self.base_addr = self.input_data_addr
    self.bound = self.vector_size * configs.data_size
    self.smem_size = configs.packet_size * 4
    
    if configs.data_size == 4:
      data_type = np.float32
    elif configs.data_size == 2:
      data_type = np.float16
    else:
      raise Exception("Unsupported data type")

    input_data = np.random.rand(self.vector_size).astype(data_type)
    input_max_data = np.max(input_data, keepdims=True)
    outputs = np.exp(input_data-input_max_data)
    reduce_outputs = np.sum(outputs, keepdims=True).astype(np.float32)
    
    self.input_data = input_data
    self.input_max_data = input_max_data
    self.outputs = outputs
    self.reduce_outputs = reduce_outputs
  
    self.input_addrs = [self.output_addr, self.reduce_output_addr, 0, 0, 0, 0, 0,
                        input_max_data[0].astype(np.float32)]


  def make_kernel(self):
    data_size = configs.data_size
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    # Initializer
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e{32}, m1\n'
    template += f'li x1, {configs.spad_addr + 3*configs.packet_size}\n'
    template += f'vmv.v.i v1, 0\n'
    template += f'vfcvt.f.x.v v1, v1\n'
    template += f'vse{32}.v v1, (x1)\n'                # Scratch pad Initialize with 0

    # Body
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
    # Shared atomic operation
    template += f'li x1, {configs.spad_addr + 3*configs.packet_size}\n' # Scratch pad addr
    template += f'li x2, {configs.spad_addr}\n'                       # 
    template += f'ld x2, (x2)\n'
    template += f'add x2, x2, OFFSET\n'
    template += f'li x3, {configs.spad_addr + 64}\n'
    template += f'flw f3, (x3)\n'

    template += f'vle{data_size*8}.v v1, (ADDR)\n'
    template += f'vfsub.vf v1, v1, f3\n'
    template += f'vfexp.v v1, v1\n'
    template += f'vse{data_size*8}.v v1, (x2)\n'

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
    template += f'vamoaddei32.v x0, (x1), v2, v3\n'

    # Finalizer
    template += f'FINALIZER:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {configs.spad_addr}\n'
    template += f'ld x2, 8(x1)\n'                                # x2 = reduce_output
    template += f'addi x1, x1, {3*configs.packet_size}\n'
    template += f'vle32.v v1, (x1)\n'   

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
    template += f'vmul.vi v2, v2, 4\n'
    template += f'vamoaddei32.v x0, (x2), v2, v1\n'
    return template
  
  def make_input_map(self):
   return make_memory_map([
        (self.input_data_addr, self.input_data), (self.input_max_addr, self.input_max_data)
    ])
  
  def make_output_map(self):
    return make_memory_map([
        (self.input_data_addr, self.input_data), (self.input_max_addr, self.input_max_data),
        (self.output_addr, self.outputs), (self.reduce_output_addr, self.reduce_outputs)])
  