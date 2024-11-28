from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs
import json

class OptVectorAdd(NdpKernel):
    def __init__(self, size : int = 1024*1024):
        super().__init__()
        self.vector_size = size
        self.input_a_addr = 0x800000000000
        self.input_b_addr = 0x810000000000
        self.output_addr =  0x820000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'vector_add'
        self.base_addr = self.input_a_addr
        self.bound = self.vector_size * configs.data_size
        self.input_a = np.random.rand(self.vector_size).astype(np.float32)#
        self.input_b = np.random.rand(self.vector_size).astype(np.float32)#
        self.output = self.input_a + self.input_b
        self.input_addrs = [self.input_b_addr, self.output_addr]

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x3, {spad_addr }\n'
        template += f'li x4, {spad_addr + 8}\n'
        template += f'ld x3, (x3)\n'
        template += f'ld x4, (x4)\n'
        template += f'add x3, x3, x2\n'
        template += f'add x4, x4, x2\n'
        template += f'vle32.v v1, (x1)\n'
        template += f'vle32.v v2, (x3)\n'
        template += f'vadd.vv v3, v1, v2\n'
        template += f'vse32.v v3, (x4)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a), (self.input_b_addr, self.input_b)])

    def make_output_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a), (self.input_b_addr, self.input_b),
                              (self.output_addr, self.output)])

class OptResidual(OptVectorAdd):
  def __init__(self, config : str = 'opt-125m'):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path, 'r'))
    self.vector_size = config_file['hidden_size']
    super().__init__(self.vector_size)


