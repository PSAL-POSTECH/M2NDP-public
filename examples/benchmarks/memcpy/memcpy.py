from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class MemCpyKernel(NdpKernel):
    def __init__(self):
        super().__init__()
        self.vector_size = int(os.getenv("MEMCPY_BYTES", default=1024)) // configs.data_size
        self.max_val = 4096
        self.input_a_addr = 0x800000000000
        self.output_addr =  0x820000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'memcpy'
        self.base_addr = self.input_a_addr
        self.bound = self.vector_size * configs.data_size
        self.input_a = np.random.randint(size=[self.vector_size], high=self.max_val, low=-self.max_val, dtype=np.int32)
        self.output = self.input_a
        self.input_addrs = [self.output_addr]

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        data_bits = configs.data_size*8
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e{data_bits}, m1, 0\n'
        template += f'li x1, {spad_addr }\n'
        template += f'ld x1, (x1)\n'
        template += f'add x1, x1, OFFSET\n'
        template += f'vle{data_bits}.v v1, (ADDR)\n'
        template += f'vse{data_bits}.v v1, (x1)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a)])

    def make_output_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a), (self.output_addr, self.output)])