from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class MemSetKernel(NdpKernel):
    def __init__(self):
        super().__init__()
        self.vector_size = int(os.getenv("MEMSET_BYTES", default=1024))
        self.output_addr =  0x820000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'memset'
        self.base_addr = self.output_addr
        self.bound = self.vector_size * 1
        self.target_byte = 0xdf
        self.input = np.random.randint(-128, 127, size=[self.vector_size], dtype=np.int8)
        self.output = np.full(self.vector_size, self.target_byte, dtype=np.int8)
        self.input_addrs = [self.target_byte]

    def make_kernel(self):
        packet_size = configs.packet_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e8, m1, 0\n'
        template += f'li x1, {spad_addr}\n'
        template += f'ld x1, (x1)\n'
        template += f"vmv.v.x v1, x1\n"
        template += f'vse8.v v1, (ADDR)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.output_addr, self.input)])

    def make_output_map(self):
      return make_memory_map([(self.output_addr, self.output)])