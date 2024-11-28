from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class NarrowKernel(NdpKernel):
    def __init__(self):
        super().__init__()
        self.vector_size = 1024
        self.max_val = 4096
        self.input_addr = 0x100000000000000
        self.output_addr = 0x102000000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'narrow'
        self.base_addr = self.output_addr   # Note! that base addr is output addr
        self.bound = self.vector_size * 2
        self.input = np.random.randn(self.vector_size).astype(np.float32)
        self.output = self.input.astype(np.float16)
        self.input_addrs = [self.input_addr]

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e32, m2, 0\n'
        template += f'li x1, {spad_addr}\n'
        template += f'ld x1, (x1)\n'
        template += f'muli x2, OFFSET, 2\n'
        template += f'add x1, x1, x2\n'
        template += f'vle32.v v2, (x1)\n'
        template += f'vsetvli 0, 0, e16, m1, 0\n'
        template += f'vfncvt.f.f.w v5, v2\n'
        template += f'vse32.v v5, (ADDR)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_addr, self.input)])

    def make_output_map(self):
      return make_memory_map([(self.input_addr, self.input),
                              (self.output_addr, self.output)])
