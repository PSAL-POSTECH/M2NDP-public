from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class UnrolledVectorAddKernel(NdpKernel):
    def __init__(self):
        super().__init__()
        self.vector_size = 1024 * 1024
        self.max_val = 4096
        self.input_a_addr = 0x100000000000000
        self.input_b_addr = 0x101000000000000
        self.output_addr = 0x102000000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'vector_add'
        self.base_addr = self.input_a_addr
        self.bound = self.vector_size * configs.data_size // configs.unroll_level
        if configs.data_size == 4:
          data_type = np.float32
        elif configs.data_size == 2:
          data_type = np.float16
        else:
          raise Exception("Unsupported data type")
        self.input_a = np.random.randint(size=[self.vector_size], high=self.max_val, low=-self.max_val).astype(data_type)
        self.input_b = np.random.randint(size=[self.vector_size], high=self.max_val, low=-self.max_val).astype(data_type)
        self.output = self.input_a + self.input_b
        self.input_addrs = [self.input_a_addr, self.input_b_addr, self.output_addr]

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        unroll_level = configs.unroll_level
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e{configs.data_size * 8}, m1, 0\n'
        template += f'li x1, {spad_addr }\n'
        template += f'li x2, {spad_addr + 8}\n'
        template += f'li x3, {spad_addr + 16}\n'
        template += f'ld x1, (x1)\n'
        template += f'ld x2, (x2)\n'
        template += f'ld x3, (x3)\n'
        if unroll_level > 1:
          import math
          template += f'slli x4, OFFSET, {int(math.log2(unroll_level))}\n'
          template += f'add x1, x1, x4\n'
          template += f'add x2, x2, x4\n'
          template += f'add x3, x3, x4\n'
        else:
          template += f'add x1, x1, OFFSET\n'
          template += f'add x2, x2, OFFSET\n'
          template += f'add x3, x3, OFFSET\n'

        for i in range(unroll_level):
          template += f'vle{configs.data_size * 8}.v v{3*i+1}, {32*i}(x1)\n'
          template += f'vle{configs.data_size * 8}.v v{3*i+2}, {32*i}(x2)\n'
        for i in range(unroll_level):
          for j in range(configs.intensity):
            template += f'vadd.vv v{3*i+2}, v{3*i+1}, v{3*i+2}\n'
        for i in range(unroll_level):
          template += f'vse{configs.data_size * 8}.v v{3*i+2}, {32*i}(x3)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a), (self.input_b_addr, self.input_b)])

    def make_output_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a), (self.input_b_addr, self.input_b),
                              (self.output_addr, self.output)])