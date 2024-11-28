from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class ThreeColANDKernel(NdpKernel):
    def __init__(self):
        super().__init__()
        self.packet_size = 64
        self.int64_size = 8
        self.input_size = int(6001664 / 8 / 8) #6001664 / 8 #6001664 bits / 8 bits/byte
        self.input_a_addr = 0x800000000000 #0x100000000000000
        self.input_b_addr = 0x810000000000 #0x101000000000000
        self.input_c_addr = 0x820000000000 #0x102000000000000
        self.output_addr = 0x830000000000 #0x103000000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'three_col_AND'
        self.base_addr = self.input_a_addr
        self.bound = self.input_size * self.int64_size

        self.input_a = np.random.randint(low=1, high=10, size=(self.input_size, 1), dtype=np.int64)
        self.input_b = np.random.randint(low=1, high=10, size=(self.input_size, 1), dtype=np.int64)
        self.input_c = np.random.randint(low=1, high=10, size=(self.input_size, 1), dtype=np.int64)

        #bitwise AND
        self.output = self.input_a & self.input_b & self.input_c

        self.input_addrs = [self.input_b_addr, self.input_c_addr, self.output_addr]

    def make_kernel(self):
        packet_size = self.packet_size
        data_size = self.int64_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli t0, a0, e64, m1\n'
        template += f'vle64.v v1, (x1)\n'
        template += f'li x4, {spad_addr }\n'
        template += f'ld x4, (x4)\n' #x4 = input_b_addr
        template += f'add x4, x4, x2\n'
        template += f'vle64.v v2, (x4)\n'
        template += f'li x5, {spad_addr + 8}\n'
        template += f'ld x5, (x5)\n' #x5 = input_c_addr
        template += f'add x5, x5, x2\n'
        template += f'vle64.v v3, (x5)\n'
        template += f'li x3, {spad_addr + 16}\n'
        template += f'ld x3, (x3)\n' #x3 = output_addr
        template += f'add x3, x3, x2\n'
        template += f'vand.vv v4, v1, v2\n'
        template += f'vand.vv v4, v4, v3\n'
        template += f'vse64.v v4, (x3)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a), (self.input_b_addr, self.input_b)
                              ,(self.input_c_addr, self.input_c)], packet_size=self.packet_size)

    def make_output_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a), (self.input_b_addr, self.input_b),
                              (self.input_c_addr, self.input_c), (self.output_addr, self.output)],
                             packet_size=self.packet_size)
