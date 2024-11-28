from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class GtEqLtINT64Kernel(NdpKernel):
    def __init__(self):
        super().__init__()
        self.packet_size = 64
        self.int64_size = 8
        self.input_size = 6001664
        self.predicate_value_min = np.int64(3)
        self.predicate_value_max = np.int64(5)
        self.input_a_addr = 0x800000000000 #0x100000000000000
        self.output_addr = 0x820000000000 #0x102000000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'gteq_lt_int64'
        self.base_addr = self.input_a_addr
        self.bound = self.input_size * self.int64_size
        
        self.input_a = np.random.randint(low=1, high=10, size=(self.input_size, 1), dtype=np.int64)
        
        self.output = np.where((self.input_a >= self.predicate_value_min) & (self.input_a < self.predicate_value_max), 1, 0)
        self.output = self.output.reshape(-1, 8)
        self.output = np.flip(self.output, axis=1)
        self.output = np.packbits(self.output, axis=-1)
        self.output = self.output.reshape(-1)
        self.output = self.output.astype(np.uint8)
        
        self.input_addrs = [self.output_addr, self.predicate_value_min, self.predicate_value_max]
    
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
        template += f'vle64.v v1, x1\n'
        template += f'li x7, {spad_addr }\n'
        template += f'vle64.v v4, (x7)\n'
        template += f'vmv.x.s x3, v4\n' #x3 = output_addr
        template += f'csrwi vstart, 1\n' 
        template += f'vmv.x.s x4, v4\n' #x4 = predicate_value_min
        template += f'csrwi vstart, 2\n'
        template += f'vmv.x.s x5, v4\n' #x5 = predicate_value_max
        
        # template += f'li x3, {spad_addr }\n'
        # template += f'li x4, {spad_addr + 8}\n'
        # template += f'li x5, {spad_addr + 16}\n'
        # template += f'ld x3, (x3)\n'
        # template += f'ld x4, (x4)\n'
        # template += f'ld x5, (x5)\n'
        
        template += f'vmsge.vx v2, v1, x4\n'
        template += f'vmslt.vx v3, v1, x5\n'
        template += f'vmand.mm v2, v2, v3\n'
        template += f'srli x5, x2, 6\n'
        template += f'add x6, x3, x5\n'
        template += f'vsetvli t0, a0, e8, m1\n'
        template += f'vmv.x.s x3, v2\n'
        template += f'sb x3, (x6)\n'

        return template


    def make_input_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a)], packet_size=self.packet_size)

    def make_output_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a),
                              (self.output_addr, self.output)], packet_size=self.packet_size)
