from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class SoftMaxKernel2(NdpKernel):
    def __init__(self):
        super().__init__()
        self.vector_size = int(os.getenv("SOFTMAX_SIZE",default=1024))
        self.max_val = 4096
        self.input_addr =  0x800000000000
        self.output_addr = 0x820000000000
        self.sync = 0
        self.kernel_id = 2
        self.kernel_name = 'softmax_kernel2'
        self.base_addr = self.input_addr
        self.bound = self.vector_size * configs.data_size
        self.smem_size = configs.packet_size * 4
        
        if configs.data_size == 4:
            data_type = np.float32
        elif configs.data_size == 2:
            data_type = np.float16
        else:
            raise Exception("Unsupported data type")

        self.input_data = np.random.rand(self.vector_size).astype(data_type)
        self.input_coefficient = np.random.rand(1).astype(np.float32)
        self.output = (self.input_data / self.input_coefficient).astype(data_type)
        self.input_addrs = [self.output_addr, 0, 0, 0, 0, 0, 0,
                            self.input_coefficient[0]]
    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'vsetvli t0, a0, e{data_size*8}, m1\n'
        template += f'li x1, {configs.spad_addr + 64}\n'
        template += f'li x2, {1}\n'
        template += f'flw f1, (x1)\n'
        template += f'fmv.w.x f2, x2\n'
        template += f'fdiv f1, f2, f1\n'            # 1 / sum(E(tensor))
        template += f'li x1, {configs.spad_addr + 3*configs.packet_size}\n'
        template += f'fsw f1, (x1)\n'

        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x1, {spad_addr }\n'
        template += f'li x3, {configs.spad_addr + 3*configs.packet_size}\n'
        template += f'ld x1, (x1)\n'
        template += f'flw f1, (x3)\n'
        template += f'add x1, x1, OFFSET\n'
        template += f'vle32.v v1, (ADDR)\n'
        template += f'vfmul.vf v3, v1, f1\n'
        template += f'vse32.v v3, (x1)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_addr, self.input_data)])

    def make_output_map(self):
      return make_memory_map([(self.input_addr, self.input_data),
                              (self.output_addr, self.output)])