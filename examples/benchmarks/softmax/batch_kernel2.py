from typing import Any
import numpy as np
import math
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class BatchSoftMaxKernel2(NdpKernel):
    def __init__(self):
        super().__init__()
        self.group_number = 32
        self.vector_size = int(os.getenv("SOFTMAX_SIZE",default=1024))
        self.max_val = 4096
        self.input_addr =       0x820000000000
        self.coefficient_addr = 0x830000000000
        self.output_addr =      0x840000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'batch_softmax_kernel2'
        self.base_addr = self.input_addr
        self.bound = self.vector_size * self.group_number * configs.data_size

        if configs.data_size == 4:
            data_type = np.float32
        elif configs.data_size == 2:
            data_type = np.float16
        else:
            raise Exception("Unsupported data type")
        input_data = np.arange(self.vector_size * self.group_number).astype(data_type)#np.random.rand(self.vector_size).astype(np.float32)
        input_data = input_data.reshape(self.group_number, self.vector_size)
        input_max_data = np.max(input_data, axis=-1, keepdims=True)
        outputs = np.exp(input_data-input_max_data)
        self.input_data = outputs
        #self.input_data = np.random.rand(self.group_number * self.vector_size).astype(data_type)
        #self.input_data = self.input_data.reshape(self.group_number, self.vector_size)
        self.coefficient = np.sum(outputs, axis=-1, keepdims=True).astype(np.float32)
        self.output = (self.input_data / self.coefficient).astype(data_type)
        self.input_addrs = [self.output_addr, self.coefficient_addr, self.group_number]

    def make_kernel(self):
        data_size = configs.data_size
        self.group_level = int(math.log2(self.vector_size*data_size))

        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'vsetvli t0, a0, e{32}, m1\n'
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x2, 8(x1)\n'   # coefficient_addr
        template += f'ld x3, 16(x1)\n'   # group_number
        template += f'slli x3, x3, {2}\n'
        template += f'addi x3, x3, {31}\n'
        template += f'srli x3, x3, {5}\n'
        template += f'li x4, 0\n'
        template += f'li x5, {configs.spad_addr + 2*configs.packet_size}\n'

        template += f'vmv.v.i v2, 1\n'
        template += f'vfcvt.f.x.v v2, v2\n'
 
        template += f'.LOOP1\n'
        template += f'vle32.v v1, (x2)\n'
        template += f'vfdiv.vv v1, v2, v1\n'
        template += f'vse32.v v1, (x5)\n'

        template += f'addi x4, x4, 1\n'
        template += f'addi x2, x2, 32\n'
        template += f'addi x5, x5, 32\n'
        template += f'bne x3, x4, .LOOP1\n'

        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e{configs.data_size * 8}, m1, 0\n'
        template += f'vle{configs.data_size * 8}.v v1, (ADDR)\n'

        template += f'li x1, {spad_addr}\n'
        template += f'li x3, {configs.spad_addr + 2*configs.packet_size}\n'
        template += f'ld x1, (x1)\n'

        template += f'srli x4, OFFSET, {self.group_level}\n'                # Is it cheating? Maybe..?
        template += f'slli x5, x4, {2}\n'
        template += f'add x3, x3, x5\n'

        template += f'flw f1, (x3)\n'
        template += f'add x1, x1, OFFSET\n'
        template += f'vfmul.vf v3, v1, f1\n'
        template += f'vse{configs.data_size * 8}.v v3, (x1)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_addr, self.input_data), (self.coefficient_addr, self.coefficient)])

    def make_output_map(self):
      return make_memory_map([(self.input_addr, self.input_data), (self.coefficient_addr, self.coefficient),
                              (self.output_addr, self.output)])
