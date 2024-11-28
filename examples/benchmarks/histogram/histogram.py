from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class HistogramKernel(NdpKernel):
    def __init__(self, bins: int):
        super().__init__()
        self.vector_size = 1048576*64
        # self.vector_size = 1024*64*2
        self.category_num = int(bins)
        print("Category num: ", self.category_num)
        self.input_a_addr = 0x800000000000
        self.output_addr =  0x810000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'histogram'
        self.base_addr = self.input_a_addr
        self.unroll = 16
        input_data = []
        with open(os.path.join(os.path.dirname(__file__), f'../../data//h_data.txt')) as f:
            lines = f.readlines()
            for line in lines:
                input_data.append(int(line) & (self.category_num -1))
        self.input_a = np.array(input_data, dtype=np.int32)
        self.vector_size = len(self.input_a)
        self.bound = self.vector_size * configs.data_size // self.unroll
        self.input_addrs = [self.output_addr, self.input_a_addr]
        self.smem_size = configs.data_size * self.category_num # 2 * v1

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += 'INITIALIZER:\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x1, {configs.spad_addr + packet_size}\n'
        template += f'li x6, 0\n'
        template += f'li x3, {self.category_num * data_size}\n'
        template += f'vmv.v.i v1, 0\n'
        template += f'li x5, {configs.spad_addr}\n'
        template += f'ld x5, (x5)\n'
        template += f'.LOOP0\n'
        template += f'bge x6, x3, .SKIP0\n'
        template += f'add x4, x1, x6\n'
        template += f'vse32.v v1, (x4)\n'
        template += f'bne NDPID, x0, .SKIP1\n'
        template += f'add x4, x5, x6\n'
        template += f'vse32.v v1, (x4)\n'
        template += f'.SKIP1\n'
        template += f'addi x6, x6, 32\n'
        template += f'j .LOOP0\n'
        template += f'.SKIP0\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x6, {configs.spad_addr}\n'
        template += f'ld x6, 8(x6)\n'
        template += f'muli x3, x2, {self.unroll}\n'
        template += f'add x6, x6, x3\n'
        for unroll in range(self.unroll):
            template += f'vle32.v v{unroll}, (x6)\n'
            template += f'addi x6, x6, 32\n'
        template += f'li x1, {configs.spad_addr + packet_size}\n'
        template += f'vmv.v.i v30, 1\n'
        for unroll in range(self.unroll):
            template += f'vmul.vi v{unroll}, v{unroll}, 4\n'
        for unroll in range(self.unroll):
            template += f'vamoaddei32.v x0, (x1), v{unroll}, v30\n'
        template += f'FINALIZER:\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x1, {configs.spad_addr + packet_size}\n'
        template += f'li x6, {configs.spad_addr}\n'
        template += f'ld x6, (x6)\n'
        template += f'li x3, {configs.spad_addr + packet_size + self.category_num * data_size}\n'
        template += f'li x4, 0\n'
        template += f'vmset.m v0\n'
        template += f'vid.v v1, v0\n'
        template += f'vmul.vi v1, v1, 4\n'
        template += f'.LOOP1\n'
        template += f'bge x1, x3, .SKIP2\n'
        template += f'vle32.v v2, (x1)\n'
        template += f'vle32.v v3, 32(x1)\n'
        template += f'vle32.v v4, 64(x1)\n'
        template += f'vle32.v v5, 96(x1)\n'
        template += f'vamoaddei32.v x0, (x6), v1, v2\n'
        template += f'addi x6, x6, 32\n'
        template += f'vamoaddei32.v x0, (x6), v1, v3\n'
        template += f'addi x6, x6, 32\n'
        template += f'vamoaddei32.v x0, (x6), v1, v4\n'
        template += f'addi x6, x6, 32\n'
        template += f'vamoaddei32.v x0, (x6), v1, v5\n'
        template += f'addi x6, x6, 32\n'
        template += f'addi x1, x1, 128\n'
        template += f'j .LOOP1\n'
        template += f'.SKIP2\n'
        return template

    def make_input_map(self):
        # self.input_a = np.random.randint(
        #     size=[self.vector_size], high=self.category_num, low=0, dtype=np.int32)
        input_memory_map = make_memory_map(
            [(self.input_a_addr, self.input_a)])
        return input_memory_map

    def make_output_map(self):
        output = np.histogram(
            self.input_a, bins=self.category_num, range=(0, self.category_num))
        self.output = np.array(output[0], dtype=np.int32)
        output_memory_map = make_memory_map(
        [(self.input_a_addr, self.input_a), (self.output_addr, self.output)])
        return output_memory_map
