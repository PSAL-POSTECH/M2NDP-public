from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class Histogram256Kernel(NdpKernel):
    def __init__(self):
        super().__init__()
        self.vector_size = 1048576*64 
        # self.vector_size = 1024*16
        self.category_num = 256
        self.input_a_addr = 0x100000000000000
        self.output_addr = 0x102000000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'histogram'
        self.base_addr = self.input_a_addr
        self.bound = self.vector_size  # 1B data size
        self.input_addrs = [self.output_addr]
    
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
        template += f'li x2, 0\n'
        template += f'li x3, {self.category_num * data_size}\n'
        template += f'vmv.v.i v1, 0\n'
        template += f'li x5, {configs.spad_addr}\n'
        template += f'ld x5, (x5)\n' 
        template += f'.LOOP0\n'
        template += f'bge x2, x3, .SKIP0\n'
        template += f'add x4, x1, x2\n'
        template += f'vse32.v v1, (x4)\n'
        template += f'bne NDPID, x0, .SKIP1\n'
        template += f'add x4, x5, x2\n'
        template += f'vse32.v v1, (x4)\n'
        template += f'.SKIP1\n'
        template += f'addi x2, x2, 32\n'
        template += f'j .LOOP0\n'
        template += f'.SKIP0\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'vle32.v v1, (ADDR)\n'
        template += f'li x1, {configs.spad_addr + packet_size}\n'
        template += f'vmv.v.i v2, 1\n'
        template += f'vand.vi v3, v1, 255\n'
        template += f'vmul.vi v3, v3, 4\n'
        template += f'vamoaddei32.v x0, (x1), v3, v2\n'
        template += f'vsrl.vi v1, v1, 8\n'
        template += f'vand.vi v3, v1, 255\n'
        template += f'vmul.vi v3, v3, 4\n'
        template += f'vamoaddei32.v x0, (x1), v3, v2\n'
        template += f'vsrl.vi v1, v1, 8\n'
        template += f'vand.vi v3, v1, 255\n'
        template += f'vmul.vi v3, v3, 4\n'
        template += f'vamoaddei32.v x0, (x1), v3, v2\n'
        template += f'vsrl.vi v1, v1, 8\n'
        template += f'vand.vi v3, v1, 255\n'
        template += f'vmul.vi v3, v3, 4\n'
        template += f'vamoaddei32.v x0, (x1), v3, v2\n'
        template += f'FINALIZER:\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x1, {configs.spad_addr + packet_size}\n'
        template += f'li x2, {configs.spad_addr}\n'
        template += f'ld x2, (x2)\n'
        template += f'li x3, {configs.spad_addr + packet_size + self.category_num * data_size}\n'
        template += f'li x4, 0\n'
        template += f'vmset.m v0\n'
        template += f'vid.v v1, v0\n'
        template += f'vmul.vi v1, v1, 4\n'
        template += f'.LOOP1\n'
        template += f'bge x1, x3, .SKIP2\n'
        template += f'vle32.v v2, (x1)\n'
        template += f'vamoaddei32.v x0, (x2), v1, v2\n'
        template += f'addi x1, x1, 32\n'
        template += f'addi x2, x2, 32\n'
        template += f'j .LOOP1\n'
        template += f'.SKIP2\n'
        return template

    def make_input_map(self):
        self.origin_input = np.random.randint(
            size=[self.vector_size], high=self.category_num, low=0, dtype=np.uint8)
        def concat_uint8_to_int32(x):
            return np.reshape(x, [-1, 4]).view(np.uint32).reshape(-1).astype(np.int32)
        self.input_a = concat_uint8_to_int32(self.origin_input)
        input_memory_map = make_memory_map(
            [(self.input_a_addr, self.input_a)])
        return input_memory_map

    def make_output_map(self):
        output = np.histogram(
            self.origin_input, bins=self.category_num, range=(0, self.category_num))
        self.output = np.array(output[0], dtype=np.int32)
        output_memory_map = make_memory_map(
        [(self.input_a_addr, self.input_a), (self.output_addr, self.output)])
        return output_memory_map
