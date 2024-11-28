from typing import Any
import numpy as np
import math
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

np.random.seed(0)
class NewBatchSoftMaxKernel2(NdpKernel):
    def __init__(self):
        super().__init__()
        self.group_number = 32  # TODO. This value is fixed...
        self.vector_size = int(os.getenv("SOFTMAX_SIZE",default=1024))
        self.max_val = 4096
        self.input_addr =       0x820000000000
        self.coefficient_addr = 0x830000000000
        self.output_addr =      0x840000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'batch_softmax_kernel2'
        self.base_addr = self.input_addr
        self.unroll_level = 4   # TODO. dynamically change, How many read vector register
        self.mask = (self.group_number * configs.data_size // configs.packet_size) - 1
        self.bound = self.vector_size * self.group_number * configs.data_size // self.unroll_level

        if configs.data_size == 4:
            data_type = np.float32
        elif configs.data_size == 2:
            data_type = np.float16
        else:
            raise Exception("Unsupported data type")
        input_data = np.random.rand(self.vector_size * self.group_number).astype(data_type)
        #input_data = np.arange(self.vector_size * self.group_number).astype(data_type)
        input_data = input_data.reshape(self.group_number, self.vector_size)
        input_max_data = np.max(input_data, axis=-1, keepdims=True)
        outputs = np.exp(input_data-input_max_data)
        self.input_data = outputs
        self.coefficient = np.sum(outputs, axis=-1, keepdims=True).astype(np.float32)
        self.output = (self.input_data * (1 /self.coefficient)).astype(data_type)
        self.input_addrs = [self.input_addr, self.output_addr, self.coefficient_addr, self.group_number * configs.data_size, int(math.log2(self.unroll_level)), self.mask]
        self.input_data = np.transpose(self.input_data)
        self.output = np.transpose(self.output)

    def make_kernel(self):
        data_size = configs.data_size
        bit_size = data_size * 8
        self.group_level = int(math.log2(self.vector_size*data_size))

        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        # Initializer
        template += f'INITIALIZER:\n'
        template += f'vsetvli t0, a0, e32, m1\n'
        template += f'li x1, {configs.spad_addr}\n'
        template += f'addi x2, x1, {configs.packet_size*2}\n'
        template += f'ld x3, 24(x1)\n'        # y stride

        if configs.data_size == 2:
            template += f'slli x3, x3, 1\n'
        template += f'li x4, 0\n'
        template += f'ld x5, 16(x1)\n'

        template += f'vmv.v.i v0, 1\n'
        template += f'vfcvt.f.x.v v0, v0\n'                                 # Should add this...

        # Reduce spad addr
        template += f'.LOOP1\n'
        template += f'vle32.v v1, (x5)\n'
        template += f'vfdiv.vv v2, v0, v1\n'
        template += f'vse32.v v2, (x2)\n'
        template += f'addi x2, x2, {configs.packet_size}\n'
        template += f'addi x4, x4, {configs.packet_size}\n'
        template += f'addi x5, x5, {configs.packet_size}\n'
        template += f'bne x3, x4, .LOOP1\n'

        # Body
        template += f'KERNELBODY:\n'
        template += f'vsetvli t0, a0, e{bit_size}, m1\n'
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x2, (x1)\n'          # input
        template += f'ld x3, 8(x1)\n'         # input
        template += f'ld x4, 24(x1)\n'        # y stride
        template += f'ld x6, 40(x1)\n'        # mask

        template += f'srli x7, OFFSET, 5\n'   # thread id
        template += f'and x8, x7, x6\n'       # thread id.x
        template += f'sub x9, x7, x8\n'       # thread id.y
        template += f'srli x9, x9, 1\n'       # TODO. mask shift is hardcoded...

        template += f'slli x10, x9, 8\n'      # TODO. packet_size * (group_size * datasize/packetsize) * unroll_level
        template += f'add x2, x2, x10\n'
        template += f'add x3, x3, x10\n'

        template += f'slli x8, x8, 5\n'
        if configs.data_size == 2:
            template += f'slli x9, x8, 1\n'       # thread id.x * 64
        else:
            template += f'addi x9, x8, 0\n'       # thread id.x * 32
        template += f'add x9, x9, x1\n'       # x9 = spad_addr + x * 64
        template += f'addi x9, x9, {configs.packet_size*2}\n'

        if configs.data_size == 2:
            template += f'vsetvli t0, a0, e32, m2\n'
            template += f'vle32.v v5, (x9)\n'     # Load 1/coeff
            template += f'vfncvt.f.f.w v6, v5\n'
        else:
            template += f'vle32.v v6, (x9)\n'     # Load 1/coeff

        template += f'add x2, x2, x8\n'
        template += f'add x3, x3, x8\n'

        if configs.data_size == 2:
            template += f'vsetvli t0, a0, e16, m1\n'
        template += f'vle{bit_size}.v v1, (x2)\n'
        template += f'add x2, x2, x4\n'
        template += f'vle{bit_size}.v v2, (x2)\n'
        template += f'add x2, x2, x4\n'
        template += f'vle{bit_size}.v v3, (x2)\n'
        template += f'add x2, x2, x4\n'
        template += f'vle{bit_size}.v v4, (x2)\n'

        template += f'vfmul.vv v1, v1, v6\n'
        template += f'vfmul.vv v2, v2, v6\n'
        template += f'vfmul.vv v3, v3, v6\n'
        template += f'vfmul.vv v4, v4, v6\n'

        template += f'vse{bit_size}.v v1, (x3)\n'
        template += f'add x3, x3, x4\n'
        template += f'vse{bit_size}.v v2, (x3)\n'
        template += f'add x3, x3, x4\n'
        template += f'vse{bit_size}.v v3, (x3)\n'
        template += f'add x3, x3, x4\n'
        template += f'vse{bit_size}.v v4, (x3)\n'


        return template

    def make_input_map(self):
      return make_memory_map([(self.input_addr, self.input_data), (self.coefficient_addr, self.coefficient)])

    def make_output_map(self):
      return make_memory_map([(self.input_addr, self.input_data), (self.coefficient_addr, self.coefficient),
                              (self.output_addr, self.output)])
