from typing import Any
import numpy as np
import os
from torch import tensor
from torch.nn.functional import gelu
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

# For debug
def output_gen(x):
    ret = 0.7978845608028654 * (x + 0.044715 * x**3)
    ret = np.exp(-1*np.abs(2*ret))
    ret = (ret - 1) / (1 + ret)
    ret = np.copysign(ret, x) + 1
    ret = ret * x / 2
    return ret 

class GeLUKernel(NdpKernel):
    def __init__(self):
        super().__init__()
        self.vector_size = 1024 * 1024
        self.input_a_addr =     0x100000000000000
        self.output_addr =      0x102000000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'gelu'
        self.base_addr = self.input_a_addr
        self.bound = self.vector_size * configs.data_size
        if configs.data_size == 4:
          data_type = np.float32
        elif configs.data_size == 2:
          data_type = np.float16
        else:
          raise Exception("Unsupported data type")
        self.input_a = np.random.randn(self.vector_size).astype(data_type)
        self.input_const = np.array([0.7978845608028654, 0.044715, 0.5], dtype=np.float32)
        self.output = gelu(tensor(self.input_a), approximate='tanh').numpy()
        self.input_addrs = [self.output_addr, 0 ,0 ,0 ,0 ,0, 0,
                            self.input_const[0], self.input_const[1], self.input_const[2]]
    
    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'
        template += f'vsetvli 0, 0, e{configs.data_size*8}, m1, 0\n'
        template += f'li x1, {spad_addr}\n'
        template += f'li x2, {spad_addr + 64}\n'
        # Load constant
        template += f'ld x1, (x1)\n'
        template += f'flw f1, (x2)\n'
        template += f'flw f2, 4(x2)\n'
        template += f'flw f3, 8(x2)\n'

        template += f'add x1, x1, OFFSET\n'
        template += f'vle{configs.data_size*8}.v v1, (ADDR)\n'

        # Calculation body
        template += f'vfmul.vv v2, v1, v1\n'    # v1 = x
        template += f'vfmul.vf v2, v2, f2\n'    # f1 = 0.797...
        template += f'vfadd.vi v2, v2, 1\n'     # f2 = 0.044715
        template += f'vfmul.vv v2, v2, v1\n' 
        template += f'vfmul.vf v2, v2, f1\n'    # v2 = 0.797 * (x + 0.044715 * x^3)

        template += f'vfmul.vi v3, v2, 2\n'
        template += f'vfsgnjx.vv v3, v3, v3\n'
        template += f'vfsgnjn.vv v4, v3, v3\n'  # v3 = -|v2|
        template += f'vfexp.v v4, v3\n'         # v4 = e^(-|v2|)

        template += f'vfsub.vi v5, v4, 1\n'     # v5 = e^(-|v2|) - 1
        template += f'vfadd.vi v6, v4, 1\n'     # v6 = e^(-|v2|) + 1
        template += f'vfdiv.vv v6, v5, v6\n' 
        template += f'vfsgnj.vv v6, v6, v2\n'
        template += f'vfadd.vi v6, v6, 1\n'     # v6 = 1 + tanh(~)
        template += f'vfmul.vv v6, v6, v1\n'
        template += f'vfmul.vf v6, v6, f3\n'
        # Store
        template += f'vse{configs.data_size*8}.v v6, (x1)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a)])

    def make_output_map(self):
      return make_memory_map([(self.input_a_addr, self.input_a), (self.output_addr, self.output)])