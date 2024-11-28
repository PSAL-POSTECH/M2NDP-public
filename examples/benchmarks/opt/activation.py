from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8, pad16
import configs
import json

class OptRelu(NdpKernel):
  def __init__(self, config : str = 'opt-125m'):
    super().__init__()
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path))
    self.vector_size = config_file['ffn_dim']
    self.input_addr = 0x800000000000
    self.output_addr = 0x810000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'{config}-relu'
    self.base_addr = self.input_addr
    self.bound = int(self.vector_size * configs.data_size)
    self.input_addrs = [self.output_addr]
    self.input_data = np.random.rand(self.vector_size).astype(np.float32)
    self.outputs = np.maximum(self.input_data, 0)

  def make_kernel(self):
    packet_size = configs.packet_size
    data_size = configs.data_size
    spad_addr = configs.spad_addr
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli 0, 0, e32, m1, 0\n'
    template += f'li x3, {spad_addr}\n'
    template += f'ld x3, (x3)\n' # load output addr
    template += f'add x3, x3, x2\n' # add offset
    template += f'vmv.v.i v2, 0\n' # v2 = 0
    template += f'vle32.v v1, (x1)\n' # load input addr
    template += f'fmv.w.x f0, x0\n' # f0 = 0
    template += f'vmsge.vf v0, v1, f0\n' # v0 = v1 > 0
    template += f'vmv.v.v v2, v1, v0\n' # v2 = v1 if v1 > 0 else 0
    template += f'vse32.v v2, (x3)\n'
    return template

  def make_input_map(self):
    return make_memory_map([(self.input_addr, self.input_data)])

  def make_output_map(self):
    return make_memory_map([(self.input_addr, self.input_data),
                            (self.output_addr, self.outputs)])