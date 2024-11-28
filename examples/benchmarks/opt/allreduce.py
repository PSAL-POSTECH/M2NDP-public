from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs
import json

class ReduceScatter(NdpKernel):
    def __init__(self, size : int = 1024*1024, model_parallelism = 1):
        super().__init__()
        self.model_parallelism = model_parallelism
        interleave_addr = 0x100000000000
        self.vector_size = size // model_parallelism
        self.input_addr = 0x010000000000
        self.output_addr =  0x020000000000 
        self.input_addrs_list = []
        for i in range(model_parallelism):
          input_addr = self.input_addr + i * interleave_addr
          next_input_addr = self.input_addr + (i + 1) % model_parallelism * interleave_addr
          output_addr = self.output_addr + i * interleave_addr
          self.input_addrs_list.append([next_input_addr, output_addr])
          self.base_addrs.append(input_addr)

        
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'ReduceScatter'
        self.base_addr = self.input_addr
        self.bound = self.vector_size * configs.data_size
        self.input = np.random.rand(model_parallelism, self.vector_size).astype(np.float32)
        self.output = np.zeros((model_parallelism, self.vector_size)).astype(np.float32)
        for i in range(model_parallelism):
          self.output[i] = self.input[i] + self.input[(i + 1) % model_parallelism]
        

    
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
        template += f'li x1, {spad_addr }\n'
        template += f'li x2, {spad_addr + 8}\n'
        template += f'ld x1, (x1)\n'
        template += f'ld x2, (x2)\n'
        template += f'add x1, x1, OFFSET\n'
        template += f'add x2, x2, OFFSET\n'
        template += f'vle32.v v1, (ADDR)\n'
        template += f'vle32.v v2, (x1)\n'
        template += f'vfadd.vv v3, v1, v2\n'
        template += f'vse32.v v3, (x2)\n'

        return template

    def make_input_map(self):
      memory_maps = []
      for i in range(self.model_parallelism):
        memory_map = []
        memory_map.append((self.base_addrs[i], self.input[i]))
        memory_map.append((self.input_addrs_list[i][0], self.input[(i + 1) % self.model_parallelism]))
        memory_map.append((self.input_addrs_list[i][1], np.zeros(self.vector_size).astype(np.float32)))
        memory_maps.append(make_memory_map(memory_map))
      return memory_maps

    def make_output_map(self):
      memory_maps = []
      for i in range(self.model_parallelism):
        memory_map = []
        memory_map.append((self.base_addrs[i], self.input[i]))
        memory_map.append((self.input_addrs_list[i][0], self.input[(i + 1) % self.model_parallelism]))
        memory_map.append((self.input_addrs_list[i][1], self.output[i]))
        memory_maps.append(make_memory_map(memory_map))
      return memory_maps

class AllGather(NdpKernel):
    def __init__(self, size : int = 1024*1024, model_parallelism = 1):
        super().__init__()
        self.model_parallelism = model_parallelism
        interleave_addr = 0x100000000000
        self.vector_size = size // model_parallelism
        self.input_addr = 0x010000000000
        self.output_addr =  0x020000000000 
        self.input_addrs_list = []
        for i in range(model_parallelism):
          input_addr = self.input_addr + i * interleave_addr
          output_addr = self.output_addr + (i + 1) % model_parallelism * interleave_addr
          self.input_addrs_list.append([output_addr])
          self.base_addrs.append(input_addr)

        
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'AllGather'
        self.base_addr = self.input_addr
        self.bound = self.vector_size * configs.data_size
        self.input = np.random.rand(model_parallelism, self.vector_size).astype(np.float32)
        self.output = np.zeros((model_parallelism, self.vector_size)).astype(np.float32)
        for i in range(model_parallelism):
          self.output[i] = self.input[i]
        
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
        template += f'li x1, {spad_addr}\n'
        template += f'ld x1, (x1)\n'
        template += f'vle32.v v1, (ADDR)\n'
        template += f'add x1, x1, OFFSET\n'
        template += f'vse32.v v1, (x1)\n'

        return template

    def make_input_map(self):
      memory_maps = []
      for i in range(self.model_parallelism):
        memory_map = []
        memory_map.append((self.base_addrs[i], self.input[i]))
        memory_map.append((self.input_addrs_list[i][0], np.zeros(self.vector_size).astype(np.float32)))
        memory_maps.append(make_memory_map(memory_map))
      return memory_maps

    def make_output_map(self):
      memory_maps = []
      for i in range(self.model_parallelism):
        memory_map = []
        memory_map.append((self.base_addrs[i], self.input[i]))
        memory_map.append((self.input_addrs_list[i][0], self.output[i]))
        memory_maps.append(make_memory_map(memory_map))
      return memory_maps

class OptReduceScatter(ReduceScatter):
  def __init__(self, config : str = 'opt-125m', model_parallelism = 1):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path, 'r'))
    self.vector_size = config_file['hidden_size']
    super().__init__(self.vector_size, model_parallelism=model_parallelism)


class OptAllGather(AllGather):
  def __init__(self, config : str = 'opt-125m', model_parallelism = 1):
    config_file_path = os.path.join(os.path.dirname(__file__), f'configs/{config}.json')
    config_file = json.load(open(config_file_path, 'r'))
    self.vector_size = config_file['hidden_size']
    super().__init__(64, model_parallelism=model_parallelism)


