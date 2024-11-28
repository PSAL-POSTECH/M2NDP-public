from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs
import math

class SparseLengthSumOptimizedKernel(NdpKernel):
    def __init__(self, batch_size=1):
        super().__init__()
        self.input_index_addr  = 0x800000000000
        self.input_offset_addr = 0x810000000000
        self.output_addr = 0x820000000000
        self.embedding_table_addr = 0x830000000000
        self.batch_size = batch_size
        self.emb_max_index = 1000000
        self.emb_dim = 16
        self.emb_table_id = '0'
        self.n_lookup = 20
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'sparse_length_sum'
        self.base_addr = self.output_addr
        self.bound = self.emb_dim * self.batch_size * 4
        
        self.input_index_data = pad8(self.make_input_data())
        self.input_offset_data = pad8(np.array([i * self.n_lookup for i in range(self.batch_size+1)], dtype=np.int32))
        self.output_data = pad8(self.make_output_data())
        self.input_addrs = [self.embedding_table_addr, self.input_index_addr, self.input_offset_addr]
    
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
        template += f'vmv.v.i v1, 0\n' # v1 = 0
        template += f'vfcvt.f.x.v v1, v1\n' # v1 = 0.0
        template += f'li x1, {spad_addr}\n'
        template += f'ld x2, (x1)\n' # x2 = embedding table addr
        template += f'ld x3, 8(x1)\n' # x3 = index_addr
        template += f'ld x4, 16(x1)\n' # x2 = offset_addr
        template += f'srli x5, OFFSET, {math.log2(self.emb_dim)}\n' # x5 = batch number * 4
        template += f'andi x6, OFFSET, {self.emb_dim * 4 - 1}\n' # x6 embeding table offset
        template += f'add x2, x2, x6\n' # x2 = embedding table base 
        template += f'add x7, x4, x5\n'
        template += f'lw x8, 4(x7)\n' # x7 = offset_bound
        template += f'lw x7, (x7)\n' # x6 = offset base
        template += f'li x10, -1\n'
        template += f'.LOOP1\n'
        template += f'muli x9, x7, {8}\n'
        template += f'add x9, x9, x3\n'
        template += f'vsetvli 0, 0, e64, m1, 0\n'
        template += f'vle32.v v3, (x9)\n' # v9 = [index 0, index2, index3, index4]
        template += f'vmset.m v0\n'
        template += f'vid.v v2, v0\n'
        template += f'vadd.vx v2, v2, x7\n' # v2 = [offset 0, offset 1, offset 2, offset 3]
        template += f'vmsge.vx v0, v2, x8\n' # v0 = v2 >= offset_bound
        template += f'vmv.v.i v3, -1, v0\n' # v2 = -1 if v2 >= offset_bound else v2
        template += f'vmv.x.s x9, v3\n' # x9 = -1 if v2 >= offset_bound else x9
        template += '.LOOP2'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'muli x9, x9, {self.emb_dim * 4}\n' # x9 = index * 4 * emb_dim
        template += f'add x9, x9, x2\n' # x9 = embedding table addr + index * 4 * emb_dim
        template += f'vle32.v v10, (x9)\n' # v10 = embedding table value
        template += f'vfadd.vv v1, v1, v10\n' # v1 = v1 + v10
        template += f'addi x7, x7, 1\n'
        template += f'vsetvli 0, 0, e64, m1, 0\n'
        template += f'vslide1down.vx v3, v3, x10\n' # v2 = [offset 1, offset 2, offset 3, -1]
        template += f'vmv.x.s x9, v3\n'
        template += f'bgez x9, .LOOP2\n' # x9 >= 0
        template += f'blt x7, x8, .LOOP1\n' # x7 < x8
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'vse32.v v1, (ADDR)\n' # STORE

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_index_addr, self.input_index_data), (self.input_offset_addr, self.input_offset_data)])

    def make_output_map(self):
      return make_memory_map([(self.input_index_addr, self.input_index_data), (self.input_offset_addr, self.input_offset_data)
                             , (self.output_addr, self.output_data)])

    def make_input_data(self):
      data = []
      with open(os.path.join(os.path.dirname(__file__), f'../../data/kaggle_emb/emb_{self.emb_table_id}.txt'), 'r') as f:
        for line in f:
          accesses = line.split(' ')
          data += accesses[:self.n_lookup]
      return np.array(data, dtype=np.longlong)
    
    def make_output_data(self):
      output = []
      for batch in range(self.batch_size):
        for dim in range(self.emb_dim * 4 // configs.packet_size):
          sum = 0.0
          for lookup in range(self.n_lookup):
            index = self.input_index_data[batch * self.n_lookup + lookup]
            addr = index * self.emb_dim * 4 + dim * configs.packet_size
            sum += addr & 0xfffff
          output = output + [sum] + [ 0 for _ in range(configs.packet_size // 4- 1)]
      return np.array(output, dtype=np.float32)
          
