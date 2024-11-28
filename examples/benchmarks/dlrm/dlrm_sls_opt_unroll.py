from typing import Any
import numpy as np
import os
from utils.utils import *
import configs
import math

class SparseLengthSumUnrolledKernel(MultipleKernel):
    def __init__(self, batch_size=4):
        super().__init__()
        self.sync = 0
        self.kernel_id = 0
        self.batch_size = int(batch_size)
        print(self.batch_size)
        if self.batch_size == 256 or self.batch_size == 32:
          self.unroll = 4
        elif self.batch_size == 4:
          self.unroll = 8
        self.num_kernel = int(256 // self.batch_size)
        self.repeat = self.num_kernel
        if self.kernel_id == 0:
          self.batch_size *= self.num_kernel
        self.emb_max_index = 1000000
        self.emb_dim = 256
        self.emb_table_id = '0'
        self.n_lookup = 80
        self.kernel_name = f'sparse_length_sum{self.kernel_id}'

        self.input_index_data = pad8(self.make_input_data())
        self.input_offset_data = np.array([], dtype=np.int32)
        if self.kernel_id == 0:
          start = 0
          for i in range(self.num_kernel):
            end = self.batch_size//self.num_kernel + 1
            self.input_offset_data = np.concatenate((self.input_offset_data, pad8(np.array([j * self.n_lookup for j in range(start, end)], dtype=np.int32))))
            # start = end - 1
        else:
          self.input_offset_data = pad8(np.array([i * self.n_lookup for i in range(self.batch_size+1)], dtype=np.int32))

        self.output_data = pad8(self.make_output_data())

        self.input_index_offset = len(self.input_index_data) * configs.data_size
        self.input_offset_offset = len(self.input_offset_data) * configs.data_size
        self.output_offset = len(self.output_data) * configs.data_size

        self.input_index_addr  = 0x800000000000 + self.input_index_offset * self.kernel_id
        self.input_offset_addr = 0x810000000000 + self.input_offset_offset * self.kernel_id
        self.output_addr = 0x820000000000 + self.output_offset * self.kernel_id
        self.base_addr = self.output_addr
        self.bound = int(self.emb_dim * batch_size * 4)
        self.embedding_table_addr = 0x830000000000

        self.input_addrs = [self.embedding_table_addr, self.input_index_addr, self.input_offset_addr]
        offset_size = len(pad8(np.array([j * self.n_lookup for j in range(0, int(batch_size) + 1)]))) * 4
        index_offset = int(self.n_lookup * batch_size * 8)
        self.input_addrs_offsets = [0, index_offset, offset_size]


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
        template += f'li x17, {spad_addr}\n'
        template += f'ld x18, (x17)\n' # x18 = embedding table addr
        template += f'ld x3, 8(x17)\n' # x3 = index_addr
        template += f'ld x4, 16(x17)\n' # x18 = offset_addr
        template += f'srli x5, x2, {int(math.log2(self.emb_dim))}\n' # x5 = batch number * 4
        template += f'andi x6, x2, {self.emb_dim * 4 - 1}\n' # x6 embeding table offset
        template += f'add x18, x18, x6\n' # x18 = embedding table base
        template += f'add x7, x4, x5\n'
        template += f'lw x8, 4(x7)\n' # x7 = offset_bound
        template += f'lw x7, (x7)\n' # x6 = offset base
        template += f'.LOOP1\n'
        template += f'muli x9, x7, {8}\n'
        template += f'add x9, x9, x3\n'
        template += f'vsetvli 0, 0, e64, m1, 0\n'
        template += f'vle32.v v3, (x9)\n' # v9 = [index 0, index2, index3, index4]
        if self.unroll == 8:
          template += f'vle32.v v4, {packet_size}(x9)\n' # v9 = [index 0, index2, index3, index4]
        template += f'vmul.vi v3, v3, {self.emb_dim * 4}\n' # v2 = -1 if v2 >= offset_bound else v2
        if self.unroll == 8:
          template += f'vmul.vi v4, v4, {self.emb_dim * 4}\n' # v2 = -1 if v2 >= offset_bound else v2
        template += f'vadd.vx v3, v3, x18\n' # v2 = embedding table addr + offset
        if self.unroll == 8:
          template += f'vadd.vx v4, v4, x18\n' # v2 = embedding table addr + offset
        template += f'vmv.x.s x9, v3\n' # x9 = -1 if v2 >= offset_bound else x9
        template += f'vslide1down.vx v3, v3, x0\n' # v2 = [offset 1, offset 2, offset 3, 0]
        template += f'vmv.x.s x10, v3\n'
        template += f'vslide1down.vx v3, v3, x0\n' # v2 = [offset 2, offset 3, 0, 0]
        template += f'vmv.x.s x11, v3\n'
        template += f'vslide1down.vx v3, v3, x0\n' # v2 = [offset 3, 0, 0, 0]
        template += f'vmv.x.s x12, v3\n'
        if self.unroll == 8:
          template += f'vmv.x.s x13, v4\n'
          template += f'vslide1down.vx v4, v4, x0\n' # v2 = [offset 3, 0, 0, 0]
          template += f'vmv.x.s x14, v4\n'
          template += f'vslide1down.vx v4, v4, x0\n' # v2 = [offset 3, 0, 0, 0]
          template += f'vmv.x.s x15, v4\n'
          template += f'vslide1down.vx v4, v4, x0\n' # v2 = [offset 3, 0, 0, 0]
          template += f'vmv.x.s x16, v4\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'vle32.v v10, (x9)\n' # v10 = embedding table value
        template += f'vle32.v v11, (x10)\n' # v10 = embedding table value
        template += f'vle32.v v12, (x11)\n' # v10 = embedding table value
        template += f'vle32.v v13, (x12)\n' # v10 = embedding table value
        if self.unroll == 8:
          template += f'vle32.v v14, (x13)\n' # v10 = embedding table value
          template += f'vle32.v v15, (x14)\n' # v10 = embedding table value
          template += f'vle32.v v16, (x15)\n' # v10 = embedding table value
          template += f'vle32.v v17, (x16)\n' # v10 = embedding table value
        template += f'vfadd.vv v1, v1, v10\n' # v1 = v1 + v10
        template += f'vfadd.vv v1, v1, v11\n' # v1 = v1 + v10
        template += f'vfadd.vv v1, v1, v12\n' # v1 = v1 + v10
        template += f'vfadd.vv v1, v1, v13\n' # v1 = v1 + v10
        if self.unroll == 8:
          template += f'vfadd.vv v1, v1, v14\n' # v1 = v1 + v10
          template += f'vfadd.vv v1, v1, v15\n' # v1 = v1 + v10
          template += f'vfadd.vv v1, v1, v16\n' # v1 = v1 + v10
          template += f'vfadd.vv v1, v1, v17\n' # v1 = v1 + v10
        if self.unroll == 8:
          template += f'addi x7, x7, 8\n'
        else:
          template += f'addi x7, x7, 4\n'
        template += f'vsetvli 0, 0, e64, m1, 0\n'
        template += f'blt x7, x8, .LOOP1\n' # x7 < x8
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'vse32.v v1, (x1)\n' # STORE

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_index_addr, self.input_index_data), (self.input_offset_addr, self.input_offset_data)])

    def make_output_map(self):
      return make_memory_map([(self.input_index_addr, self.input_index_data), (self.input_offset_addr, self.input_offset_data)
                             , (self.output_addr, self.output_data)])

    def make_input_data(self):
      data = []
      start = self.kernel_id * self.batch_size
      with open(os.path.join(os.path.dirname(__file__), f'../../data/kaggle_emb/emb_{self.emb_table_id}.txt'), 'r') as f:
        i = 0
        for line in f:
          # if i >= start and i < self.batch_size + start:
          accesses = line.split(' ')
          data += accesses[:self.n_lookup]
          # i += 1
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