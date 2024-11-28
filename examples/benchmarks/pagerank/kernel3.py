from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class PageRankKernel3(NdpKernel):
  def __init__(self, iter=0):
    super().__init__()
    self.vector_size = 299072
    self.node_num = 299067
    self.input_rows_addr = 0x800000000000
    self.input_cols_addr = 0x810000000000
    self.input_data_addr = 0x820000000000
    self.input_cnts_addr = 0x830000000000
    self.page_rank_addr1 = 0x840000000000
    self.page_rank_addr2 = 0x850000000000
    self.sync = 0
    self.kernel_id = 0
    self.kernel_name = f'pagerank_kernel3_iter{iter}'
    self.base_addr = self.page_rank_addr2
    self.bound = self.vector_size * configs.data_size
    self.input_addrs = [self.page_rank_addr1, self.node_num]
    self.iter = iter
    input_pr1 = []
    input_pr2 = []
    output_pr1 = []
    output_pr2 = []
    mode = ''
    file_name = os.path.join(os.path.dirname(__file__), f'../../cuda/pagerank/kernel3_input{iter}.txt')
    with open(file_name, 'r') as f:
        for line in f:
            if 'pagerank1' in line:
                mode = 'pr1'
                continue
            elif 'pagerank2' in line:
                mode = 'pr2'
                continue
            if mode == 'pr1':
                input_pr1.append(float(line))
            elif mode == 'pr2':
                input_pr2.append(float(line))

    file_name = os.path.join(os.path.dirname(__file__), f'../../cuda/pagerank/kernel3_output{iter}.txt')
    with open(file_name, 'r') as f:
        for line in f:
            if 'pagerank1' in line:
                mode = 'pr1'
                continue
            elif 'pagerank2' in line:
                mode = 'pr2'
                continue
            if mode == 'pr1':
                output_pr1.append(float(line))
            elif mode == 'pr2':
                output_pr2.append(float(line))
    self.input_pr1 = pad8(np.array(input_pr1, dtype=np.float32), constant_values=-1)
    self.input_pr2 = pad8(np.array(input_pr2, dtype=np.float32), constant_values=-1)
    self.output_pr1 = pad8(np.array(output_pr1, dtype=np.float32), constant_values=-1)
    self.output_pr2 = pad8(np.array(output_pr2, dtype=np.float32), constant_values=-1)
    self.smem_size = configs.data_size * 10

  def make_kernel(self):
    spad_addr = configs.spad_addr
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x4, 100\n'
    template += f'fmv.w.x f0, x4\n' # f0 = 100.0
    template += f'li x4, 15\n'
    template += f'fmv.w.x f1, x4\n' # f1 = 15.0
    template += f'fdiv f1, f1, f0\n' # f1 = 0.15
    template += f'li x4, 85\n'
    template += f'fmv.w.x f2, x4\n' # f2 = 85.0
    template += f'fdiv f2, f2, f0\n' # f1 = 0.85
    template += f'li x3, {spad_addr}\n'
    template += f'fsw f2, 32(x3)\n'
    template += f'ld x5, 8(x3)\n' # num_nodes
    template += f'fmv.w.x f0, x5\n' # (float)num_nodes
    template += f'fdiv f0, f1, f0\n' # 0.15f / (float)num_nodes
    template += f'fsw f0, 36(x3)\n'
    template += f'li x4, -1\n'
    template += f'fmv.w.x f1, x4\n' # -1.0
    template += f'fsw f1, 40(x3)\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    # pagerank2 kernel
    template += f'li x3, {spad_addr}\n'
    template += f'ld x5, 8(x3)\n' # num_nodes
    template += f'flw f1, 40(x3)\n'
    template += f'flw f0, 36(x3)\n'
    template += f'flw f2, 32(x3)\n'

    template += f'vle32.v v2, (x1)\n'
    template += f'vfmul.vf v2, v2, f2\n' # 0.85f * page_rank2[tid];
    template += f'vfadd.vf v2, v2, f0\n' # 0.15f / (float)num_nodes + 0.85f * page_rank2[tid];

    template += f'vmv.v.i v1, 0\n'
    template += f'vfcvt.f.x.v v1, v1\n' # page_rank2[tid] = 0.0f

    template += f'srli x4, x2, 2\n'
    template += f'addi x4, x4, 8\n'
    template += f'blt x4, x5, .SKIP0\n'

    template += f'sub x4, x5, x4\n'
    template += f'addi x4, x4, 8\n'
    template += f'csrw vstart, x4\n'
    template += f'vfmv.v.f v2, f1\n'
    template += f'csrw vstart, x4\n'
    template += f'vfmv.v.f v1, f1\n'

    template += f'.SKIP0\n'
    template += f'ld x4, (x3)\n'
    template += f'add x4, x2, x4\n'
    template += f'vse32.v v2, (x4)\n'
    template += f'vse32.v v1, (x1)\n'

    return template

  def make_input_map(self):
   return make_memory_map([
        (self.page_rank_addr1, self.input_pr1),
        (self.page_rank_addr2, self.input_pr2)])

  def make_output_map(self):
    return make_memory_map([
        (self.page_rank_addr1, self.output_pr1),
        (self.page_rank_addr2, self.output_pr2)])

