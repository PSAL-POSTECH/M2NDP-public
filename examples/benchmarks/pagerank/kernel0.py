from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class PageRankKernel0(NdpKernel):
  def __init__(self):
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
    self.kernel_name = 'pagerank_kernel0'
    self.base_addr = self.page_rank_addr1
    self.bound = self.vector_size * configs.data_size
    self.input_addrs = [self.page_rank_addr2, self.node_num]
    self.smem_size = 32 * 2 # v1, v2 (pagerank1, pagerank2)


  def make_kernel(self):
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x3, {configs.spad_addr}\n'
    template += f'ld x5, 8(x3)\n'
    template += f'fmv.w.x f0, x5\n' # (float)num_node
    template += f'vmv.v.i v2, 1\n' # page_rank2
    template += f'vfcvt.f.x.v v2, v2\n'
    template += f'vfdiv.vf v2, v2, f0\n' # page_rank1 = 1 / (float)num_nodes
    template += f'li x5, {configs.spad_addr + configs.packet_size}\n'
    template += f'vse32.v v2, (x5)\n'
    template += f'vmv.v.i v1, 0\n' # page_rank2 = 0
    template += f'vfcvt.f.x.v v1, v1\n'
    template += f'addi x5, x5, 32\n'
    template += f'vse32.v v1, (x5)\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x5, {configs.spad_addr}\n'
    template += f'ld x4, 8(x5)\n'
    template += f'ld x6, (x5)\n' # {page_rank_addr2}
    template += f'vle32.v v2, {configs.packet_size}(x5)\n'
    template += f'addi x5, x5, 32\n'
    template += f'vle32.v v1, {configs.packet_size}(x5)\n'
    template += f'add x6, x6, x2\n'
    template += f'srli x3, x2, 2\n'
    template += f'addi x3, x3, 8\n'
    template += f'blt x3, x4, .SKIP0\n'
    template += f'sub x3, x4, x3\n'
    template += f'addi x3, x3, 8\n'
    template += f'vfmv.f.s f1, v1\n'
    template += f'csrw vstart, x3\n'
    template += f'vfmv.v.f v2, f1\n'
    template += f'.SKIP0\n'
    template += f'vse32.v v2, (x1)\n' # store page_rank1
    template += f'vse32.v v1, (x6)\n' # store page_rank2
    return template

  def make_input_map(self):
    return make_memory_map([])

  def make_output_map(self):
    file_name = os.path.join(os.path.dirname(__file__), '../../cuda/pagerank/kernel0_output.txt')
    output1 = []
    output2 = []
    with open(file_name, 'r') as f:
        mode = ''
        for line in f:
            if 'pagerank1' in line:
                mode = 'pagerank1'
                continue
            elif 'pagerank2' in line:
                mode = 'pagerank2'
                continue
            if mode == 'pagerank1':
                output1.append(float(line))
            elif mode == 'pagerank2':
                output2.append(float(line))
    output1 = pad8(np.array(output1, dtype=np.float32))
    output2 = pad8(np.array(output2, dtype=np.float32))
    return make_memory_map([(self.page_rank_addr1, output1), (self.page_rank_addr2, output2)])

