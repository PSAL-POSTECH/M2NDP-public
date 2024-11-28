from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class PageRankKernel1(NdpKernel):
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
    self.kernel_name = f'pagerank_kernel1'
    self.base_addr = self.input_rows_addr
    self.bound = self.vector_size * configs.data_size
    self.input_addrs = [self.input_cols_addr, self.input_cnts_addr, self.input_data_addr, self.node_num]
    input_cols = []
    input_rows = []
    input_cnts = []
    outputs = []
    mode = ''
    file_name = os.path.join(os.path.dirname(__file__), '../../cuda/pagerank/kernel1_input.txt')
    with open(file_name, 'r') as f:
        for line in f:
            if 'col_array' in line:
                mode = 'col'
                continue
            elif 'row_array' in line:
                mode = 'row'
                continue
            elif 'col_cnt' in line:
                mode = 'cnt'
                continue
            if mode == 'col':
                input_cols.append(int(line))
            elif mode == 'row':
                input_rows.append(int(line))
            elif mode == 'cnt':
                input_cnts.append(int(line))
    file_name = os.path.join(os.path.dirname(__file__), '../../cuda/pagerank/kernel1_output.txt')
    with open(file_name, 'r') as f:
        for line in f:
            outputs.append(float(line))
    self.input_cols = pad8(np.array(input_cols, dtype=np.int32), constant_values=-1)
    self.input_rows = pad8(np.array(input_rows, dtype=np.int32), constant_values=-1)
    self.input_cnts = pad8(np.array(input_cnts, dtype=np.int32), constant_values=-1)
    self.outputs = pad8(np.array(outputs, dtype=np.float32))
    self.smem_size = configs.data_size


  def make_kernel(self):
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'INITIALIZER:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x7, {configs.spad_addr}\n'
    template += f'li x8, 1\n'
    template += f'fmv.w.x f1, x8\n' # f1 = 1.0
    template += f'fsw f1, 32(x7)\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x27, 8\n'
    template += f'li x28, {configs.spad_addr}\n'
    template += f'ld x3, (x28)\n'       #{input_cols_addr}
    template += f'ld x4, 8(x28)\n'      #{input_cnts_addr}
    template += f'ld x5, 16(x28)\n'     #{input_data_addr}
    template += f'ld x29, 24(x28)\n'    #{node_num}\n'
    template += f'vmv.v.x v10, x3\n'
    template += f'flw f1, 32(x28)\n'  # f1 = 1.0
    template += f'li x31, -1\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v2, v0\n' # [0, 1, 2, 3, 4, 5, 6, 7]
    # inicsr kernel
    template += f'srli x7, x2, 2\n'
    template += f'addi x7, x7, 8\n'
    template += f'bgt x7, x29, .SKIP1\n' # id > num_nodes : you don't need to load 64B
    template += f'vsetvli t0, a0, e32, m2\n'
    template += f'.SKIP1\n'
    template += f'vle32.v v1, (x1)\n'
    template += f'vmv.x.s x7, v1\n' # start = row[id]
    template += f'vslide1down.vx v1, v1, x31\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'j .SKIP2\n'
    template += f'.LOOP1\n'
    template += f'vmv.x.s x7, v1\n' # start = row[id]
    template += f'vslide1down.vx v1, v1, x31\n'
    template += f'.SKIP2\n'
    template += f'vmv.x.s x8, v1\n' # end = row[id + 1]
    template += f'bltz x8, .SKIP0\n'
    template += f'.LOOP0\n'
    template += f'vadd.vx v3, v2, x7\n' # start + [0, 1, 2, 3, 4, 5, 6, 7]
    template += f'vmslt.vx v0, v3, x8\n' # mask = index < end
    template += f'vsll.vi v3, v3, 2\n' # index * 4 = offset
    template += f'vluxei32.v v4, (x3), v3, v0\n'
    template += f'vsll.vi v4, v4, 2\n' # index * 4 = offset
    template += f'vluxei32.v v5, (x4), v4, v0\n' # cnts_vector
    template += f'vfcvt.f.x.v v5, v5\n'
    template += f'vfrdiv.vf v5, v5, f1\n' # 1.0 / (float)col_cnt
    template += f'vsuxei32.v v5, (x5), v3, v0\n'
    template += f'sub x6, x8, x7\n'
    template += f'addi x7, x7, 8\n'
    template += f'bgt x6, x27, .LOOP0\n'
    template += f'j .LOOP1\n'
    template += f'.SKIP0\n'
    return template

  def make_input_map(self):
   return make_memory_map([
        (self.input_rows_addr, self.input_rows),
        (self.input_cols_addr, self.input_cols),
        (self.input_cnts_addr, self.input_cnts)
    ])

  def make_output_map(self):
    return make_memory_map([
        (self.input_rows_addr, self.input_rows),
        (self.input_cols_addr, self.input_cols),
        (self.input_cnts_addr, self.input_cnts),
        (self.input_data_addr, self.outputs)])

