from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class PageRankKernel2Split(NdpKernel):
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
    self.kernel_name = f'pagerank_kernel2_iter{iter}'
    self.base_addr = self.input_rows_addr
    self.bound = self.node_num * configs.packet_size
    self.input_addrs = [self.input_rows_addr, self.input_cols_addr, self.input_data_addr, self.page_rank_addr1, self.page_rank_addr2]
    self.iter = iter
    input_cols = []
    input_rows = []
    input_data = []
    input_pr1 = []
    input_pr2 = []
    outputs = []
    mode = ''
    file_name = os.path.join(os.path.dirname(__file__), f'../../cuda/pagerank/kernel2_input{iter}.txt')
    with open(file_name, 'r') as f:
        for line in f:
            if 'col_array' in line:
                mode = 'col'
                continue
            elif 'row_array' in line:
                mode = 'row'
                continue
            elif 'data_array' in line:
                mode = 'data'
                continue
            elif 'pagerank1' in line:
                mode = 'pr1'
                continue
            elif 'pagerank2' in line:
                mode = 'pr2'
                continue

            if mode == 'col':
                input_cols.append(int(line))
            elif mode == 'row':
                input_rows.append(int(line))
            elif mode == 'data':
                input_data.append(float(line))
            elif mode == 'pr1':
                input_pr1.append(float(line))
            elif mode == 'pr2':
                input_pr2.append(float(line))

    file_name = os.path.join(os.path.dirname(__file__), f'../../cuda/pagerank/kernel2_output{iter}.txt')
    with open(file_name, 'r') as f:
        for line in f:
            outputs.append(float(line))
    self.input_cols = pad8(np.array(input_cols, dtype=np.int32), constant_values=-1)
    self.input_rows = pad8(np.array(input_rows, dtype=np.int32), constant_values=-1)
    self.input_data = pad8(np.array(input_data, dtype=np.float32), constant_values=-1)
    self.input_pr1 = pad8(np.array(input_pr1, dtype=np.float32), constant_values=-1)
    self.input_pr2 = pad8(np.array(input_pr2, dtype=np.float32), constant_values=-1)
    self.outputs = pad8(np.array(outputs, dtype=np.float32), constant_values=-1)
    self.smem_size = configs.data_size * 5

  def make_kernel(self):
    spad_addr = configs.spad_addr
    template = ''
    template += f'-kernel name = {self.kernel_name}\n'
    template += f'-kernel id = {self.kernel_id}\n'
    template += '\n'
    template += f'KERNELBODY:\n'
    template += f'vsetvli t0, a0, e32, m1\n'
    template += f'li x1, {spad_addr}\n'
    template += f'ld x9, (x1)\n'#{input_rows_addr}
    template += f'ld x3, 8(x1)\n'#{input_cols_addr}
    template += f'ld x4, 16(x1)\n'#{input_data_addr}
    template += f'ld x5, 24(x1)\n'#{input_vector_addr}
    template += f'ld x6, 32(x1)\n'#{output_vector_addr}
    template += f'li x28, 8\n'
    template += f'li x31, -1\n'
    template += f'vmset.m v0\n'
    template += f'vid.v v3, v0\n'
    template += f'vmv.v.i v2, 0\n'
    template += f'vfcvt.f.x.v v2, v2\n'
    template += f'srli x1, x2, 3\n'
    template += f'add x9, x9, x1\n'
    template += f'lw x1, (x9)\n'
    template += f'lw x9, 4(x9)\n'
    template += f'ble x9, x1, .SKIP1\n'
    # spmv_csr_scalar_kernel
    template += f'li x8, 0\n'
    template += f'vmv.v.i v11, 0\n' # sum = 0
    template += f'vfcvt.f.x.v v11, v11\n'
    template += f'.LOOP0\n'
    template += f'vadd.vx v4, v3, x1\n' # start + [0, 1, 2, 3, 4, 5, 6, 7]
    template += f'vmslt.vx v0, v4, x9\n' # mask = index < end
    template += f'vsll.vi v4, v4, 2\n'
    template += f'vluxei32.v v5, (x3), v4, v0\n' # col[j]
    template += f'vsll.vi v5, v5, 2\n'
    template += f'vluxei32.v v5, (x5), v5, v0\n' # x[col[j]]
    template += f'vluxei32.v v6, (x4), v4, v0\n' # data[j]
    template += f'vfmacc.vv v11, v5, v6\n' # vd[j] = data[j] * x[col[j]]
    template += f'sub x7, x9, x1\n'
    template += f'addi x1, x1, 8\n'

    template += f'bgt x7, x28, .LOOP0\n'
    template += f'vmv.v.i v7, 0\n' # sum = 0
    template += f'vfredosum.vs v11, v11, v7\n' # sum += data[j] * x[col[j]]
    template += f'vfmv.f.s f0, v11\n' # f0 = v6[0]
    template += f'srli x1, x2, 3\n'
    template += f'add x1, x1, x6\n'
    template += f'fsw f0, (x1)\n'
    template += f'.SKIP1\n'
    template += 'li x1, 1\n'
    return template

  def make_input_map(self):
   return make_memory_map([
        (self.input_rows_addr, self.input_rows),
        (self.input_cols_addr, self.input_cols),
        (self.input_data_addr, self.input_data),
        (self.page_rank_addr1, self.input_pr1),
        (self.page_rank_addr2, self.input_pr2)])

  def make_output_map(self):
    return make_memory_map([
        (self.input_rows_addr, self.input_rows),
        (self.input_cols_addr, self.input_cols),
        (self.input_data_addr, self.input_data),
        (self.page_rank_addr1, self.input_pr1),
        (self.page_rank_addr2, self.outputs)])
