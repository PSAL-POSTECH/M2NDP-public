from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs
from .csr_generator import SparseGraph

class SpMV(NdpKernel):
    def __init__(self, nnz_per_row=-1):
        super().__init__()

        self.input_rows_addr = 0x800000000000
        self.input_cols_addr = 0x810000000000
        self.input_data_addr = 0x820000000000
        self.in_vector_addr =  0x840000000000
        self.out_vector_addr = 0x850000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = f'SpMV'
        
        input_cols = []
        input_rows = []
        input_cnts = []
        out_vector = []
        mode = ''
        if nnz_per_row < 0:
            file_name = os.path.join(os.path.dirname(__file__), '../../cuda/SpMV/gpu.out')
            with open(file_name, 'r') as f:
                lines = f.readlines()
                info = lines[0]
                self.num_rows = int(info.split()[0])
                self.num_cols = int(info.split()[1])
                self.num_nonzeros = int(info.split()[2])
                mode = ''
                for line in lines[1:]:
                    line = line.strip()
                    if 'rows' in line:
                        mode = 'row'
                        continue
                    elif 'cols' in line:
                        mode = 'col'
                        continue
                    elif 'values' in line:
                        mode = 'val'
                        continue
                    elif 'inVector' in line:
                        mode = 'in'
                        continue
                    elif 'outVector' in line:
                        mode = 'out'
                        continue
                    if mode == 'row':
                        input_rows = line.split(' ')
                        print(len(input_rows))
                    elif mode == 'col':
                        input_cols = line.split(' ')
                    elif mode == 'val':
                        input_vals = line.split(' ')
                    elif mode == 'in':
                        in_vector = line.split(' ')
                    elif mode == 'out':
                        out_vector = line.split(' ')
        else:
            self.num_rows = 7168
            self.num_cols = int(self.num_rows * nnz_per_row)
            self.num_nonzeros = int(self.num_rows * nnz_per_row)
            sparse_graph = SparseGraph(self.num_rows, self.num_cols, seed=0)
            input_vals, input_rows, input_cols = sparse_graph.convert_to_csr()
            in_vector = [1.0 for i in range(self.num_rows)]
            out_vector = sparse_graph.spmv(in_vector)
            sparse_graph.store_to_mtx_format(f'spmv_{nnz_per_row}.mtx')

        self.input_cols = pad8(np.array(input_cols, dtype=np.int32), constant_values=-1)
        self.input_rows = pad8(np.array(input_rows, dtype=np.int32), constant_values=-1)
        self.input_data = pad8(np.array(input_vals, dtype=np.float32), constant_values=-1)
        self.in_vector = pad8(np.array(in_vector, dtype=np.float32))
        self.out_vector = pad8(np.array(out_vector, dtype=np.float32))
        self.vector_size = self.num_rows
        self.node_num = self.num_rows
        self.base_addr = self.input_rows_addr
        self.bound = self.vector_size * configs.packet_size
        self.input_addrs = [self.input_rows_addr, self.input_cols_addr, self.input_data_addr, self.in_vector_addr, self.out_vector_addr]

        max_cols = 0
        per_row_col = np.array(input_rows, dtype=np.int32)[1:] - np.array(input_rows, dtype=np.int32)[:-1]
        max_cols = max(per_row_col)
        print(np.percentile(per_row_col, 99.99))
        print(max_cols)
        print(np.mean(per_row_col))


    def make_kernel(self):
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'    
        template += f'KERNELBODY:\n'
        template += f'vsetvli t0, a0, e32, m1\n'
        template += f'li x9, {spad_addr}\n'
        template += f'ld x10, (x9)\n'#{input_rows_addr}
        template += f'ld x3, 8(x9)\n'#{input_cols_addr}
        template += f'ld x4, 16(x9)\n'#{input_data_addr}
        template += f'ld x5, 24(x9)\n'#{input_vector_addr}
        template += f'ld x6, 32(x9)\n'#{output_vector_addr}
        template += f'li x28, 8\n'
        template += f'li x31, -1\n'
        template += f'vmset.m v0\n'
        template += f'vid.v v3, v0\n'
        template += f'vmv.v.i v2, 0\n'
        template += f'vfcvt.f.x.v v2, v2\n'
        template += f'srli x9, x2, 3\n'
        template += f'add x10, x10, x9\n'
        template += f'lw x9, (x10)\n'
        template += f'lw x10, 4(x10)\n'
        template += f'ble x10, x9, .SKIP1\n'
        # spmv_csr_scalar_kernel
        template += f'li x8, 0\n'
        template += f'vmv.v.i v11, 0\n' # sum = 0
        template += f'vfcvt.f.x.v v11, v11\n'
        template += f'.LOOP0\n'
        template += f'vadd.vx v4, v3, x9\n' # start + [0, 1, 2, 3, 4, 5, 6, 7]
        template += f'vmslt.vx v0, v4, x10\n' # mask = index < end
        template += f'vsll.vi v4, v4, 2\n'
        template += f'vluxei32.v v5, (x3), v4, v0\n' # col[j]
        template += f'vsll.vi v5, v5, 2\n'
        template += f'vluxei32.v v5, (x5), v5, v0\n' # x[col[j]]
        template += f'vluxei32.v v6, (x4), v4, v0\n' # data[j]
        template += f'vfmacc.vv v11, v5, v6\n' # vd[j] = data[j] * x[col[j]]
        template += f'sub x7, x10, x9\n'
        template += f'addi x9, x9, 8\n'
        
        template += f'bgt x7, x28, .LOOP0\n'
        template += f'vmv.v.i v7, 0\n' # sum = 0
        template += f'vfredosum.vs v11, v11, v7\n' # sum += data[j] * x[col[j]]
        template += f'vfmv.f.s f0, v11\n' # f0 = v6[0]
        template += f'srli x9, x2, 3\n'
        template += f'add x9, x9, x6\n'
        template += f'fsw f0, (x9)\n'
        template += f'.SKIP1\n'
        template += f'li x9, 1\n'

        return template
    
    def make_input_map(self):
        return make_memory_map([
                (self.input_rows_addr, self.input_rows),
                (self.input_cols_addr, self.input_cols),
                (self.input_data_addr, self.input_data),
                (self.in_vector_addr, self.in_vector),
            ])
    
    def make_output_map(self):
        return make_memory_map([
            (self.input_rows_addr, self.input_rows),
            (self.input_cols_addr, self.input_cols),
            (self.input_data_addr, self.input_data),
            (self.in_vector_addr, self.in_vector),
            (self.out_vector_addr, self.out_vector)
            ])