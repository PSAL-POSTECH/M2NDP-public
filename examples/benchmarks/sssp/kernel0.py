from typing import Any
import numpy as np
import math
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class SsspKernel0(NdpKernel):
    def __init__(self):
        super().__init__()
        self.INT32_SIZE = 4
        self.INT_MAX = 99999999
        self.info_addr    = 0x800000000000
        self.row_addr     = 0x810000000000
        self.col_addr     = 0x820000000000
        self.data_addr    = 0x830000000000
        self.vector1_addr = 0x840000000000
        self.vector2_addr = 0x850000000000
        self.spad_addr    = 0x1000000000000000
        self.base_addr = self.vector1_addr
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'sssp_kernel0'
        self.input_file = os.path.join(os.path.dirname(__file__), f'../../data/USA-road-d.NY.gr')

        # input memorymap
        input_coo = self.read_coo()
        self.total_nodes = input_coo[0]
        self.total_edges = input_coo[1]
        self.graph_info = np.array([0, self.total_nodes, self.total_edges, 0, 0, 0, 0, 0], dtype=np.int32)
        self.graph_info = np.expand_dims(self.graph_info, axis=0)

        self.input_file = os.path.join(os.path.dirname(__file__), f'../../cuda/sssp/raw/init_output.txt')
        input = self.read_output_file(self.input_file)
        self.row = pad8(np.array(input[1], dtype=np.int32))
        self.col = pad8(np.array(input[2], dtype=np.int32))
        self.data = pad8(np.array(input[3], dtype=np.int32))
        self.vector1 = pad8(np.array(input[4], dtype=np.int32))
        self.vector2 = pad8(np.array(input[5], dtype=np.int32))

        # output memorymap
        self.output_file = os.path.join(os.path.dirname(__file__), f'../../cuda/sssp/raw/loop_0_0.txt')
        output = self.read_output_file(self.output_file)
        self.o_graph_info = (np.array(output[0], dtype=np.int32))
        self.o_row = pad8(np.array(output[1], dtype=np.int32))
        self.o_col = pad8(np.array(output[2], dtype=np.int32))
        self.o_data = pad8(np.array(output[3], dtype=np.int32))
        self.o_vector1 = pad8(np.array(output[4], dtype=np.int32))
        self.o_vector2 = pad8(np.array(output[5], dtype=np.int32))

        self.vector_size = math.ceil(self.total_nodes / 8) * 8
        self.bound = self.vector_size * configs.data_size
        elems_per_packet = configs.packet_size / configs.data_size
        n_node = int(math.ceil(self.total_nodes / elems_per_packet) * elems_per_packet)
        self.input_addrs = [self.info_addr, self.row_addr, self.col_addr, self.data_addr,
                            self.vector1_addr, self.vector2_addr, self.total_nodes, n_node]

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        elems_per_packet = packet_size / data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'
        template+= f'vsetvli 0, 0, e32, m1, 0\n'

        template+= f'li x1, {configs.spad_addr}\n'

        template+= f'ld x13, 8(x1)\n' # Row addr
        template+= f'ld x12, 32(x1)\n' # vector1 addr
        template+= f'ld x3, 40(x1)\n' # vector2 addr
        template+= f'ld x14, 56(x1)\n' # total_nodes

        template+= f'add x4, x13, x2\n' # row addr + offset
        template+= f'vle32.v v1, (x4)\n'  # row [offset : offset + packet_size]
        template+= f'add x13, x12, x2\n'
        template+= f'vle32.v v2, (x13)\n' # load vector1
        template += f'vmv.v.i v3, 99999999\n'
        template+= f'addi x13, x2, {packet_size}\n'
        template+= f'srli x13, x13, 2\n'
        template+= f'sub x13, x13, x14\n'  # x9 = next row node index - n_node
        template+= f'bgez x13, .SKIP0\n'  # do not load when x9 >= 0
        # if next pkt exist
        template+= f'lw x13, {packet_size}(x4)\n' # change to lw
        # if not exist
        template+= f'.SKIP0\n'
        template+= f'ld x4, 16(x1)\n' # col addr
        template+= f'ld x5, 24(x1)\n' # data addr
        template+= f'ld x6, 48(x1)\n' # INT_MAX
        template+= f'li x7, {elems_per_packet}\n' # 8
        template+= f'vmv.v.v v4, v1\n' # v4 = row
        template+= f'vslide1down.vx v5, v4, x13\n' # shift left
        template+= f'vsub.vv v4, v5, v1\n' # n edges
        template+= f'li x13, 0\n'  # x13 = node count
        template+= f'.LOOP0\n'  # node loop (8)
        template+= f'srli x8, x2, 2\n'  # x8 = offset / 4
        template+= f'add x8, x8, x13\n' # x8 = count + offset / 4
        template+= f'ble x6, x8, .SKIP1\n'  # skip to SKIP if total_nodes <= curr idx
        template+= f'csrw vstart, x13\n'  # x13 = count
        template+= f'vmv.x.s x8, v2\n' # vector1[count]
        template+= f'csrw vstart, x13\n'  # x13 = count
        template+= f'vmv.x.s  x9, v1\n'  # x9 = row[count], starting edge idx
        template+= f'csrw vstart, x13\n' # x13 = count
        template+= f'vmv.x.s x10, v4\n' # x10 = row_end - row_start
        template+= f'add x11, x9, x10\n' # x11 = row_end

        # index vector
        template+= f'.LOOP1\n'
        template+= f'vid.v v5\n' # v5 = [0, 1, 2, 3, 4, 5, 6, 7]
        template+= f'vadd.vx v5, v5, x9\n' # v5 = v5 + row[start]
        template+= f'vmslt.vx v0, v5, x11\n'
        template+= f'vsll.vi v5, v5, 2\n'
        template+= f'vluxei32.v v6, (x4), v5, v0\n' # col index load
        template+= f'vluxei32.v v7, (x5), v5, v0\n' # data(weight) index load
        template+= f'vsll.vi v6, v6, 2\n'
        template+= f'vluxei32.v v8, (x12), v6, v0\n' # vector1 index load
        template+= f'vadd.vv v6, v8, v7\n' # edge + node weight
        template+= f'vmv.v.i v7, 0\n'
        template+= f'vmv.s.x v7, x8\n'
        template+= f'vredmin.vs v5, v6, v7, v0\n'
        template+= f'vmv.x.s x8, v5\n' # x8 = min
        template+= f'addi x9 , x9, {elems_per_packet}\n'
        template+= f'bgt x11, x9, .LOOP1\n'  # jump to LOOP2 if edge is left
        template+= f'csrw vstart, x13\n'
        template+= f'vmv.s.x v3, x8\n'
        template+= f'addi x13, x13, 1\n'
        template+= f'bgt x7, x13, .LOOP0\n'  # node loop (8)
        template+= f'.SKIP1\n' # All node done

        # save v2
        template+= f'add x13, x3, x2\n'
        template+= f'vse32.v v3, (x13)\n'
        return template

    def make_input_map(self):
      return make_memory_map([(self.info_addr, self.graph_info),
                              (self.row_addr, self.row),
                              (self.col_addr, self.col),
                              (self.data_addr, self.data),
                              (self.vector1_addr, self.vector1),
                              (self.vector2_addr, self.vector2)])

    def make_output_map(self):
      return make_memory_map([(self.info_addr, self.o_graph_info),
                              (self.row_addr, self.o_row),
                              (self.col_addr, self.o_col),
                              (self.data_addr, self.o_data),
                              (self.vector1_addr, self.o_vector1),
                              (self.vector2_addr, self.o_vector2)])

    def read_coo(self):
      t_nodes = 0
      t_edges = 0
      head_list = []
      tail_list = []
      weight_list = []

      count = 0
      f = open(self.input_file, 'r')
      lines = f.readlines()
      graph_lines = lines[7:]

      for line in lines:
        s_list = line.split()
        if s_list[0] == 'c':
          continue
        elif s_list[0] == 'p':
          t_nodes = int(s_list[2])
          t_edges = int(s_list[3])
        elif s_list[0] == 'a':
          head_list.append(int(s_list[1]) - 1)
          tail_list.append(int(s_list[2]) - 1)
          weight_list.append(int(s_list[3]))

      f.close()

      stable_sort = sorted(enumerate(head_list), key=lambda x: x[1])
      sort_index = [index for index, _ in stable_sort]

      head_sort = [head_list[i] for i in sort_index]
      tail_sort = [tail_list[i] for i in sort_index]
      weight_sort = [weight_list[i] for i in sort_index]

      return t_nodes, t_edges, head_sort, tail_sort, weight_sort

    def read_output_file(self, file_path):
      info = []
      row = []
      col = []
      data = []
      vector1 = []
      vector2 = []

      file = open(file_path, 'r')
      lines = file.readlines()

      step = 0
      for line in lines:
        line = line.replace('\n', "")
        if line == "graph_info":
          step = 0
          continue
        elif line == "row":
          step = 1
          continue
        elif line == "col":
          step = 2
          continue
        elif line == "data":
          step = 3
          continue
        elif line == "vector1":
          step = 4
          continue
        elif line == "vector2":
          step = 5
          continue

        s_list = line.split()
        tmp = [int(i) for i in s_list]
        if step == 0:
          info.extend(tmp)
        elif step == 1:
          row.extend(tmp)
        elif step == 2:
          col.extend(tmp)
        elif step == 3:
          data.extend(tmp)
        elif step == 4:
          vector1.extend(tmp)
        elif step == 5:
          vector2.extend(tmp)

      return info, row, col, data, vector1, vector2