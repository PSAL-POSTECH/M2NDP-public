from typing import Any
import numpy as np
import math
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class SsspKernel1(NdpKernel):
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
        self.kernel_name = 'sssp_kernel1'
        self.input_file = os.path.join(os.path.dirname(__file__), f'../../data/USA-road-d.NY.gr')

        # input memorymap
        input_coo = self.read_coo()
        self.total_nodes = input_coo[0]
        self.total_edges = input_coo[1]
        self.graph_info = np.array([0, self.total_nodes, self.total_edges, 0, 0, 0, 0, 0], dtype=np.int32)
        self.graph_info = np.expand_dims(self.graph_info, axis=0)

        self.input_file = os.path.join(os.path.dirname(__file__), f'../../cuda/sssp/raw/loop_0_0.txt')
        input = self.read_output_file(self.input_file)
        self.row = pad8(np.array(input[1], dtype=np.int32))
        self.col = pad8(np.array(input[2], dtype=np.int32))
        self.data = pad8(np.array(input[3], dtype=np.int32))
        self.vector1 = pad8(np.array(input[4], dtype=np.int32))
        self.vector2 = pad8(np.array(input[5], dtype=np.int32))

        # output memorymap
        self.output_file = os.path.join(os.path.dirname(__file__), f'../../cuda/sssp/raw/loop_0_1.txt')
        output = self.read_output_file(self.output_file)
        self.o_graph_info = (np.array(output[0], dtype=np.int32))
        self.o_row = pad8(np.array(output[1], dtype=np.int32))
        self.o_col = pad8(np.array(output[2], dtype=np.int32))
        self.o_data = pad8(np.array(output[3], dtype=np.int32))
        self.o_vector1 = pad8(np.array(output[4], dtype=np.int32))
        self.o_vector2 = pad8(np.array(output[5], dtype=np.int32))

        self.vector_size = math.ceil(self.total_nodes / 8) * 8
        self.bound = self.vector_size * configs.data_size
        self.input_addrs = [self.info_addr, self.row_addr, self.col_addr, self.data_addr, self.vector1_addr, self.vector2_addr]

    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        elems_per_packet = packet_size / data_size
        n_node = int(math.ceil(self.total_nodes / elems_per_packet) * elems_per_packet)
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += f'KERNELBODY:\n'

        template+= f'li x1, {configs.spad_addr}\n'

        template+= f'ld x11, (x1)\n'
        template+= f'ld x3, 8(x1)\n'
        template+= f'ld x4, 16(x1)\n'
        template+= f'ld x5, 24(x1)\n'
        template+= f'ld x6, 32(x1)\n'
        template+= f'ld x7, 40(x1)\n'

        template+= f'vsetvli 0, 0, e32, m1, 0\n'

        template+= f'li x8, 0\n'  # stop = 0, loop = 1

        template+= f'add x9, x7, x2\n'
        template+= f'vle32.v v1, (x9)\n'  # load vector2
        template+= f'add x9, x6, x2\n'
        template+= f'vle32.v v0, (x9)\n'  # load vector1
        template+= f'vsub.vv v2, v1, v0\n'  # v2 = v1 - v0

        template+= f'vmv.v.i v3, 0\n'
        template+= f'vredor.vs v4, v2, v3\n'  # vector reduce or
        template+= f'vmv.x.s x10, v4\n' # x10 = v4[0]

        template+= f'beqz x10, .SKIP0\n'  # Skip if x10 == 0 (Same)
        template+= f'li x8, 1\n'  # vector1 != vector2

        template+= f'vse32.v v1, (x9)\n' # store vector2 to vector1 addr

        template+= f'vle32.v v5, (x11)\n'  # load graph info
        template+= f'vmv.s.x v5, x8\n'  # STOP = 1
        template+= f'vse32.v v5, (x11)\n'

        template+= f'.SKIP0\n'  # vector1 == vector2

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