from typing import Any
import numpy as np
import math
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class SsspInitKernel(NdpKernel):
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
        self.kernel_name = 'sssp_init'
        self.input_file = os.path.join(os.path.dirname(__file__), f'../../data/USA-road-d.NY.gr')

        # input memorymap
        input_coo = self.read_coo()
        self.source_node = 0
        self.total_nodes = input_coo[0]
        self.total_edges = input_coo[1]
        self.graph_info = np.array([0, self.total_nodes, self.total_edges, 0, 0, 0, 0, 0], dtype=np.int32)
        self.graph_info = np.expand_dims(self.graph_info, axis=0)

        row_array = []
        prev = -1
        for idx in range(self.total_edges):
          curr = input_coo[2][idx]
          if curr != prev:
            row_array.append(idx)
            prev = curr
        row_array.append(self.total_edges)

        self.row = pad8(np.array(row_array, dtype=np.int32))
        self.col = pad8(np.array(input_coo[3], dtype=np.int32))
        self.data = pad8(np.array(input_coo[4], dtype=np.int32))
        self.vector1 = pad8(np.zeros((1, math.ceil(self.total_nodes / 8) * 8), dtype=np.int32))
        self.vector2 = pad8(np.zeros((1, math.ceil(self.total_nodes / 8) * 8), dtype=np.int32))

        # output memorymap
        self.output_file = os.path.join(os.path.dirname(__file__), f'../../cuda/sssp/raw/init_output.txt')
        output = self.read_output_file(self.output_file)
        self.o_graph_info = (np.array(output[0], dtype=np.int32))
        self.o_row = pad8(np.array(output[1], dtype=np.int32))
        self.o_col = pad8(np.array(output[2], dtype=np.int32))
        self.o_data = pad8(np.array(output[3], dtype=np.int32))
        self.o_vector1 = pad8(np.array(output[4], dtype=np.int32))
        self.o_vector2 = pad8(np.array(output[5], dtype=np.int32))

        self.vector_size = math.ceil(self.total_nodes / 8) * 8
        self.bound = self.vector_size * configs.data_size
        self.input_addrs = [self.info_addr, self.row_addr, self.col_addr, self.data_addr, self.vector1_addr, self.vector2_addr, self.source_node]

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

        template+= f'vsetvli t0, a0, e32, m1, 0\n'
        template+= f'li x1, {configs.spad_addr}\n'
        template+= f'li x4, {self.INT_MAX}\n'
        template+= f'vmv.v.x v2, x4\n'

        # Initialize Source Node
        template+= f'csrwi vstart, 3\n'
        template+= f'ld x4, 48(x1)\n' # src node
        template+= f'ld x5, 32(x1)\n'
        template+= f'ld x6, 40(x1)\n'
        template+= f'add x5, x5, x2\n'
        template+= f'add x6, x6, x2\n'
        template+= f'vmset.m v0\n'
        template+= f'vid.v v1, v0\n' # v1 = [0, 1, 2, 3, 4, 5, 6, 7]
        template+= f'srli x3, x2, 2\n'  # x20 = starting index of packet
        template+= f'vadd.vx v1, v1, x3\n'  # v1 = [OFFSET, OFFSET+1, OFFSET+2, OFFSET+3, OFFSET+4, OFFSET+5, OFFSET+6, OFFSET+7]
        template+= f'vmseq.vx v0, v1, x4\n' # v0 = [0, 0, 0, 0, 0, 0, 0, 0] if v1 == src node
        template+= f'vmv.v.i v2, 0, v0\n'   # v1 = [0, 0, 0, 0, 0, 0, 0, 0] if v1 == src node
        template+= f'vse32.v v2, (x5)\n'
        template+= f'vse32.v v2, (x6)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.info_addr, self.graph_info),
                              (self.row_addr, self.row),
                              (self.col_addr, self.col),
                              (self.data_addr, self.data),
                              (self.vector1_addr, self.vector1),
                              (self.vector2_addr, self.vector2)])
      # return make_memory_map([(self.info_addr, self.graph_info)])

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