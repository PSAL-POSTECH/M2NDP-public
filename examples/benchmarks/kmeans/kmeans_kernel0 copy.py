from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class KmeansKernel0(NdpKernel):
    def __init__(self):
        super().__init__()
        self.dimension = 32
        self.num_clusters = 8
        self.max_val = 10
        self.input_a_addr = 0x800000000000 #0x100000000000000
        self.input_b_addr = 0x810000000000 #0x101000000000000
        self.output_addr = 0x820000000000 #0x102000000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'Kmeans_kernel0'
        self.base_addr = self.input_a_addr
        self.bound = self.dimension * self.num_clusters * configs.data_size
        self.query = pad8(np.random.randint(size=[self.dimension], high=self.max_val, low=0, dtype=np.int32))
        self.matrix = np.random.randint(size=[self.num_clusters, self.dimension], high=self.max_val, low=0, dtype=np.int32)
        #euclidean distance with each point
        self.output = pad8(np.sum((self.query - self.matrix)**2, axis=1).astype(np.int32))
        self.matrix_flat = self.matrix.flatten()
        self.input_addrs = [self.input_b_addr, self.output_addr, self.dimension*configs.data_size, self.num_clusters]
    
    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'
        template += 'INITIALIZER:\n'
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x2, (x1)\n' # x2: query vector address at global memory
        template += f'li x3, {configs.spad_addr + packet_size*2}\n' # x3: start address of query vector at spad
        template += f'ld x4, 16(x1)\n' # x4: dimension size in bytes: 32 * 4 = 128
        template += f'ld x9, 8(x1)\n' # x9: output_addr at global memory
        
        #copy query vector from global memory to spad
        template += f'li x5, 0\n' # x5: offset
        template += f'.LOOP0\n'
        template += f'add x6, x3, x5\n' # x6: spad_addr + offset
        template += f'add x7, x2, x5\n' # x7: query vector address at global memory + offset
        template += f'vle32.v v1, (x7)\n' # v1: query vector
        template += f'vse32.v v1, (x6)\n' # v1: query vector
        template += f'addi x5, x5, {configs.packet_size}\n' # x5: offset += packet_size
        template += f'blt x5, x4, .LOOP0\n' #if offset < dimension_size, jump to .LOOP0       
        
        #initialize output vector with 0
        #As max cluster size is 5 at this moment, we can use vse32.v to move output vector
        template += f'vmv.v.i v2, 0\n' # v2: initialize output with 0
        template += f'li x5, {configs.spad_addr + packet_size*2 + self.dimension * configs.data_size}\n' # x5: start address of output vector
        template += f'vse32.v v2, (x5)\n' # v2: output vector to spad mem
        template += f'vse32.v v2, (x9)\n' # v2: output vector to global mem
        
        #debug
        # template += f'vle32.v v3, (x5)\n'
        # template += f'TEST.vv v3, v3\n'
        
        
        
        template += f'KERNELBODY:\n'
        
        template += f'vsetvli t0, a0, e32, m1\n'
        #load centroid
        template += f'vle32.v v1, (ADDR)\n' #v1: centroid
        
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x2, 16(x1)\n' # x2: dimension size in bytes: 32 * 4 = 128
        
        template += f'li x4, {configs.spad_addr + packet_size*2}\n' # x4: start address of query vector
        template += f'rem x5, OFFSET, x2\n' # x5: OFFSET % dimension_size
        template += f'add x5, x4, x5\n' # x5: spad_addr + packet_size*2 + OFFSET % dimension_size
        template += f'vle32.v v2, (x5)\n' # v2: query vector
        template += f'vmv.v.i v3, 0\n' # v3: initialize output with 0
        #calculate euclidean distance
        template += f'vsub.vv v4, v1, v2\n' # v4: centroid - query
        template += f'vmul.vv v5, v4, v4\n' # v5: (centroid - query)^2
        template += f'vredsum.vs v6, v5, v3\n' # v6: sum((centroid - query)^2)
        #calculate output address at spad
        template += f'li x6, {configs.spad_addr + packet_size*2 + self.dimension * configs.data_size}\n' # x6: start address of output vector
        template += f'div x7, OFFSET, x2\n' # x7: OFFSET / dimension_size = output index
        template += f'slli x7, x7, 2\n' # x7: output index * 4byte = output offset
        template += f'add x6, x6, x7\n' # x6: output_base_addr + output offset
        #store output with amoadd
        template += f'vamoaddei32.v x0, (x6), v3, v6\n'


        template += f'FINALIZER:\n'
        
        template += f'vsetvli t0, a0, e32, m1\n'
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x2, 8(x1)\n' # x2: output_addr at global memory
        template += f'li x4, {configs.spad_addr + packet_size*2 + self.dimension * configs.data_size}\n' # x4: start address of output vector
        
        #move output vector from spad to global memory
        #As max cluster size is 5 at this moment, we can use vse32.v to move output vector
        template += f'vle32.v v1, (x4)\n' # v1: output vector
        #use vamoaddei32 to store output vector to global memory
        template += f'vmset.m v0\n'
        template += f'vid.v v2, v0\n' #[0, 1, 2, 3, 4, 5, 6, 7]
        template += f'vsll.vi v2, v2, 2\n' # 4byte
        template += f'vamoaddei32.v x0, (x2), v2, v1\n' #v2: offset, v1: output vector
      
        return template

    def make_input_map(self):
      return make_memory_map([(self.input_a_addr, self.matrix_flat), (self.input_b_addr, self.query)])

    def make_output_map(self):
      return make_memory_map([(self.input_a_addr, self.matrix_flat), (self.input_b_addr, self.query),
                              (self.output_addr, self.output)])
