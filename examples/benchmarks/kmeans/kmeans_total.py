from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class KmeansTotal(NdpKernel):
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
        self.kernel_name = 'Kmeans_total'
        self.base_addr = self.input_a_addr
        self.bound = 32
        self.query = pad8(np.random.randint(size=[self.dimension], high=self.max_val, low=0, dtype=np.int32))
        self.matrix = np.random.randint(size=[self.num_clusters, self.dimension], high=self.max_val, low=0, dtype=np.int32)
        #euclidean distance with each point
        self.output = np.sum((self.query - self.matrix)**2, axis=1).astype(np.int32)
        print(self.output)
        self.output = pad8(np.array([np.argmin(self.output).astype(np.int32)]))
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
        
        template += f'KERNELBODY:\n'
        
        template += f'vsetvli 0, 0, e32, m1, 0\n'
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x2, (x1)\n' # x2: query vector address at global memory
        # template += f'li x3, {configs.spad_addr + packet_size*2}\n' # x3: start address of query vector at spad
        template += f'ld x9, 8(x1)\n' # x9: output_addr at global memory
        template += f'ld x10, 16(x1)\n' # x10: dimension size in bytes: 32 * 4 = 128
        template += f'ld x11, 24(x1)\n' # x11: num_clusters
        
        #copy query vector from global memory to spad
        # template += f'li x5, 0\n' # x5: offset
        # template += f'.LOOP0\n'
        # template += f'add x6, x3, x5\n' # x6: spad_addr + offset
        # template += f'add x7, x2, x5\n' # x7: query vector address at global memory + offset
        # template += f'vle32.v v1, (x7)\n' # v1: query vector
        # template += f'vse32.v v1, (x6)\n' # v1: query vector
        # template += f'addi x5, x5, {configs.packet_size}\n' # x5: offset += packet_size
        # template += f'blt x5, x10, .LOOP0\n' #if offset < dimension_size, jump to .LOOP0       
        
        #initialize output vector with 0
        #As max cluster size is 5 at this moment, we can use vse32.v to move output vector
        template += f'vmv.v.i v10, 0\n' # v10: initialize output with 0
        template += f'li x12, {configs.spad_addr + packet_size*2 + self.dimension * configs.data_size}\n' # x12: start address of output vector
        template += f'vse32.v v10, (x12)\n' # v10: output vector to spad mem    
        
        #loop over all centroids
        template += f'mul x8, x10, x11\n' # x8: dimension_size * num_clusters in bytes
        template += f'li x5, 0\n' # x5: offset
        template += f'.LOOP1\n'
        #load centroid
        template += f'add x6, ADDR, x5\n' # x6: base_addr + offset
        template += f'vle32.v v1, (x6)\n' #v1: centroid
        template += f'rem x7, x5, x10\n' # x7: OFFSET % dimension_size = query index
        template += f'add x7, x2, x7\n' # x7: query_base_addr + query_index
        template += f'vle32.v v2, (x7)\n' # v2: query vector
        #calculate euclidean distance
        template += f'vsub.vv v3, v1, v2\n' # v3: centroid - query
        template += f'vmul.vv v4, v3, v3\n' # v4: (centroid - query)^2
        template += f'vredsum.vs v5, v4, v10\n' # v5: sum((centroid - query)^2)
        #calculate output address at spad
        template += f'div x7, x5, x10\n' # x7: OFFSET / dimension_size = output index
        template += f'slli x7, x7, 2\n' # x7: output index * 4byte = output offset
        template += f'add x7, x12, x7\n' # x7: output_base_addr + output offset
        #store output with amoadd
        template += f'vamoaddei32.v x0, (x7), v10, v5\n'
        #check if offset < dimension_size * num_clusters
        template += f'addi x5, x5, {configs.packet_size}\n' # x5: offset += packet_size
        template += f'blt x5, x8, .LOOP1\n' #if offset < dimension_size * num_clusters, jump to .LOOP1
        
        #load output vector from spad and find min distance index
        template += f'vle32.v v1, (x12)\n' # v1: output vector
                
        template += f'vmset.m v0\n'
        template += f'vmv.v.i v5, 0\n' # v5: initialize output with 0
        
        template += f'vredmin.vs v2, v1, v1, v0\n' #v2: min distance 
        template += f'vmv.x.s x7, v2\n' # x7: min distance  
        #find the index of min distance
        template += f'vmseq.vx v3, v1, x7\n'
        template += f'vfirst.m x4, v3, v3\n' # x4: min distance index
        #store min distance index to global memory
        template += f'vmv.s.x v5, x4\n'
        template += f'vse32.v v5, (x9)\n'

        return template

    def make_input_map(self):
      return make_memory_map([(self.input_a_addr, self.matrix_flat), (self.input_b_addr, self.query)])

    def make_output_map(self):
      return make_memory_map([(self.input_a_addr, self.matrix_flat), (self.input_b_addr, self.query),
                              (self.output_addr, self.output)])
