from typing import Any
import numpy as np
import os
from utils.utils import NdpKernel, make_memory_map, pad8
import configs

class KmeansKernel1(NdpKernel):
    def __init__(self):
        super().__init__()
        self.dimension = 40
        self.num_clusters = 8
        self.max_val = 10
        self.input_a_addr = 0x800000000000 #0x100000000000000
        self.output_addr = 0x820000000000 #0x102000000000000
        self.sync = 0
        self.kernel_id = 0
        self.kernel_name = 'Kmeans_kernel1'
        self.base_addr = self.input_a_addr
        self.bound = self.num_clusters * configs.data_size
        self.distances = pad8(np.random.randint(size=[self.num_clusters], high=self.max_val, low=0, dtype=np.int32))
        #min distance index
        self.output = pad8(np.array([np.argmin(self.distances).astype(np.int32)]))
        self.input_addrs = [self.output_addr]
    
    def make_kernel(self):
        packet_size = configs.packet_size
        data_size = configs.data_size
        spad_addr = configs.spad_addr
        template = ''
        template += f'-kernel name = {self.kernel_name}\n'
        template += f'-kernel id = {self.kernel_id}\n'
        template += '\n'

        template += f'KERNELBODY:\n'
        
        template += f'vsetvli t0, a0, e32, m1\n'
        template += f'vle32.v v1, ADDR\n' #v1: distances
        
        template += f'li x1, {configs.spad_addr}\n'
        template += f'ld x2, (x1)\n' #x2: output addr at global memory
        template += f'vmset.m v0\n'
        template += f'vmv.v.i v5, 0\n' # v5: initialize output with 0
        
        template += f'vredmin.vs v2, v1, v1, v0\n' #v2: min distance 
        template += f'vmv.x.s x3, v2\n' # x3: min distance  
        #find the index of min distance
        template += f'vmseq.vx v3, v1, x3\n'
        template += f'vfirst.m x4, v3, v3\n' # x4: min distance index
        #store min distance index to global memory
        template += f'vmv.s.x v5, x4\n'
        template += f'vse32.v v5, (x2)\n'
        return template

    def make_input_map(self):
      return make_memory_map([(self.input_a_addr, self.distances)])

    def make_output_map(self):
      return make_memory_map([(self.input_a_addr, self.distances),
                              (self.output_addr, self.output)])
