from abc import *
import os
import numpy as np

class NdpKernel:
    def __init__(self) -> None:
        self.kernel_name = ''
        self.sync = False
        self.kernel_id = 0
        self.base_addr = 0
        self.base_addrs = []
        self.bound = 0
        self.smem_size = 0
        self.input_addrs = []
        self.input_addrs_list = [[]]

    @abstractmethod
    def make_kenrel(self):
        pass
    
    @abstractmethod
    def make_input_map(self):
        pass

    @abstractmethod
    def make_output_map(self):
        pass

    def get_kernel_info(self, num_m2ndps=1):
        smem_size = self.smem_size + 8 * len(self.input_addrs)
        if num_m2ndps == 1:
            result = f'{self.sync} {self.kernel_id} 0x{self.base_addr:x} 0x{self.bound:x}'
            result += f' 0x{smem_size:x}'
            result += f' 0x{8 * len(self.input_addrs):x}'
            #have to add delimiter of float32 type (FP32)
            first_fp32 = False
            for addr in self.input_addrs:
                if isinstance(addr, np.float32):
                    if not first_fp32:
                        result += ' FP32'
                        first_fp32 = True
                    result += f' {addr}'
                else:
                    result += f' 0x{addr:x}'
            return result
        else:
            result_list = []
            for i in range(num_m2ndps):
                result = f'{self.sync} {self.kernel_id} 0x{self.base_addrs[i]:x} 0x{self.bound:x}'
                smem_size = self.smem_size + 8 * len(self.input_addrs_list[i])
                result += f' 0x{smem_size:x}'
                result += f' 0x{8 * len(self.input_addrs_list[i]):x}'
                #have to add delimiter of float32 type (FP32)
                first_fp32 = False
                for addr in self.input_addrs_list[i]:
                    if isinstance(addr, np.float32):
                        if not first_fp32:
                            result += ' FP32'
                            first_fp32 = True
                        result += f' {addr}'
                    else:
                        result += f' 0x{addr:x}'
                result_list.append(result)
            return result_list
        
class MultipleKernel(NdpKernel):
    def __init__(self, repeat=1) -> None:
        super().__init__()
        self.repeat = repeat
        self.base_offset = None
        self.input_addrs_offsets = []
    
    def get_kernel_info(self):
        smem_size = self.smem_size + 8 * len(self.input_addrs)
        result = ''
        base_offset = self.base_offset if self.base_offset is not None else self.bound
        for itr in range(self.repeat):
            base_addr = self.base_addr + itr * base_offset
            result += f'{self.sync} {self.kernel_id} 0x{base_addr:x} 0x{self.bound:x}'
            result += f' 0x{smem_size:x}'
            result += f' 0x{8 * len(self.input_addrs):x}'
            first_fp32 = False
            for addr , offset in zip(self.input_addrs, self.input_addrs_offsets):
                if(isinstance(addr, np.float32)):
                    if not first_fp32:
                        result += ' FP32'
                        first_fp32 = True
                    result += f' {addr+ itr * offset}'
                else:
                    result += f' 0x{addr + itr * offset:x}'
            result += '\n'
        return result
    

def pad_n(np_array, n=8, constant_values=0):
    if len(np_array) % n == 0:
        return np_array
    else:
        return np.pad(np_array, (0, 8 - len(np_array) % 8), 'constant', constant_values=constant_values)
    
def pad16(np_array, constant_values=0):
    if len(np_array) % 16 == 0:
        return np_array
    else:
        return np.pad(np_array, (0, 16 - len(np_array) % 16), 'constant', constant_values=constant_values)
        return np.pad(np_array, (0, n - len(np_array) % n), 'constant', constant_values=constant_values)

def pad8(np_array, constant_values=0):
    return pad_n(np_array, n=8, constant_values=constant_values)

def pad(np_array, size=8, constant_values=0):
    if len(np_array) % size == 0:
        return np_array
    else:
        return np.pad(np_array, (0, size - len(np_array) % size), 'constant', constant_values=constant_values)

def make_memory_map(memory_map, packet_size=32):
    template = ''
    for addr, data in memory_map:
        template += '_META_\n'
        if data.dtype == np.float16:
            template += 'float16\n'
            elems_per_row = packet_size // 2
        elif data.dtype == np.float32:
            template += 'float32\n'
            elems_per_row = packet_size // 4
        elif data.dtype == np.int16:
            template += 'int16\n'
            elems_per_row = packet_size // 2
        elif data.dtype == np.int32:
            template += 'int32\n'
            elems_per_row = packet_size // 4
        elif data.dtype == np.int64:
            template += 'int64\n'
            elems_per_row = packet_size // 8
        elif data.dtype == np.int8:
            template += 'char8\n'
            elems_per_row = packet_size // 1
        elif data.dtype == np.uint8:
            template += 'uint8\n'
            elems_per_row = packet_size // 1
        else:
            raise ValueError(f'Unsupported data type {data.dtype}')

        # Add Padding
        data = pad_n(data.reshape(-1), n=elems_per_row)
        template += '_DATA_\n'
        offset = 0
        for row in np.reshape(data, (-1, elems_per_row)):
            template += f'0x{addr + offset * packet_size:x}'
            for elem in row:
                template += f' {elem}'
            template += '\n'
            offset += 1
    return template

def make_input_files(kernel_name, kernel_string, input_memory_map, output_memory_map, launch_info, file_dir='./'):
    os.makedirs(file_dir, exist_ok=True)
    file = open(f'{file_dir}/{kernel_name}.traceg', 'w')
    file.write(kernel_string)
    file.close()
    file = open(f'{file_dir}/{kernel_name}_input.data', 'w')
    file.write(input_memory_map)
    file.close()
    file = open(f'{file_dir}/{kernel_name}_output.data', 'w')
    file.write(output_memory_map)
    file.close()
    file = open(f'{file_dir}/{kernel_name}_launch.txt', 'w')
    file.write(f'{launch_info}')
    file.close()
    file = open(f'{file_dir}/kernelslist.g', 'w')
    file.write(f'{kernel_name}')
    file.close()

def make_input_files_multi_ndp(kernel_name, kernel_string, input_memory_map, output_memory_map, launch_info, file_dir='./', model_parallelism=1):
    for i in range(model_parallelism):
        os.makedirs(file_dir + f'/{i}', exist_ok=True)
        file = open(f'{file_dir}/{i}/{kernel_name}.traceg', 'w')
        file.write(kernel_string)
        file.close()
        file = open(f'{file_dir}/{i}/{kernel_name}_input.data', 'w')
        file.write(input_memory_map[i])
        file.close()
        file = open(f'{file_dir}/{i}/{kernel_name}_output.data', 'w')
        file.write(output_memory_map[i])
        file.close()
        file = open(f'{file_dir}/{i}/{kernel_name}_launch.txt', 'w')
        file.write(f'{launch_info[i]}')
        file.close()
        file = open(f'{file_dir}/{i}/kernelslist.g', 'w')
        file.write(f'{kernel_name}')
        file.close()


def execute_functional_sim(kernel_name, config='m2ndp.config', file_dir='./'):
    os.system("echo \"Running FuncSim..\"")
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    exec_file = os.path.join(this_file_dir, '../../build/bin/FuncSim')
    config_dir = os.path.join(this_file_dir, '../../config/functional_only')
    script = f'{exec_file} --ndp_trace {file_dir}/{kernel_name}.traceg' \
                            + f' --memory_map  {file_dir}/{kernel_name}_input.data' \
                            + f' --target_map  {file_dir}/{kernel_name}_output.data' \
                            + f' --launch_file {file_dir}/{kernel_name}_launch.txt ' \
                            + f' --config {config_dir}/{config}'
    print(script)
    os.system(script)

def get_min_per_ndp_unit_count(config, size):
    stride = config.stride
    ndp_units = config.ndp_units
    packet_size = config.packet_size

    counts = [0 for _ in range(ndp_units)]
    addr = 0
    while addr < size:
        counts[addr // stride % ndp_units] += 1
        addr += packet_size
    return min( x for x in counts if x > 0)