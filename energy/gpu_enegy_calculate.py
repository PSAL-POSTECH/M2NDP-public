import os
import argparse 

# MEM_BK =  16 * 10**-9 #DRAM Banks 
# MEM_RD =  0.24  * 10**-9#DRAM Reads
# MEM_WR = 0.26   * 10**-9#DRAM Writes
# MEM_PRE =  6.39 * 10**-9 #DRAM Precharges
# MEM_ACT = 19.17 * 10**-9 #DRAM Activates

MEM_BK = 16
MEM_RD = 1.11 * 10**-9#DRAM Reads
MEM_WR = 1.11 * 10**-9#DRAM Writes
MEM_PRE = 1.8 * 10**-9 #DRAM Precharges
MEM_ACT =  0  * 10**-9 #DRAM Activates

# MEM_RD =  23.175 * 10**-12
# MEM_WR = 20.925 * 10**-12
# MEM_PRE =  114.75 * 10**-12
# MEM_ACT = 214.2 * 10**-12

FLIT_SIZE = 64 #8B flit size
CXL_ENERGY = 0.008 * 10**-9 * FLIT_SIZE #CXL Link 
XBAR_ENERGY =  0.086101  * 10**-9 #XBar



pwr_cmp_t  = {
  'IBP' : 'GPU SRAM', ## instruction buffer power
  'ICP' : 'GPU SRAM', # i cache
  'DCP' : 'GPU L1D', # data cache
  'TCP' : 'GPU SRAM', # tex cache
  'CCP' : 'GPU SRAM', # constant cache
  'SHRDP' : 'GPU SHARED', ## shmem?
  'RFP' : 'GPU REG', # register file
  'INTP' : 'GPU Compute', # int?
  'FPUP' : 'GPU Compute', # floating point unit?
  'DPUP' : 'GPU Compute', # double precision unit?
  'INT_MUL24P' : 'GPU Compute', #
  'INT_MUL32P' : 'GPU Compute', # 
  'INT_MULP' : 'GPU Compute', # mult
  'INT_DIVP' : 'GPU Compute', # div
  'FP_MULP' : 'GPU Compute', # mult
  'FP_DIVP' : 'GPU Compute', # div
  'FP_SQRTP' : 'GPU Compute', # sqrt
  'FP_LGP' : 'GPU Compute', # log
  'FP_SINP' : 'GPU Compute', # sin / cosine
  'FP_EXP' : 'GPU Compute', # douple precision exponent power
  'DP_MULP': 'GPU Compute', # douple precision mult power
  'DP_DIVP': 'GPU Compute', # double precision div power
  'TENSORP' : 'GPU Compute', # tensor core power
  'TEXP' : 'GPU Compute', # texture unit power
  'SCHEDP' : 'GPU Compute', # scheduler power
  'L2CP' : 'GPU L2D', # l2 cache power
  'MCP' :'DRAM', # mem ctrl power
  'NOCP' : 'NOC', # noc power
  'DRAMP' : 'DRAM', # dram power
  'PIPEP' : 'GPU Compute', # pipeline power
  'IDLE_COREP' : 'Static', #idle core power
  'CONSTP': 'Const', # constant power
  'STATICP':'Static' # static power
};
mem_energy = {'read_requests': MEM_RD,
              'write_requests': MEM_WR, 
              'precharges' : MEM_PRE, 
              'precharge_alls' : MEM_PRE * MEM_BK, 
              'refreshes': MEM_BK * (MEM_PRE + MEM_ACT),
              'activations': MEM_ACT}


def calaultate_GPU_energy( kernel_dir, const_scale = 1.0, static_scale = 1.0, config_freqs=1.0, offset= 0):
  GPUwattch = { 'IBP':0, 'ICP': 0, 'DCP':0, 'TCP': 0, 'CCP': 0, 'SHRDP':0, 'RFP':0, 'INTP':0,
        'FPUP':0, 'DPUP': 0, 'INT_MUL24P': 0, 'INT_MUL32P': 0, 'INT_MULP': 0, 'INT_DIVP': 0,
        'FP_MULP': 0, 'FP_DIVP': 0, 'FP_SQRTP': 0, 'FP_LGP': 0, 'FP_SINP': 0, 'FP_EXP': 0,
        'DP_MULP': 0, 'DP_DIVP': 0, 'TENSORP': 0, 'TEXP': 0, 'SCHEDP':0, 'L2CP':0, 'MCP':0,
        'NOCP':0, 'DRAMP':0, 'PIPEP':0, 'IDLE_COREP':0, 'CONSTP':0, 'STATICP':0}
  GPU_Energy = {
    'GPU Compute': 0.0,
    'GPU REG': 0.0,
    'GPU L1D': 0.0,
    'GPU L2D': 0.0,
    'GPU SRAM': 0.0,
    'GPU SHARED' : 0.0,
    'DRAM': 0.0,
    'Link': 0.0,
    'NOC': 0.0,
    'Static': 0.0,
    'Const': 0.0
  }
  total_enery = 0
  ## Parse GPUwattch
  kernel_names = []
  kernel_cycles = []
  energy_dict_list = []
  energy_sample_list = []
  with open(os.path.join(kernel_dir, 'gpu_stat.txt'), 'r') as f:
    kernel_name = ''
    for line in f.readlines():
      if 'kernel_name =' in line:
        kernel_name = line[line.index('kernel_name ='):].split()[2]
        kernel_names.append(kernel_name.strip(' '))
      if 'gpu_sim_cycle = ' in line:
        cycle = int(line.split()[2])
        kernel_cycles.append(cycle)

      
  with open(os.path.join(kernel_dir, 'accelwattch_power_report.log'),"r") as f:
      lines = f.readlines()
      dict_tmp = {}
      index = -1
      kernel_name = ''
      for line in lines:
        if 'kernel_name =' in line:
          index += 1
          kernel_name = line.split()[2].strip(' ')
        if 'gpu_tot_TOT_INST' in line:
          energy_dict_list.append(dict_tmp)
          print(dict_tmp)
          dict_tmp = {}
        if not 'gpu_avg' in line:
          continue
        splitted = line.split()
        if kernel_name != kernel_names[index]:
          print(kernel_name, '!=', kernel_names[index])
        key = splitted[0].strip(',').strip('gpu_avg_') 
        if key in GPUwattch.keys():
            if key == 'STATICP':
              dict_tmp[key] = float(splitted[2]) * static_scale
            elif key == 'CONSTP':
              dict_tmp[key] = float(splitted[2]) * const_scale
              print(f'{key}: {splitted[2]}')
            elif key == 'L2CP':
              dict_tmp[key] = float(splitted[2]) / 4.0 # due to we count 4 times duplicated counter
            else:
              dict_tmp[key] = float(splitted[2])
        
        
  print('Executed kernels +1 : ',len(energy_dict_list))
  for i in range(len(energy_dict_list)):
    dict_tmp = energy_dict_list[i]
    time = kernel_cycles[i] / config_freqs
    for key in dict_tmp.keys():
      if key == 'STATICP' or key == 'CONSTP':
        time = time + offset / config_freqs
      GPU_Energy[pwr_cmp_t[key]] += float((dict_tmp[key]*time))

  ## Parse Ramulator
  remote_dram_energy = 0
  
  def get_dram_energy(ramulator_out_file):
    dram_energy = 0.0
    for line in ramulator_out_file.readlines():
      splitted = line.split()
      for key in mem_energy.keys():
        if key in splitted[0]:
          dram_energy += float(splitted[1]) * mem_energy[key]
    return dram_energy

  if os.path.exists(os.path.join(kernel_dir, 'output', 'ramulator.stats')):
    with open(os.path.join(kernel_dir, 'output', 'ramulator.stats'),"r") as f:
      GPU_Energy['DRAM'] = get_dram_energy(f)
  else:
    with open(os.path.join(kernel_dir, 'ramulator.stats'),"r") as f:
      GPU_Energy['DRAM'] = get_dram_energy(f)

  ## Parse network 
  cxl_energy_total = 0
  xbar_energy_total = 0
  if os.path.exists(os.path.join(kernel_dir, 'm2ndp_stats.txt')):
    with open(os.path.join(kernel_dir, 'm2ndp_stats.txt'),"r") as f:
      count = 0
      for line in f.readlines():
        if 'Total sent flits' in line:
            if count < 2:
              total_enery += CXL_ENERGY * float(line.split()[-1])
              cxl_energy_total += CXL_ENERGY * float(line.split()[-1])
              GPU_Energy['Link'] += CXL_ENERGY * float(line.split()[-1])
            else:
              total_enery += XBAR_ENERGY * float(line.split()[-1])
              xbar_energy_total += XBAR_ENERGY * float(line.split()[-1])
              GPU_Energy['NOC'] += XBAR_ENERGY * float(line.split()[-1])
            count += 1
  return GPU_Energy

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--kernel_dir', type=str, default='kernel_dir')
  args = parser.parse_args()
  print(calaultate_GPU_energy(args.kernel_dir))