import argparse
import os
import csv
import re
import yaml



Config_Categories = {
  'I_ISSUE_CNT': 'NDP Compute',
  'F_ISSUE_CNT': 'NDP Compute',
  'SF_ISSUE_CNT': 'NDP Compute', 
  'ADDR_ISSUE_CNT': 'NDP Compute', 
  'LDST_ISSUE_CNT': 'NDP Compute', 
  'SPAD_ISSUE_CNT': 'NDP Compute', 
  'V_ISSUE_CNT': 'NDP Compute', 
  'V_SF_ISSUE_CNT': 'NDP Compute', 
  'V_ADDR_ISSUE_CNT': 'NDP Compute', 
  'V_LDST_ISSUE_CNT': 'NDP Compute', 
  'V_SPAD_ISSUE_CNT': 'NDP Compute', 
  'SPAD_ACCESS_CNT': 'NDP SHARED',
  'XREG_RD': 'NDP REG',
  'FREG_RD': 'NDP REG',
  'VREG_RD': 'NDP REG',
  'XREG_WR': 'NDP REG',
  'FREG_WR': 'NDP REG',
  'VREG_WR': 'NDP REG',
  'ITLB_RH' : 'NDP SRAM',
  'ITLB_RM' : 'NDP SRAM',
  'ITLB_WH' : 'NDP SRAM',
  'ITLB_WM' : 'NDP SRAM',
  'DTLB_RH' : 'NDP SRAM',
  'DTLB_RM' : 'NDP SRAM',
  'DTLB_WH' : 'NDP SRAM',
  'DTLB_WM' : 'NDP SRAM',
  'L0I-CACHE_RH' : 'NDP SRAM',
  'L0I-CACHE_RM' : 'NDP SRAM',
  'L0I-CACHE_WH' : 'NDP SRAM',
  'L0I-CACHE_WM' : 'NDP SRAM',
  'L1I-CACHE_RH' : 'NDP SRAM',
  'L1I-CACHE_RM' : 'NDP SRAM',
  'L1I-CACHE_WH' : 'NDP SRAM',
  'L1I-CACHE_WM' : 'NDP SRAM',
  'DCACHE_H' : 'NDP SRAM',
  'DCACHE_M' : 'NDP SRAM',
  'L1D_RH' : 'NDP L1D',
  'L1D_RM' : 'NDP L1D',
  'L1D_WH' : 'NDP L1D',
  'L1D_WM' : 'NDP L1D',
  'L2D_RH' : 'NDP L2D',
  'L2D_RM' : 'NDP L2D',
  'L2D_WH' : 'NDP L2D',
  'L2D_WM' : 'NDP L2D',
  'MEM_BK': 'DRAM',
  'MEM_RD': 'DRAM',
  'MEM_WR': 'DRAM',
  'MEM_PRE':'DRAM',
  'MEM_ACT': 'DRAM',
  'MEM_REF' : 'DRAM',
  'LINK': 'Link',
  'XBAR': 'NOC',
  'STATIC': 'NDP Static',
  'CONST': 'NDP Const',
}

def calculate_NDP_energy(input_file, config, scale):
  NDP_Energy = {
  'NDP Const' : 0.0,
  'NDP Static' : 0.0,
  'NDP Compute' : 0.0,
  'NDP REG' : 0.0,
  'NDP SRAM' : 0.0,
  'NDP SHARED' : 0.0,
  'NDP L1D' : 0.0,
  'NDP L2D' : 0.0,
  'Link': 0.0,
  'DRAM': 0.0,
  'NOC': 0.0,
  }
  lines = []
  with open(input_file) as f:
    lines = f.readlines()
  config_file = config
  with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config['MEM_REF'] = config['MEM_BK'] * ( config['MEM_ACT'] + config['MEM_PRE'] )
    config['LINK'] = config['LINK_FLIT_SIZE'] * config['LINK']
    config['CONST'] *= scale
  energy = 0;
  for line in lines:
    line = line.replace(' ', '')
    key, value = line.split(':')
    if key in config:
      energy += config[key] * float(value)
      NDP_Energy[Config_Categories[key]] += config[key] * float(value)
    else:
      print(f"Warning: {key} not in config")
  for key in NDP_Energy:
    if key != 'NDP Static' and key != 'NDP Const':
      NDP_Energy[key] = NDP_Energy[key] * 10**-9
  return NDP_Energy


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get energy from ndp_stat')
  parser.add_argument('--config', type=str, default='.', help='config file')
  parser.add_argument('--input', type=str, default='energy.out', help='input ndp_stat file')
  parser.add_argument('--output', type=str, default='.', help='output path')
  args = parser.parse_args()
  
  # Read input file
  input_file = args.input
  energy = calculate_NDP_energy(input_file, args.config)
  print(energy)

  if args.output:
    csvfile = open(args.output, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(['Category', 'Energy (nJ)'])
    for key in energy.keys():
      writer.writerow([key, energy[key]])
      
  
