import pandas as pd
import sys
import os
import math
import seaborn as sns
import matplotlib.pyplot as plt

CONFIG_NAME = 'M2NDP'
io_overhead = 1500
output_dir = os.path.join('outputs')
columns = ['Model','Config','Time(ns)']

def get_ndp_cycle_from_result_dir(ndp_config, ndp_model):
  try:
    expr_dir = os.path.join(output_dir, ndp_model)
    result_file = os.path.join(expr_dir, f'sim.log')
    bw_printed = False
    for line in open(result_file):
      if 'EXPR FINISHED' in line:
        print(int(line.split()[-1]))
        return int(line.split()[-1]) * 0.5 # 0.5 ns per cycle
    print('WARN: cannot find tot_sim_cycle in {}'.format(result_file))
  except:
    print('WARN: cannot find result of {} {}'.format(ndp_config, ndp_model))
  return math.inf




def get_histo():
  bins = [256, 4096]
  df = pd.DataFrame(columns=columns)
  for bin in bins:
    time = get_ndp_cycle_from_result_dir(CONFIG_NAME, f'histo{bin}')
    # df = df.append({'Model':f'HISTO{bin}', 'Config':'M2uthr', 'Time(ns)': time + io_overhead}, ignore_index=True)
    df = df.append({'Model':f'HISTO{bin}', 'Config':'M2NDP', 'Time(ns)': time}, ignore_index=True)
  return df

def get_spmv():
  time = get_ndp_cycle_from_result_dir(CONFIG_NAME, 'spmv')
  df = pd.DataFrame(columns=columns)
  # df = df.append({'Model':'SPMV', 'Config':'M2uthr', 'Time(ns)': time + io_overhead}, ignore_index=True)
  df = df.append({'Model':'SPMV', 'Config':'M2NDP', 'Time(ns)': time}, ignore_index=True)
  return df

def get_pagerank():
  total_iter = 20
  time0 = get_ndp_cycle_from_result_dir(CONFIG_NAME, 'pagerank0')
  time1 = get_ndp_cycle_from_result_dir(CONFIG_NAME, 'pagerank1')
  time2 = get_ndp_cycle_from_result_dir(CONFIG_NAME, 'pagerank2')
  time3 = get_ndp_cycle_from_result_dir(CONFIG_NAME, 'pagerank3')
  time = time0 + time1 + total_iter * (time2 + time3)
  pg_io_overhead = io_overhead + io_overhead + total_iter * (io_overhead + io_overhead)
  df = pd.DataFrame(columns=columns)
  # df = df.append({'Model':'PGRANK', 'Config':'M2uthr', 'Time(ns)': time + pg_io_overhead}, ignore_index=True)
  df = df.append({'Model':'PGRANK', 'Config':'M2NDP', 'Time(ns)': time}, ignore_index=True)
  return df

def get_sssp():
  total_iter = 1200
  time0 = get_ndp_cycle_from_result_dir(CONFIG_NAME, 'ssspinit')
  time1 = get_ndp_cycle_from_result_dir(CONFIG_NAME, 'sssp0')
  time2 = get_ndp_cycle_from_result_dir(CONFIG_NAME, 'sssp1')
  time = time0 + total_iter * (time1 + time2)
  sssp_io_overhead = io_overhead + 1200 * (io_overhead + io_overhead)
  df = pd.DataFrame(columns=columns)
  # df = df.append({'Model':'SSSP', 'Config':'M2uthr', 'Time(ns)': time + sssp_io_overhead}, ignore_index=True)
  df = df.append({'Model':'SSSP', 'Config':'M2NDP', 'Time(ns)': time}, ignore_index=True)
  return df

def get_dlrm():
  batches = [4, 32, 256]
  df = pd.DataFrame(columns=columns)
  for batch in batches:
    time = get_ndp_cycle_from_result_dir(CONFIG_NAME, f'sls_b{batch}')
    # df = df.append({'Model':f'DLRM(SLS)-B{batch}', 'Config':'M2uthr', 'Time(ns)': time + io_overhead}, ignore_index=True)
    df = df.append({'Model':f'DLRM(SLS)-B{batch}', 'Config':'M2NDP', 'Time(ns)': time}, ignore_index=True)
  return df
  
def get_opt():
  models = ['opt-2.7b', 'opt-30b']
  model_names = ['OPT-2.7B(Gen)', 'OPT-30B(Gen)']
  layers = [('attention1', 1),
            ('attention2', 1),
            ('fc1', 1),
            ('fc2', 1),
            ('layernorm_0', 2),
            ('layernorm_1', 2),
            ('out_proj', 1),
            ('qkv_proj', 1),
            ('relu', 1),
            ('residual', 4)]
  df = pd.DataFrame(columns=columns)
  for model, name in zip(models, model_names):
    time = 0
    tot_io_overhead = 0
    for layer, num in layers:
      time += get_ndp_cycle_from_result_dir(CONFIG_NAME, f'{model}/{layer}') * num
      tot_io_overhead += io_overhead * num
    # df = df.append({'Model':model, 'Config':'M2uthr', 'Time(ns)': time + tot_io_overhead}, ignore_index=True)
    df = df.append({'Model':name, 'Config':'M2NDP', 'Time(ns)': time}, ignore_index=True)
  return df


if __name__ == '__main__':
  # Read the results from the result directory
  df = pd.read_csv(os.path.join(output_dir, 'gpu_baseline.csv'))
  df = df.append(get_spmv(), ignore_index=True)
  df = df.append(get_histo(), ignore_index=True)
  df = df.append(get_pagerank(), ignore_index=True)
  df = df.append(get_sssp(), ignore_index=True)
  df = df.append(get_dlrm(), ignore_index=True)
  df = df.append(get_opt(), ignore_index=True)

  #Speedup
  for model in df['Model'].unique():
    baseline_time = df[(df['Model'] == model) & (df['Config'] == 'Baseline')]['Time(ns)'].values[0]
    df.loc[(df['Model'] == model), 'Speedup'] = baseline_time / df[(df['Model'] == model)]['Time(ns)']
  #Geomean of speedup for each config
  for config in df['Config'].unique():
    geomean = df[df['Config'] == config]['Speedup'].prod() ** (1/len(df[df['Config'] == config]))
    df = df.append({'Model':'Geomean', 'Config':config, 'Speedup':geomean}, ignore_index=True)
  df.to_csv(os.path.join(output_dir,'gpu_speedup.csv'))

  # Plot the results
  fig, ax = plt.subplots(figsize=(12, 8))
  sns.barplot(x='Model', y='Speedup', hue='Config', data=df, ax=ax)
  #add data labels rotation 90
  for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', rotation=90)
  ax.set_ylabel('Speedup')
  ax.set_xlabel('Model')
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.savefig('outputs/Figure_10c.png')