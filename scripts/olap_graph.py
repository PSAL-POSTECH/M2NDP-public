import os
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np

breakdown_labels = {'Evaluate': 0, 'Filter': 1, 'Etc': 2, 'Total': 3}
#nanoseconds
baseline_runtime = {'Q14': [16848395.9375, 3830280.625, 17104721.88], 'Q6': [25088926.875, 4080347.8125, 1085796.875], 'Q1_1': [27191954.0625, 1363036.25, 1025373.4375], 'Q1_2': [22541265.9375, 658127.1875, 916796.25], 'Q1_3': [23709930.9375, 661586.25, 907629.0625]}
cpu_ndp_evaluate_runtime = {'Q14': [177053.5875], 'Q6': [452537.6787], 'Q1_1': [538964.0602], 'Q1_2': [538964.0602], 'Q1_3': [538964.0602]}
ideal_ndp_evaluate_runtime = {'Q14': [119121.5625], 'Q6': [306150.9375], 'Q1_1': [364760.9375], 'Q1_2': [364760.9375], 'Q1_3': [364760.9375]}
query_list = ['Q14', 'Q6', 'Q1_1', 'Q1_2', 'Q1_3']
labels = ['Baseline', 'CPU-NDP', 'NDP', 'Ideal NDP']
olap_kernel_list = ['imdb_gteq_lt_int64', 'imdb_lt_int64', 'imdb_gt_lt_fp32', 'imdb_three_col_and']
ndp_cycle = {}

def parsing_runtime(log_file_dir, target_ndp_kernel, target_m2ndp_dir):
    #make the log file path
    log_file_path = log_file_dir + "/" + target_ndp_kernel + "/sim.log"
    #check log file path exists or not
    if not os.path.exists(log_file_path):
        print("log file path does not exist")
        return math.inf
    
    ndp_cycle = 0
    #open the log file and find the line with "Gantt info: host"
    with open(log_file_path, "r") as log_file:
        lines = log_file.readlines()
        for line in lines:
            if "Gantt info: host" in line:
                #get the ndp cycle
                ndp_cycle = int(line.split("ndp cycle ")[1].split(" at CXL")[0])
    
    return ndp_cycle



#main function that get ndp cycle using parsing_runtime function and go for a loop for all the target ndp kernels and make a csv for each target ndp kernels
#receive the log file directory, target NDP kernel names and target ndp configuration name as input argument

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--log_file_dir', type=str, default='./outputs')
    parser.add_argument('-c', '--target_m2ndp_dir', type=str, default='0')
    parser.add_argument('-f', '--freq', type=int, default=2) #GHz
    args = parser.parse_args()
    kernel_result = {}
    for kernel in olap_kernel_list:
        ndp_cycle = parsing_runtime(args.log_file_dir, kernel, args.target_m2ndp_dir)
        kernel_result[kernel] = ndp_cycle / args.freq #ns
    
    #calculate the ndp cycle for each kernel
    ndp_evaluate_runtime = {'Q14': [kernel_result['imdb_gteq_lt_int64']],
                            'Q6': [kernel_result['imdb_gteq_lt_int64'] + kernel_result['imdb_lt_int64'] + kernel_result['imdb_gt_lt_fp32'] + kernel_result['imdb_three_col_and']],
                            'Q1_1': [kernel_result['imdb_gteq_lt_int64']*2 + kernel_result['imdb_lt_int64'] + kernel_result['imdb_three_col_and']],
                            'Q1_2': [kernel_result['imdb_gteq_lt_int64']*3 + kernel_result['imdb_three_col_and']],
                            'Q1_3': [kernel_result['imdb_gteq_lt_int64']*3 + kernel_result['imdb_three_col_and']]}
    
    #Runtime normalized to baseline
    for query in query_list:
        total = 0
        for i in range(3):
            total += baseline_runtime[query][i]
        baseline_runtime[query].append(total)
        
    baseline_norm_runtime = {}
    ndp_norm_runtime = {}
    cpu_ndp_norm_runtime = {}
    ideal_ndp_norm_runtime = {}
    ndp_eval_speedup = {}
    cpu_ndp_eval_speedup = {}
    ideal_ndp_eval_speedup = {}
    for query in query_list:
        baseline_norm_runtime[query] = []
        ndp_norm_runtime[query] = []
        ndp_eval_speedup[query] = []
        cpu_ndp_norm_runtime[query] = []
        ideal_ndp_norm_runtime[query] = []
        cpu_ndp_eval_speedup[query] = []
        ideal_ndp_eval_speedup[query] = []
        for i in range(3):
            if i == breakdown_labels['Evaluate']:
                baseline_norm_runtime[query].append(baseline_runtime[query][i] / baseline_runtime[query][breakdown_labels['Total']])
                ndp_norm_runtime[query].append(ndp_evaluate_runtime[query][i] / baseline_runtime[query][breakdown_labels['Total']])
                ndp_eval_speedup[query].append(int(round(baseline_runtime[query][i] / ndp_evaluate_runtime[query][i])))
                cpu_ndp_norm_runtime[query].append(cpu_ndp_evaluate_runtime[query][i] / baseline_runtime[query][breakdown_labels['Total']])
                cpu_ndp_eval_speedup[query].append(int(round(baseline_runtime[query][i] / cpu_ndp_evaluate_runtime[query][i])))
                ideal_ndp_norm_runtime[query].append(ideal_ndp_evaluate_runtime[query][i] / baseline_runtime[query][breakdown_labels['Total']])
                ideal_ndp_eval_speedup[query].append(int(round(baseline_runtime[query][i] / ideal_ndp_evaluate_runtime[query][i])))
            else:
                baseline_norm_runtime[query].append(baseline_runtime[query][i] / baseline_runtime[query][breakdown_labels['Total']])
                ndp_norm_runtime[query].append(baseline_runtime[query][i] / baseline_runtime[query][breakdown_labels['Total']])
                cpu_ndp_norm_runtime[query].append(baseline_runtime[query][i] / baseline_runtime[query][breakdown_labels['Total']])
                ideal_ndp_norm_runtime[query].append(baseline_runtime[query][i] / baseline_runtime[query][breakdown_labels['Total']])
    
    #plotting
    fig, ax1 = plt.subplots(figsize=(14, 10))
    bar_width = 0.15
    gap = 0.05
    index = np.arange(len(query_list))
    small_index = np.arange(len(query_list) * len(labels))
    colors = ['#a6cee3', '#b2df8a', '#fb9a99']
    
    #stacked bar graph
    bottom_baseline = np.zeros(len(query_list))
    bottom_cpu_ndp = np.zeros(len(query_list))
    bottom_ndp = np.zeros(len(query_list))
    bottom_ideal_ndp = np.zeros(len(query_list))
    for i, category in enumerate(['Etc', 'Filter', 'Evaluate']):
        ax1.bar(index - bar_width*1.5 - gap*1.5, [baseline_norm_runtime[query][breakdown_labels[category]] for query in query_list], bar_width,
                bottom=bottom_baseline, color=colors[i], label=f'{category}')
        ax1.bar(index - bar_width/2 - gap/2, [cpu_ndp_norm_runtime[query][breakdown_labels[category]] for query in query_list], bar_width,
                bottom=bottom_cpu_ndp, color=colors[i])
        ax1.bar(index + bar_width/2 + gap/2, [ndp_norm_runtime[query][breakdown_labels[category]] for query in query_list], bar_width,
                bottom=bottom_ndp, color=colors[i])
        ax1.bar(index + bar_width*1.5 + gap*1.5, [ideal_ndp_norm_runtime[query][breakdown_labels[category]] for query in query_list], bar_width,
                bottom=bottom_ideal_ndp, color=colors[i])
        bottom_baseline += np.array([baseline_norm_runtime[query][breakdown_labels[category]] for query in query_list])
        bottom_ndp += np.array([ndp_norm_runtime[query][breakdown_labels[category]] for query in query_list])
        bottom_cpu_ndp += np.array([cpu_ndp_norm_runtime[query][breakdown_labels[category]] for query in query_list])
        bottom_ideal_ndp += np.array([ideal_ndp_norm_runtime[query][breakdown_labels[category]] for query in query_list])
    
    # Plotting speedup line
    ax2 = ax1.twinx()
    for i in range(len(query_list)):
        # Baseline point at 1
        ax2.plot(index[i] - bar_width*1.5 - gap*1.5, 1, 'bs', label='"Evaluate" speedup' if i == 0 else '')
        ax2.text(index[i] - bar_width*1.5 - gap*1.5, 1.3, '1', ha='center', va='bottom', color='black')
        
        #CPU-NDP point representing the speedup
        ax2.plot(index[i] - bar_width/2 - gap/2, cpu_ndp_eval_speedup[query_list[i]][0], 'bs')
        ax2.text(index[i] - bar_width/2 - gap/2, cpu_ndp_eval_speedup[query_list[i]][0] + 0.3, f'{cpu_ndp_eval_speedup[query_list[i]][0]}', ha='center', va='bottom', color='black')
        
        # NDP point representing the speedup
        ax2.plot(index[i] + bar_width/2 + gap/2, ndp_eval_speedup[query_list[i]][0], 'bs')
        ax2.text(index[i] + bar_width/2 + gap/2, ndp_eval_speedup[query_list[i]][0] + 0.3, f'{ndp_eval_speedup[query_list[i]][0]}', ha='center', va='bottom', color='black')
        
        # Ideal NDP point representing the speedup
        ax2.plot(index[i] + bar_width*1.5 + gap*1.5, ideal_ndp_eval_speedup[query_list[i]][0], 'bs')
        ax2.text(index[i] + bar_width*1.5 + gap*1.5, ideal_ndp_eval_speedup[query_list[i]][0] + 0.3, f'{ideal_ndp_eval_speedup[query_list[i]][0]}', ha='center', va='bottom', color='black')
        
        # Connecting lines within each query
        ax2.plot([index[i] - bar_width*1.5 - gap*1.5, index[i] - bar_width/2 - gap/2], [1, cpu_ndp_eval_speedup[query_list[i]][0]], 'k--')
        ax2.plot([index[i] - bar_width/2 - gap/2, index[i] + bar_width/2 + gap/2], [cpu_ndp_eval_speedup[query_list[i]][0], ndp_eval_speedup[query_list[i]][0]], 'k--')
        ax2.plot([index[i] + bar_width/2 + gap/2, index[i] + bar_width*1.5 + gap*1.5], [ndp_eval_speedup[query_list[i]][0], ideal_ndp_eval_speedup[query_list[i]][0]], 'k--')    
    
    # Formatting
    ax1.set_ylabel('Runtime normalized to baseline (bar chart)')
    ax2.set_ylabel('Speedup for "Evaluate" kernel (line chart)')
    ax1.set_xticks(index)
    ax1.set_xticklabels([])
    ax1.legend(loc='upper left', bbox_to_anchor=(0,1), ncol=3)
    ax2.legend(loc='upper right')

    # Adding Baseline and NDP labels under each bar group with more space between lines
    for i in range(len(query_list)):
        ax1.text(index[i] - bar_width*1.5 - gap*1.5, -0.01, 'Baseline', ha='center', va='top', color='black', fontsize=10, rotation=90)
        ax1.text(index[i] - bar_width/2 - gap/2, -0.01, 'CPU-NDP', ha='center', va='top', color='black', fontsize=10, rotation=90)
        ax1.text(index[i] + bar_width/2 + gap/2, -0.01, 'NDP', ha='center', va='top', color='black', fontsize=10, rotation=90)
        ax1.text(index[i] + bar_width*1.5 + gap*1.5, -0.01, 'Ideal NDP', ha='center', va='top', color='black', fontsize=10, rotation=90)  
    for i in range(len(query_list)):
        ax1.text(index[i], -0.1, f'{query_list[i]}', ha='center', va='top', color='black', fontsize=10)

    plt.tight_layout()
    # Save the figure
    plt.savefig(f'outputs/Figure_10a.png')
