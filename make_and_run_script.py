import os
from os import path
import random
import shutil
import subprocess
import math
from datetime import datetime
import argparse

def get_slurm_node():
    #node_list = ['g5']
    #node_list = ['n2', 'n3', 'n4', 'n1', 'n8']
    node_list = ['n9', 'n10']
    #node_list = ['n4', 'n1', 'n6']
    #node_list = ['n4', 'n1', 'n5']
    #node_list = ['n4', 'n1']
    #node_list = ['n4', 'n1', 'n3']
    return node_list[random.randint(1, len(node_list)) - 1]
    #return 'n{}'.format(random.randint(2, 6))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark', type=str, default='examples/benchmarks/pagerank/kernel0')
    parser.add_argument('-c', '--config', type = str, default = 'config/performance/M2NDP/m2ndp.config')
    parser.add_argument('-o', '--output', type = str, default = 'output')
    parser.add_argument('-n', '--num_m2ndps', type = int, default = 1)

    args = parser.parse_args()
    
    #get current working directory
    working_dir = os.getcwd()
    
    today = str(datetime.today().strftime('%Y_%m_%d'))
    
    #bin path
    sim_bin_path = os.path.join(working_dir, 'build/bin/NDPSim')
    
    #make benchmark path
    target_benchmark_path = os.path.join(working_dir, args.benchmark)
    
    #make config path
    config_path = os.path.join(working_dir, args.config)
    
    #make output path
    #config name
    config_parts = args.config.split('/')
    config_dir_name = config_parts[-2]
    config_file_name = os.path.basename(config_path) 
    config_file_name = os.path.splitext(config_file_name)[0]
    config_name = config_dir_name + '_' + config_file_name
    
    
    output_dir_path = os.path.join(working_dir, args.output, today, args.benchmark, config_name)
    
    os.makedirs(output_dir_path, exist_ok=True)
    
    assert(os.path.isdir(target_benchmark_path))
    assert(os.path.isdir(output_dir_path))
    assert(os.path.isfile(config_path))
    print(args)
    
    #make sbatch script
    sbatch_content = ''
    env_content = ''
    run_content = ''
    
    sbatch_content += "#!/bin/bash\n\n"
    job_name = '{}/{}/{}'.format(
        today, args.benchmark.replace('/', '_'), config_name)
    sbatch_content += "#SBATCH --job-name={}\n".format(job_name)
    sbatch_content += "#SBATCH --output=ndp.result\n"
    sbatch_content += "#SBATCH --partition=allcpu\n"
    sbatch_content += "#SBATCH --nodelist={}\n".format(get_slurm_node())
    sbatch_content += "#SBATCH --parsable\n"
    sbatch_content += "\n"

    # Export LD_LIBRARY_PATH to make accelsim use the gpgpusim library file 
    # in the bin directory
    env_content += "module load gnu8\n"
    env_content += "export CC=`which gcc`\n"
    env_content += "export CXX=`which g++`\n"
    env_content += "export CMAKE_C_COMPILER=$CC\n"
    env_content += "export CMAKE_CXX_COMPILER=$CXX\n"
    env_content += "\n"

    run_content += "{} --trace {} --num_hosts 1 --num_m2ndps {} --config {}\n".format(
        sim_bin_path, target_benchmark_path, args.num_m2ndps, config_path)

    with open(path.join(output_dir_path, 'srun.sh'), 'w') as file:
        file.write(sbatch_content)
        file.write(env_content)
        file.write(run_content)
    
    # go to output directory
    os.chdir(output_dir_path)
    # Run sbatch command.
    jobid = subprocess.check_output(
        ['sbatch', '--export=ALL', 'srun.sh'], encoding = 'utf-8')
    jobid = jobid.strip()
    
    print("JobID: {}".format(jobid))
    
    