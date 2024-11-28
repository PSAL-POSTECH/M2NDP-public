#!/bin/bash

# Define the maximum number of concurrent processes
MAX_PROCESSES=1
BIN1=`pwd`/build/bin/NDPSim
BIN2=`pwd`/build2/bin/NDPSim
TRACE_DIR=`pwd`/traces
CONFIG_FILE1=`pwd`/config/performance/M2NDP/m2ndp.config
CONFIG_FILE2=`pwd`/config/performance/M2NDP/m2ndp_64B.config
ARGS1="--num_hosts 1 --num_m2ndps 1 --synthetic_memory false"
ARGS2="--num_hosts 1 --num_m2ndps 1 --synthetic_memory true --serial_launch true"
# Define the job list (binary files to run)
JOB_LIST=(
  "imdb_gt_lt_fp32"
  "imdb_gteq_lt_int64"
  "imdb_lt_int64"
  "imdb_three_col_and"
  "spmv"
  "histo256"
  "histo4096"
  "pagerank0"
  "pagerank1"
  "pagerank2"
  "pagerank3"
  "ssspinit"
  "sssp0"
  "sssp1"
  "sls_b4"
  "sls_b32"
  "sls_b256"
  "opt-2.7b/attention1"
  "opt-2.7b/attention2"
  "opt-2.7b/fc1"
  "opt-2.7b/fc2"
  "opt-2.7b/layernorm_0"  
  "opt-2.7b/layernorm_1"  
  "opt-2.7b/out_proj"  
  "opt-2.7b/qkv_proj"  
  "opt-2.7b/relu"  
  "opt-2.7b/residual"
  "opt-30b/attention1"
  "opt-30b/attention2"
  "opt-30b/fc1"
  "opt-30b/fc2"
  "opt-30b/layernorm_0"  
  "opt-30b/layernorm_1"  
  "opt-30b/out_proj"  
  "opt-30b/qkv_proj"  
  "opt-30b/relu"  
  "opt-30b/residual"
)
# Function to run a job
run_job() {
  local job=$1
  #IF "sls" in job name, then use ARGS2
  if [[ $job == *"sls"* ]]; then
    ARGS=$ARGS2
  else 
    ARGS=$ARGS1
  fi
  if [[ $job == *"imdb"* ]]; then
    BIN=$BIN2
    CONFIG_FILE=$CONFIG_FILE2
  else
    BIN=$BIN1
    CONFIG_FILE=$CONFIG_FILE1
  fi
  echo "Starting $job..."
  TRACE_FEILE=$TRACE_DIR/$job
  
  mkdir -p ./outputs/$job
  pushd ./outputs/$job
  $BIN --trace $TRACE_FEILE --config $CONFIG_FILE $ARGS > sim.log 2>&1
  echo "$job finished."
  popd
}

# Function to manage concurrent processes
manage_jobs() {
  local job
  local running_jobs=0

  for job in "${JOB_LIST[@]}"; do
    # Launch job in the background
    run_job "$job" &

    # Increment the running jobs counter
    ((running_jobs++))

    # If we've reached the MAX_PROCESSES limit, wait for one to finish
    if (( running_jobs >= MAX_PROCESSES )); then
      wait -n
      ((running_jobs--))
    fi
  done

  # Wait for any remaining background jobs to finish
  wait
}

# Run the job manager
manage_jobs

echo "All jobs completed."
