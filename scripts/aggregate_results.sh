#!/bin/bash

OUTPUT_DIR=`pwd`/outputs

#Figure. 10a (CPU workloads)
python3 scripts/olap_graph.py -d $OUTPUT_DIR

#Figure. 10c (GPU workloads)
python3 scripts/aggregate_gpu_results.py


