#!/bin/bash
#OLAP
echo "Generating OLAP traces"
python3 examples/runner.py --kernel LtINT64Kernel --output_dir ./traces/imdb_lt_int64 --skip_functional_sim
python3 examples/runner.py --kernel GtEqLtINT64Kernel --output_dir ./traces/imdb_gteq_lt_int64 --skip_functional_sim
python3 examples/runner.py --kernel GtLtFP32Kernel --output_dir ./traces/imdb_gt_lt_fp32 --skip_functional_sim
python3 examples/runner.py --kernel ThreeColANDKernel --output_dir ./traces/imdb_three_col_and --skip_functional_sim
#Histogram
echo "Generating Histogram 256 traces"
python3 examples/runner.py --kernel HistogramKernel --output_dir ./traces/histo256 --skip_functional_sim --arg 256
echo "Generating Histogram 4096 traces"
python3 examples/runner.py --kernel HistogramKernel --output_dir ./traces/histo4096 --skip_functional_sim --arg 4096
#SPMV
echo "Generating SpMV traces"
python3 examples/runner.py --kernel SpMV --output_dir ./traces/spmv --skip_functional_sim
#PageRank
echo "Generating PageRank traces"
python3 examples/runner.py --kernel PageRankKernel0 --output_dir ./traces/pagerank0 --skip_functional_sim
python3 examples/runner.py --kernel PageRankKernel1 --output_dir ./traces/pagerank1 --skip_functional_sim
python3 examples/runner.py --kernel PageRankKernel2Split --output_dir ./traces/pagerank2 --skip_functional_sim
python3 examples/runner.py --kernel PageRankKernel3 --output_dir ./traces/pagerank3 --skip_functional_sim
#SSSP
echo "Generating SSSP traces"
python3 examples/runner.py --kernel SsspInitKernel --output_dir ./traces/ssspinit --skip_functional_sim
python3 examples/runner.py --kernel SsspKernel0 --output_dir ./traces/sssp0 --skip_functional_sim
python3 examples/runner.py --kernel SsspKernel1 --output_dir ./traces/sssp1 --skip_functional_sim
#DLRM
echo "Generating DLRM-B4 traces"
python3 examples/runner.py --kernel SparseLengthSumUnrolledKernel --output_dir ./traces/sls_b4 --skip_functional_sim --arg 4
echo "Generating DLRM-B32 traces"
python3 examples/runner.py --kernel SparseLengthSumUnrolledKernel --output_dir ./traces/sls_b32 --skip_functional_sim --arg 32
echo "Generating DLRM-B256 traces"
python3 examples/runner.py --kernel SparseLengthSumUnrolledKernel --output_dir ./traces/sls_b256 --skip_functional_sim --arg 256
#OPT
echo "Generating OPT-2.7B traces"
python3 examples/opt_runner.py --model='opt-2.7b' --output_dir='traces/opt-2.7b' --context_len=1024 --skip_functional_sim
echo "Generating OPT-30B traces"
python3 examples/opt_runner.py --model='opt-30b' --output_dir='traces/opt-30b' --context_len=1024 --skip_functional_sim

echo "Done generating traces"