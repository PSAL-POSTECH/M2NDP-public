#!/bin/bash

mkdir roofline
for j in 2 4 8
do
    export NDP_UNROLL_LEVEL=$j
    for i in 1 2 4 8 16 32 64
    do
        export NDP_INTENSITY=$i
        python3 runner.py --kernel UnrolledVectorAddKernel > /dev/null && cd .. && \
            ./build/bin/NDPSim --trace examples/tmp --num_hosts 1 --num_m2ndps 1 \
            --config ./config/performance/M2NDP/m2ndp.config --output vectoradd${i}_${j}.log > /dev/null && cd examples
    done
done