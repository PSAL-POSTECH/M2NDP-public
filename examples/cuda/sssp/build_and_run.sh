#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd $SCRIPT_DIR

SOURCE=0
START=0
SIZE=3

make

mkdir raw
LD_PRELOAD=/home/shared/MM-NDP/accelsim-ramulator/util/tracer_nvbit/tracer_tool/tracer_tool.so ./sssp ../../data/USA-road-d.NY.gr 0 $SOURCE $START $SIZE
popd
