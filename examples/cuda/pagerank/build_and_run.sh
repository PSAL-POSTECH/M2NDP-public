#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd $SCRIPT_DIR
make
rm *.txt
./pagerank_spmv.out ../../data/coAuthorsDBLP.graph 1
popd