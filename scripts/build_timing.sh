#!/bin/bash

# for GPU
mkdir build
cd build
conan install ..
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_C_COMPILER:FILEPATH=$CC \
	      -DCMAKE_CXX_COMPILER:FILEPATH=$CXX -DPERFORMANCE_BUILD=1 -S../ -B./ -G "Unix Makefiles"
cmake --build . --config Release --target all -j 34 --
cd -

# for CPU
mkdir build2
cd build2
conan install ..
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_C_COMPILER:FILEPATH=$CC \
	      -DCMAKE_CXX_COMPILER:FILEPATH=$CXX -DMEM_ACCESS_SIZE=64 -DPERFORMANCE_BUILD=1 -S../ -B./ -G "Unix Makefiles"
cmake --build . --config Release --target all -j 34 --