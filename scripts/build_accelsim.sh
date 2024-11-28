#!/bin/bash

mkdir build
cd build
conan install ..
CXXFLAGS="-Wall -O3 -g3 -fPIC -std=c++11"
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=$CC \
	      -DCMAKE_CXX_COMPILER:FILEPATH=$CXX -DACCELSIM_BUILD=1 -DCMAKE_CXX_FLAGS=$(CXXFLAGS) -S../ -B./ -G "Unix Makefiles"
cmake --build . --config Debug --target all -j 34 --
