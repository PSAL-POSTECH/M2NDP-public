#!/bin/bash

mkdir build
cd build
conan install ..
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH=$CC \
	      -DCMAKE_CXX_COMPILER:FILEPATH=$CXX -S../ -B./ -G "Unix Makefiles"
cmake --build . --config Debug --target all -j 34 --
