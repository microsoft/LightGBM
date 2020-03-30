#!/usr/bin/bash
rm -rf build
mkdir build
cd build
#cmake -DUSE_CUDA=1 ..
cmake ..
make -j40
