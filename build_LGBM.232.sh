#!/usr/bin/bash
rm -rf build
mkdir build
cd build
cmake -DUSE_CUDA=1 ..
make -j40
