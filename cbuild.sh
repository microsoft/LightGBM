#!/bin/bash

if cmake -B build -S . -DUSE_CUDA=1; then
    if cmake --build build -j4; then
        echo "build complete"
    else
        echo "cmake --build build -j4 failed"
        exit 1
    fi
else
    echo "CMake configuration failed"
    exit 1
fi

cd examples/binary_classification
"../../lightgbm" config=train.conf > train.output
"../../lightgbm" config=predict.conf > predict.output