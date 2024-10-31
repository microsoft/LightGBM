#!/bin/bash

if cmake -B build -S . -DUSE_CUDA=0 -DUSE_DEBUG=ON; then
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

cd experiments || exit
"../lightgbm" config=train.conf > train.output
"../lightgbm" config=predict.conf > predict.output
# python3 calcAccuracy.py >> Accuracy.txt

cd ../..
# python3 plot_model.py
