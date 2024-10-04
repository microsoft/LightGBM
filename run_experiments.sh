#!/bin/bash

if cmake -B build -S . -DUSE_CUDA=1 -DUSE_DEBUG=ON; then
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
echo "Training complete"
"../lightgbm" config=predict.conf > predict.output
echo "Prediction complete"

# cd ../..
# python3 plot_model.py
