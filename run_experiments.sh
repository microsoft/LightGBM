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
if "../lightgbm" config=train.conf > train.output; then
    echo "Training complete"
else
    echo "Training failed"
    exit 1
fi
# echo "Training complete"
if "../lightgbm" config=predict.conf > predict.output; then
    echo "Prediction complete"
else
    echo "Prediction failed"
    exit 1
fi

# cd ../..
# python3 plot_model.py
