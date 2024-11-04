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
rm train?.output train.output
rm model?.txt

"../lightgbm" config=train.conf output_model=model1.txt > train.output
# python3 calcAccuracy.py >> Accuracy.txt

cd ..
cd examples/binary_classification || exit
"../../lightgbm" config=train.conf output_model=model1.txt > train.output
cd ../..
# python3 plot_model.py
