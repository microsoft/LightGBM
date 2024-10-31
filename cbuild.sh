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
"../lightgbm" config=train.conf output_model=model2.txt > train1.output
"../lightgbm" config=train.conf output_model=model3.txt > train2.output
"../lightgbm" config=train.conf output_model=model4.txt > train3.output
"../lightgbm" config=train.conf output_model=model5.txt > train4.output
"../lightgbm" config=train.conf output_model=model6.txt > train5.output
"../lightgbm" config=train.conf output_model=model7.txt > train6.output
"../lightgbm" config=train.conf output_model=model8.txt > train7.output
# python3 calcAccuracy.py >> Accuracy.txt

cd ../..
# python3 plot_model.py
