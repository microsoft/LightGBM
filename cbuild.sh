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
rm -rf Model
rm -rf Output
mkdir Model
mkdir Output
# $(seq 5 5 40)
for i in 2; do
  for j in 3; do
    "../lightgbm" config=train.conf num_trees=$i max_depth=$j output_model=Model/model_trees${i}_depth${j}.txt > Output/train_trees${i}_depth${j}.output
  done
done
# python3 calcAccuracy.py >> Accuracy.txt

cd ..
#cd examples/binary_classification || exit
#"../../lightgbm" config=train.conf output_model=model1.txt > train.output
#cd ../..
# python3 plot_model.py
