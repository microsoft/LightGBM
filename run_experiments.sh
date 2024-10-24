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
# "../lightgbm" config=train.conf num_trees=20 > train.output
END=40
for i in $(seq 5 5 $END); do 
    if "../lightgbm" config=train.conf num_trees=$i output_model=results/model.$i.trees.txt > results/train.$i.trees.out; then
        echo "Training model $i complete"
    else
        echo "Training model $i failed"
        exit 1  
    fi
done
for i in $(seq 2 2 10); do 
    if "../lightgbm" config=train.conf num_trees=10 max_depth=$i output_model=results/model.$i.depth.txt > results/train.$i.depth.out; then
        echo "Training model $i complete"
    else
        echo "Training model $i failed"
        exit 1  
    fi
done
echo "Training complete"

if "../lightgbm" config=predict.conf > predict.output; then
    echo "Prediction complete"
else
    echo "Prediction failed"
    exit 1
fi

cd python || exit
python3 evaluate_models.py