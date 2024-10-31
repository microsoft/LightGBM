# #!/bin/bash

if cmake -B build -S . -A x64 -DUSE_CUDA=0; then
    if cmake --build build --target ALL_BUILD --config Release; then
        echo "build complete"
    else
        echo "cmake build failed"
        exit 1
    fi
else
    echo "CMake configuration failed"
    exit 1
fi

cd experiments || exit
mkdir -p results 
"../Release/lightgbm" config=train.conf max_depth=3 num_trees=50 tinygbdt_forestsize=64000 > train.output
END=300
for i in $(seq 5 15 $END); do 
    if "../Release/lightgbm" config=train.conf max_depth=5 num_iterations=$i output_model=results/model.$i.num_iterations.txt > results/train.$i.num_iterations.out; then
        echo "Training model $i complete"
    else
        echo "Training model $i failed / not complete"
        # exit 1  
    fi
    # sleep 5
done
for i in $(seq 1 2 15); do 
    if "../Release/lightgbm" config=train.conf max_depth=$i num_trees=50 output_model=results/model.$i.max_depth.txt > results/train.$i.max_depth.out; then
        echo "Training model $i complete"
    else
        echo "Training model $i failed"
        # exit 1  
    fi
done
for i in 8000 16000 32000 64000 128000 256000 512000 1024000; do 
    if "../Release/lightgbm" config=train.conf max_depth=3 num_trees=50 tinygbdt_forestsize=$i output_model=results/model.$i.tinygbdt_forestsize.txt > results/train.$i.tinygbdt_forestsize.out; then
        echo "Training model $i complete"
    else
        echo "Training model $i failed / not complete"
        # exit 1  
    fi
    # sleep 1
done
echo "Training complete"

# if "../Release/lightgbm" config=predict.conf > predict.output; then
#     echo "Prediction complete"
# else
#     echo "Prediction failed"
#     exit 1
# fi

cd python || exit
python3 evaluate_models.py