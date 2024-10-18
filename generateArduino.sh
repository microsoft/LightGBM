#!/bin/bash

model_string="$1"
output_string="$2"

# Check if the sufficient arguments are passed
if [ $# -lt 2 ]; then
    echo "Usage: $0 model_string output_string"
    exit 1
fi

g++ -o genModel parseModel.cpp
./genModel ${model_string} ${output_string}
cp ${output_string} ./
g++ -o genPred main.cpp 
./genPred
