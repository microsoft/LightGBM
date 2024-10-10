#!/bin/bash

# input a model path as paramter to bash call, otherwise binary_classification model is used as default
input_model="${1:-./examples/binary_classification/LightGBM_model.txt}"

./lightgbm task=convert_model input_model=$input_model convert_model=ifelse_model.cpp