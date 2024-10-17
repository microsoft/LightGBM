#!/bin/bash
g++ -o genModel parseModel.cpp 
./genModel
g++ -o genPred main.cpp 
./genPred
