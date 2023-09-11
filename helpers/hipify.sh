#!/bin/bash
# Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.

for DIR in ./src ./include
do
    for EXT in cpp h hpp cu
    do
        for FILE in $(find ${DIR} -name *.${EXT})
        do
           echo "hipifying $FILE in-place"
           hipify-perl $FILE -inplace &
        done
    done
done

echo "waiting for all hipify-perl invocations to finish"
wait
