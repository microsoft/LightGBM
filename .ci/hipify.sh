#!/bin/bash
# Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.

for DIR in ./src ./include
do
    for EXT in cpp h hpp cu
    do
        find ${DIR} -name "*.${EXT}" -exec sh -c '
          echo "hipifying $1 in-place"
           hipify-perl "$1" -inplace &
        ' sh {} \;
    done
done

echo "waiting for all hipify-perl invocations to finish"
wait
