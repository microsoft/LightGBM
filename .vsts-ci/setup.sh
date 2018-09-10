#!/bin/bash

sudo apt-get update
if [[ $TASK == "mpi" ]]; then
    sudo apt-get install -y libopenmpi-dev openmpi-bin
fi
if [[ $TASK == "gpu" ]]; then
    sudo apt-get install --no-install-recommends -y ocl-icd-opencl-dev libboost-dev libboost-system-dev libboost-filesystem-dev
    cd $AGENT_HOMEDIRECTORY
    wget https://github.com/Microsoft/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.136-GA-linux64.tar.bz2
    tar -xjf AMD-APP-SDK*.tar.bz2
    mkdir -p $OPENCL_VENDOR_PATH
    sh AMD-APP-SDK*.sh --tar -xf -C $AMDAPPSDK
    mv $AMDAPPSDK/lib/x86_64/sdk/* $AMDAPPSDK/lib/x86_64/
    echo libamdocl64.so > $OPENCL_VENDOR_PATH/amdocl64.icd
fi
