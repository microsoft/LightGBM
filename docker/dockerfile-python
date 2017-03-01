FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y cmake build-essential gcc g++ git wget && \

# open-mpi
    cd /usr/local/src && mkdir openmpi && cd openmpi && \
    wget https://www.open-mpi.org/software/ompi/v2.0/downloads/openmpi-2.0.1.tar.gz && \
    tar -xzf openmpi-2.0.1.tar.gz && cd openmpi-2.0.1 && \
    ./configure --prefix=/usr/local/openmpi && make && make install && \
    export PATH="/usr/local/openmpi/bin:$PATH" && \

# lightgbm
    cd /usr/local/src && mkdir lightgbm && cd lightgbm && \
    git clone --recursive https://github.com/Microsoft/LightGBM && \
    cd LightGBM && mkdir build && cd build && cmake -DUSE_MPI=ON .. && make && \

# python-package
    # miniconda
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p /opt/conda && \
    export PATH="/opt/conda/bin:$PATH" && \
    # lightgbm
    conda install -y numpy scipy scikit-learn pandas && \
    cd ../python-package && python setup.py install && \

# clean
    apt-get autoremove -y && apt-get clean && \
    conda clean -i -l -t -y && \
    rm -rf /usr/local/src/*

ENV PATH /opt/conda/bin:$PATH
