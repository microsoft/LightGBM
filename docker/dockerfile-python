FROM ubuntu:16.04

ARG CONDA_DIR=/opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        gcc \
        g++ \
        git \
        wget && \

# python-package
    # miniconda
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    export PATH="$CONDA_DIR/bin:$PATH" && \
    conda config --set always_yes yes --set changeps1 no && \
    # lightgbm
    conda install -q -y numpy scipy scikit-learn pandas && \
    git clone --recursive --branch stable --depth 1 https://github.com/Microsoft/LightGBM && \
    cd LightGBM/python-package && python setup.py install && \

# clean
    apt-get autoremove -y && apt-get clean && \
    conda clean -a -y && \
    rm -rf /usr/local/src/*
