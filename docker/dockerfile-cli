FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y cmake build-essential gcc g++ git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --recursive --branch stable https://github.com/Microsoft/LightGBM && \
    mkdir LightGBM/build && \
    cd LightGBM/build && \
    cmake .. && \
    make -j4 && \
    make install && \
    cd ../.. && \
    rm -rf LightGBM 

ENTRYPOINT ["lightgbm"]
