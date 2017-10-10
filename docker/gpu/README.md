# Dockerfile for LightGBM GPU Version with Python

A docker file with LightGBM utilizing nvidia-docker. The file is based on the nvidia/cuda:8.0 image. LightGBM can be utilized in GPU and CPU modes and via Python (2.7 & 3.5)

## Contents

- LightGBM (cpu + gpu)
- Python 2.7 (Conda) + scikit-learn notebooks pandas matplotlib
- Python 3.5 (Conda) + scikit-learn notebooks pandas matplotlib

Running the container starts a jupyter notebook at localhost:8888

jupyter password: keras

## Requirements

Requires docker and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on host machine.

## Quickstart

### Build Docker Image

```sh
mkdir lightgbm-docker
cd lightgbm-docker
wget https://github.com/Microsoft/LightGBM/blob/master/docker/gpu/dockerfile.gpu
docker build -f dockerfile.gpu -t lightgbm-gpu .
```

### Run Image

```sh
nvidia-docker run --rm -d --name lightgbm-gpu -p 8888:8888 -v /home:/home lightgbm-gpu
```

### Attach with Command Line Access (if required)

```sh
docker exec -it lightgbm-gpu bash
```

### Jupyter Notebook

```sh
localhost:8888
```
