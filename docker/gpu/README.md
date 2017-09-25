# Dockerfile for LightGBM supporting GPU with Python
A docker file with lightgbm utilizing nvidia-docker. The file is based on the nvidia/cuda:8.0 image. lightgbm can be utilized in gpu and cpu modes and via python (2.7 & 3.5)
### Contents
- LightGBM (cpu + gpu)
- Python 2.7 (Conda) + scikit-learn notebooks pandas matplotlib
- Python 3.5 (Conda) + scikit-learn notebooks pandas matplotlib

Running the container starts a jupyter notebook at localhost:8888

jupyter password: keras
### Requirements
Requires docker and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on host machine.
### Quickstart

##### Build Docker Image
```sh
mkdir lightgbm-docker
cd lightgbm-docker
git clone --recursive https://github.com/nji-syd/lightgbm-docker
cd lightgbm-docker
docker build -f dockerfile.gpu -t lightgbm-gpu .
```
##### Run Image
```sh
nvidia-docker run --rm -d --name lightgbm-gpu -p 8888:8888 -v /home:/home lightgbm-gpu
```

##### Attach with Command Line Access (if required)
```sh
docker exec -it lightgbm-gpu bash
```
##### Jupyter Notebook
```sh
localhost:8888
```

