# Using LightGBM via Docker

This directory contains `Dockerfile` to make it easy to build and run LightGBM via [Docker](http://www.docker.com/).

## Installing Docker

Follow the general installation instructions
[on the Docker site](https://docs.docker.com/installation/):

* [macOS](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [Ubuntu](https://docs.docker.com/installation/ubuntulinux/)

## Using CLI Version of LightGBM via Docker

Build a Docker image with LightGBM CLI:

```
mkdir lightgbm-docker
cd lightgbm-docker
wget https://raw.githubusercontent.com/Microsoft/LightGBM/master/docker/dockerfile-cli
docker build -t lightgbm-cli -f dockerfile-cli .
```

where `lightgbm-cli` is the desired Docker image name.

Run the CLI from the container:

```
docker run --rm -it \
--volume $HOME/lgbm.conf:/lgbm.conf \
--volume $HOME/model.txt:/model.txt \
--volume $HOME/tmp:/out \
lightgbm-cli \
config=lgbm.conf
```

In the above example, three volumes are [mounted](https://docs.docker.com/engine/reference/commandline/run/#mount-volume--v-read-only)
from the host machine to the Docker container:

* `lgbm.conf` - task config, for example

```
app=multiclass
num_class=3
task=convert_model
input_model=model.txt
convert_model=/out/predict.cpp
convert_model_language=cpp
```

* `model.txt` - an input file for the task, could be training data or, in this case, a pre-trained model.
* `out` - a directory to store the output of the task, notice that `convert_model` in the task config is using it.

`config=lgbm.conf` is a command-line argument passed to the `lightgbm` executable, more arguments can
be passed if required.

## Running the Python-package Сontainer

Build the container, for Python users:

```
mkdir lightgbm-docker
cd lightgbm-docker
wget https://raw.githubusercontent.com/Microsoft/LightGBM/master/docker/dockerfile-python
docker build -t lightgbm -f dockerfile-python .
```

After build finished, run the container:

```
docker run --rm -it lightgbm
```
