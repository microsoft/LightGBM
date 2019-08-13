# Using LightGBM via Docker

This directory contains `Dockerfile`s to make it easy to build and run LightGBM via [Docker](https://www.docker.com/).

## Installing Docker

Follow the general installation instructions [on the Docker site](https://docs.docker.com/install/):

* [macOS](https://docs.docker.com/docker-for-mac/install/)
* [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [Windows](https://docs.docker.com/docker-for-windows/install/)

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

In the above example, three volumes are [mounted](https://docs.docker.com/engine/reference/commandline/run/#mount-volume--v---read-only)
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

`config=lgbm.conf` is a command-line argument passed to the `lightgbm` executable, more arguments can be passed if required.

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

## Running the R-package Сontainer

Build the container based on the [`verse` Rocker image](https://www.rocker-project.org/images/), for R users:

```
mkdir lightgbm-docker
cd lightgbm-docker
wget https://raw.githubusercontent.com/Microsoft/LightGBM/master/docker/dockerfile-r
docker build -t lightgbm-r -f dockerfile-r .
```

After the build is finished you have two options to run the container:

1. Start [RStudio](https://www.rstudio.com/products/rstudio/), an interactive development environment, so that you can develop your analysis using LightGBM or simply try out the R package. You can open RStudio in your web browser.
2. Start a regular R session.

In both cases you can simply call

```
library("lightgbm")
```

to load the installed LightGBM R package.

**RStudio**

```
docker run --rm -it -e PASSWORD=lightgbm -p 8787:8787 lightgbm-r
```

Open the browser at http://localhost:8787 and log in.
See the [`rocker/rstudio`](https://hub.docker.com/r/rocker/rstudio) image documentation for further configuration options.

**Regular R**

If you just want a vanilla R process, change the executable of the container:

```
docker run --rm -it lightgbm-r R
```
