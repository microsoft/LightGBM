# Using LightGBM via Docker

This directory contains `Dockerfile` to make it easy to build and run LightGBM via [Docker](http://www.docker.com/).

## Installing Docker

Follow the general installation instructions
[on the Docker site](https://docs.docker.com/installation/):

* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)

## Running the container

Build the container, for python users: 

    $ docker build -t lightgbm -f dockerfile-python .

After build finished, run the container:

    $ docker run --rm -it lightgbm
