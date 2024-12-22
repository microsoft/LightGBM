#!/bin/bash

export SKBUILD_LOGGING_LEVEL="INFO"

pip --version
pip install pytest numpy pandas scipy scikit-learn psutil cloudpickle
