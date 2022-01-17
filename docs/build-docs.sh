#!/bin/bash

set -e -u -o pipefail

rm -f ./_FIRST_RUN.flag

ARCH=$(uname -m)
export PATH="${CONDA}/bin:${PATH}"

curl \
    -L \
    -o ${HOME}/conda.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-${ARCH}.sh"

bash ${HOME}/conda.sh -b -p ${CONDA}

conda config --set always_yes yes --set changeps1 no

mamba env create \
    --name docs-env \
    --file env.yml

source activate docs-env

${CONDA}/envs/docs-env/bin/python \
    -m sphinx \
    -T \
    -E \
    -W \
    --keep-going \
    -b html \
    -d _build/doctrees \
    -D language=en \
    . _build/html

echo "Done builing docs. Open docs/_build/html/index.html in a web browser to view them."
