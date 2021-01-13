FROM nonsense/dask-lgb-test-base:123

COPY . /home/jovyan/testing

# RUN apt install -y openmpi-bin libopenmpi-dev
RUN apt-get install -y lsof net-tools

RUN cd /home/jovyan/testing/python-package && \
    python setup.py sdist && \
    # pip install dist/lightgbm*.tar.gz --install-option=--mpi && \
    pip install dist/lightgbm*.tar.gz && \
    conda clean --all

WORKDIR /home/jovyan/testing

