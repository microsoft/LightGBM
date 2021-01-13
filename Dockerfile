FROM nonsense/dask-lgb-test-base:123

COPY . /home/jovyan/testing

RUN cd /home/jovyan/testing/python-package && \
    python setup.py sdist && \
    pip install dist/lightgbm*.tar.gz && \
    conda clean --all

WORKDIR /home/jovyan/testing
