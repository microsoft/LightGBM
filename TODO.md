All of the follow commands are currently supported, and we should try to preserve as many of them as possible.

* `{command}` is `python setup.py` right now
* `{dist}` is either the name `lightgbm` (pull from PyPI) or a path to a `*.tar.gz` or `*.whl`

Installing from source tree.

```shell
{command} install
{command} install --precompile
{command} install --precompile --user
{command} install --mingw
{command} install --mpi
{command} install --nomp
{command} install --gpu
{command} install --gpu --opencl-include-dir=/usr/local/cuda/include/
{command} install --cuda
{command} install --hdfs
{command} install --bit32
```

Building an sdist.

```shell
{command} sdist
{command} sdist --formats gztar
```

Building wheels.

```shell
{command} bdist_wheel --plat-name=macosx --python-tag py3
{command} bdist_wheel --integrated-opencl --plat-name=$PLATFORM --python-tag py3
{command} bdist_wheel --integrated-opencl --plat-name=win-amd64 --python-tag py3
{command} bdist_wheel --gpu
{command} bdist_wheel --cuda
{command} bdist_wheel --mpi
```

Customized installation of an sdist (possibly from PyPI).

```shell
pip install {dist} --install-option=--nomp
pip install {dist} --install-option=--mpi
pip install {dist} --install-option=--gpu
pip install {dist} --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"
pip install {dist} --install-option=--cuda
pip install {dist} --install-option=--hdfs
pip install {dist} --install-option=--mingw
pip install {dist} --install-option=--bit32
pip install {dist} --install-option=--time-costs
pip install {dist}[dask]
pip install --no-binary `:all:` {dist}
pip install --user {dist}
```

## Other stuff

Other features supported by `setup.py`.

* redirecting CMake logs to a file ([link](https://github.com/microsoft/LightGBM/blob/f975d3fafcdbb3739dbe4eac40dc2b7e1e3244d7/python-package/setup.py#L95-L96))
* testing with `MSBuild` and then multiple versions of MSVC ([link](https://github.com/microsoft/LightGBM/blob/f975d3fafcdbb3739dbe4eac40dc2b7e1e3244d7/python-package/setup.py#L168-L204))
    - looks like `scikit-build` will check for multiple versions of MSVC: https://github.com/scikit-build/scikit-build/blob/master/docs/generators.rst
* raising an informative warning compiling the 32-bit version ([link](https://github.com/microsoft/LightGBM/blob/f975d3fafcdbb3739dbe4eac40dc2b7e1e3244d7/python-package/setup.py#L250-L251))
