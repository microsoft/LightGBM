LightGBM Python Package
=======================

|License| |Python Versions| |PyPI Version|

Installation
------------

Preparation
'''''''''''

`setuptools <https://pypi.python.org/pypi/setuptools>`_ is needed.

For macOS users, gcc with OpenMP support must be installed first. Refer to `Installation Guide <https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#macos>`__ for installing gcc with OpenMP support.

Note: 32-bit Python is not supported. Please install 64-bit version.

Install from `PyPI <https://pypi.python.org/pypi/lightgbm>`_ Using ``pip``
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
For Windows users, `VC runtime <https://go.microsoft.com/fwlink/?LinkId=746572>`_ is needed if Visual Studio (2015 or 2017) is not installed.

Install `wheel <http://pythonwheels.com>`_ via ``pip install wheel`` first. After that download the wheel file and install from it:

.. code:: sh

    pip install lightgbm

Build from Sources
******************

.. code:: sh

    pip install --no-binary :all: lightgbm

For Linux and macOS users, installation from sources requires installed `CMake <https://cmake.org/>`_.

For macOS users, you need to specify compilers by runnig ``export CXX=g++-7 CC=gcc-7`` first.

For Windows users, Visual Studio (or `MS Build <https://www.visualstudio.com/downloads/>`_) is needed. If you get any errors during installation, you may need to install `CMake <https://cmake.org/>`_ (version 3.8 or higher).

Build MPI Version
~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --install-option=--mpi

For Windows users, compilation with MinGW-w64 is not supported and `CMake <https://cmake.org/>`_ (version 3.8 or higher) is strongly required in this case.

Note: MPI libraries are needed: details for installation can be found in `Installation Guide <https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-mpi-version>`__.

Build GPU Version
~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --install-option=--gpu

For Windows users, `CMake <https://cmake.org/>`_ (version 3.8 or higher) is strongly required in this case.

Note: Boost and OpenCL are needed: details for installation can be found in `Installation Guide <https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-gpu-version>`__. You need to add ``OpenCL_INCLUDE_DIR`` to the environmental variable **'PATH'** and export ``BOOST_ROOT`` before installation.

Also you may pass options to CMake via pip options, like

.. code:: sh

    pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"

All available options:

- boost-root

- boost-dir

- boost-include-dir

- boost-librarydir

- opencl-include-dir

- opencl-library

For more details see `FindBoost <https://cmake.org/cmake/help/v3.8/module/FindBoost.html>`__ and `FindOpenCL <https://cmake.org/cmake/help/v3.8/module/FindOpenCL.html>`__.

Build HDFS Version
~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --install-option=--hdfs

For Windows users, `CMake <https://cmake.org/>`_ (version 3.8 or higher) is strongly required in this case.

Build with MinGW-w64 on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --install-option=--mingw

Note: `CMake <https://cmake.org/>`_ and `MinGW-w64 <https://mingw-w64.org/>`_ should be installed first.

Install from GitHub
'''''''''''''''''''

For Linux and macOS users, installation from GitHub requires installed `CMake <https://cmake.org/>`_.

For Windows users, Visual Studio (or `MS Build <https://www.visualstudio.com/downloads/>`_) is needed. If you get any errors during installation and there is the warning ``WARNING:LightGBM:Compilation with MSBuild from existing solution file failed.`` in the log, you should install `CMake <https://cmake.org/>`_ (version 3.8 or higher).

.. code:: sh

    git clone --recursive https://github.com/Microsoft/LightGBM.git
    cd LightGBM/python-package
    # export CXX=g++-7 CC=gcc-7  # for macOS users only
    python setup.py install

Note: ``sudo`` (or administrator rights in Windows) may be needed to perform the command.

Run ``python setup.py install --mpi`` to enable MPI support. For Windows users, compilation with MinGW-w64 is not supported and `CMake <https://cmake.org/>`_ (version 3.8 or higher) is strongly required in this case. MPI libraries are needed: details for installation can be found in `Installation Guide <https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-mpi-version>`__.

Run ``python setup.py install --mingw`` if you want to use MinGW-w64 on Windows instead of Visual Studio. `CMake <https://cmake.org/>`_ and `MinGW-w64 <https://mingw-w64.org/>`_ should be installed first.

Run ``python setup.py install --gpu`` to enable GPU support. For Windows users, `CMake <https://cmake.org/>`_ (version 3.8 or higher) is strongly required in this case. Boost and OpenCL are needed: details for installation can be found in `Installation Guide <https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-gpu-version>`__. You can pass additional options to CMake: ``python setup.py install --gpu --opencl-include-dir=/usr/local/cuda/include/``, see `Build GPU Version <#build-gpu-version>`__ for complete list of them.

Run ``python setup.py install --hdfs`` to enable HDFS support. For Windows users, `CMake <https://cmake.org/>`_ (version 3.8 or higher) is strongly required in this case.

If you get any errors during installation or due to any other reason, you may want to build dynamic library from sources by any method you prefer (see `Installation Guide <https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst>`__) and then run ``python setup.py install --precompile``.

Examples
--------

Refer to the walk through examples in `Python guide folder <https://github.com/Microsoft/LightGBM/tree/master/examples/python-guide>`_.

Troubleshooting
---------------

In case you are facing any errors during the installation process, you can examine ``$HOME/LightGBM_compilation.log`` file, in which all operations are logged, to get more details about occurred problem. Also, please attach this file to the issue on GitHub to help faster indicate the cause of the error.

Refer to `FAQ <https://github.com/Microsoft/LightGBM/tree/master/docs/FAQ.rst>`_.

Developments
------------

The code style of Python-package follows `pep8 <https://www.python.org/dev/peps/pep-0008/>`_. If you would like to make a contribution and not familiar with pep-8, please check the pep8 style guide first. Otherwise, the check won't pass. You should be careful about:

- E1 Indentation (check pep8 link above)
- E202 whitespace before and after brackets
- E225 missing whitespace around operator
- E226 missing whitespace around arithmetic operator
- E261 at least two spaces before inline comment
- E301 expected 1 blank line in front of and at the end of a method
- E302 expected 2 blank lines in front of and at the end of a function or a class

E501 can be ignored (line too long).

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/Microsoft/LightGBM/blob/master/LICENSE
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/lightgbm.svg
   :target: https://pypi.python.org/pypi/lightgbm
.. |PyPI Version| image:: https://img.shields.io/pypi/v/lightgbm.svg
   :target: https://pypi.python.org/pypi/lightgbm
