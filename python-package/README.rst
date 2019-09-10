LightGBM Python-package
=======================

|License| |Python Versions| |PyPI Version| |Downloads|

Installation
------------

Preparation
'''''''''''

32-bit Python is not supported. Please install 64-bit version. If you have a strong need to install with 32-bit Python, refer to `Build 32-bit Version with 32-bit Python section <#build-32-bit-version-with-32-bit-python>`__.

`setuptools <https://pypi.org/project/setuptools>`_ is needed.

Install from `PyPI <https://pypi.org/project/lightgbm>`_ Using ``pip``
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

For **Windows** users, `VC runtime <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ is needed if **Visual Studio** (2015 or newer) is not installed.

For **Linux** users, **glibc** >= 2.14 is required.

For **macOS** users:

- Starting from version 2.2.1, the library file in distribution wheels is built by the **Apple Clang** (Xcode_8.3.3) compiler. This means that you don't need to install the **gcc** compiler anymore. Instead of that you need to install the **OpenMP** library, which is required for running LightGBM on the system with the **Apple Clang** compiler. You can install the **OpenMP** library by the following command: ``brew install libomp``.

- For version smaller than 2.2.1 and not smaller than 2.1.2, **gcc-8** with **OpenMP** support must be installed first. Refer to `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#gcc>`__ for installation of **gcc-8** with **OpenMP** support.

- For version smaller than 2.1.2, **gcc-7** with **OpenMP** is required.

Install `wheel <https://pythonwheels.com>`_ via ``pip install wheel`` first. After that download the wheel file and install from it:

.. code:: sh

    pip install lightgbm

Build from Sources
******************

.. code:: sh

    pip install --no-binary :all: lightgbm

For **Linux** and **macOS** users, installation from sources requires installed `CMake`_.

For **macOS** users, you can perform installation either with **Apple Clang** or **gcc**.

- In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#apple-clang>`__) first and **CMake** version 3.12 or higher is required.

  In some cases **OpenMP** cannot be found which causes installation failures. So, if you encounter errors during the installation process, try to pass paths to **CMake** via ``pip`` options, like

  .. code:: sh

      pip install lightgbm --install-option="--openmp-include-dir=/usr/local/opt/libomp/include/" --install-option="--openmp-library=/usr/local/opt/libomp/lib/libomp.dylib"

- In case you prefer **gcc**, you need to install it (details for installation can be found in `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#gcc>`__) and specify compilers by running ``export CXX=g++-7 CC=gcc-7`` (replace "7" with version of **gcc** installed on your machine) first.

For **Windows** users, **Visual Studio** (or `VS Build Tools <https://visualstudio.microsoft.com/downloads/>`_) is needed. If you get any errors during installation, you may need to install `CMake`_ (version 3.8 or higher).

Build Threadless Version
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --install-option=--nomp

All remarks, except the **OpenMP** requirement for **macOS** users, from `Build from Sources section <#build-from-sources>`__ are actual in this case.

It is **strongly not recommended** to use this version of LightGBM!

Build MPI Version
~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --install-option=--mpi

All remarks from `Build from Sources section <#build-from-sources>`__ are actual in this case.

For **Windows** users, compilation with **MinGW-w64** is not supported and `CMake`_ (version 3.8 or higher) is strongly required.

**MPI** libraries are needed: details for installation can be found in `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-mpi-version>`__.

Build GPU Version
~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --install-option=--gpu

All remarks from `Build from Sources section <#build-from-sources>`__ are actual in this case.

For **Windows** users, `CMake`_ (version 3.8 or higher) is strongly required.

**Boost** and **OpenCL** are needed: details for installation can be found in `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-gpu-version>`__. Almost always you also need to pass ``OpenCL_INCLUDE_DIR``, ``OpenCL_LIBRARY`` options for **Linux** and ``BOOST_ROOT``, ``BOOST_LIBRARYDIR`` options for **Windows** to **CMake** via ``pip`` options, like

.. code:: sh

    pip install lightgbm --install-option=--gpu --install-option="--opencl-include-dir=/usr/local/cuda/include/" --install-option="--opencl-library=/usr/local/cuda/lib64/libOpenCL.so"

All available options:

- boost-root

- boost-dir

- boost-include-dir

- boost-librarydir

- opencl-include-dir

- opencl-library

For more details see `FindBoost <https://cmake.org/cmake/help/latest/module/FindBoost.html>`__ and `FindOpenCL <https://cmake.org/cmake/help/latest/module/FindOpenCL.html>`__.

Build HDFS Version
~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --install-option=--hdfs

Note that the installation process of HDFS version was tested only on **Linux**.

Build with MinGW-w64 on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --install-option=--mingw

`CMake`_ and `MinGW-w64 <https://mingw-w64.org/>`_ should be installed first.

It is recommended to use **Visual Studio** for its better multithreading efficiency in **Windows** for many-core systems
(see `Question 4 <https://github.com/microsoft/LightGBM/blob/master/docs/FAQ.rst#4-i-am-using-windows-should-i-use-visual-studio-or-mingw-for-compiling-lightgbm>`__ and `Question 8 <https://github.com/microsoft/LightGBM/blob/master/docs/FAQ.rst#8-cpu-usage-is-low-like-10-in-windows-when-using-lightgbm-on-very-large-datasets-with-many-core-systems>`__).

Build 32-bit Version with 32-bit Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --install-option=--bit32

By default, installation in environment with 32-bit Python is prohibited. However, you can remove this prohibition on your own risk by passing ``bit32`` option.

It is **strongly not recommended** to use this version of LightGBM!

Install from GitHub
'''''''''''''''''''

All remarks from `Build from Sources section <#build-from-sources>`__ are actual in this case.

For **macOS** users who compile with **Apple Clang**, to pass **OpenMP** paths to **CMake** use the following syntax: ``python setup.py install --openmp-include-dir=/usr/local/opt/libomp/include/ --openmp-library=/usr/local/opt/libomp/lib/libomp.dylib``.

For **Windows** users, if you get any errors during installation and there is the warning ``WARNING:LightGBM:Compilation with MSBuild from existing solution file failed.`` in the log, you should install `CMake`_ (version 3.8 or higher).

.. code:: sh

    git clone --recursive https://github.com/microsoft/LightGBM.git
    cd LightGBM/python-package
    # export CXX=g++-7 CC=gcc-7  # macOS users, if you decided to compile with gcc, don't forget to specify compilers (replace "7" with version of gcc installed on your machine)
    python setup.py install

Note: ``sudo`` (or administrator rights in **Windows**) may be needed to perform the command.

Run ``python setup.py install --nomp`` to disable **OpenMP** support. All remarks from `Build Threadless Version section <#build-threadless-version>`__ are actual in this case.

Run ``python setup.py install --mpi`` to enable **MPI** support. All remarks from `Build MPI Version section <#build-mpi-version>`__ are actual in this case.

Run ``python setup.py install --mingw``, if you want to use **MinGW-w64** on **Windows** instead of **Visual Studio**. All remarks from `Build with MinGW-w64 on Windows section <#build-with-mingw-w64-on-windows>`__ are actual in this case.

Run ``python setup.py install --gpu`` to enable GPU support. All remarks from `Build GPU Version section <#build-gpu-version>`__ are actual in this case. To pass additional options to **CMake** use the following syntax: ``python setup.py install --gpu --opencl-include-dir=/usr/local/cuda/include/``, see `Build GPU Version section <#build-gpu-version>`__ for the complete list of them.

Run ``python setup.py install --hdfs`` to enable HDFS support. All remarks from `Build HDFS Version section <#build-hdfs-version>`__ are actual in this case.

Run ``python setup.py install --bit32``, if you want to use 32-bit version. All remarks from `Build 32-bit Version with 32-bit Python section <#build-32-bit-version-with-32-bit-python>`__ are actual in this case.

If you get any errors during installation or due to any other reasons, you may want to build dynamic library from sources by any method you prefer (see `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst>`__) and then just run ``python setup.py install --precompile``.

Troubleshooting
---------------

In case you are facing any errors during the installation process, you can examine ``$HOME/LightGBM_compilation.log`` file, in which all operations are logged, to get more details about occurred problem. Also, please attach this file to the issue on GitHub to help faster indicate the cause of the error.

Refer to `FAQ <https://github.com/microsoft/LightGBM/tree/master/docs/FAQ.rst>`_.

Examples
--------

Refer to the walk through examples in `Python guide folder <https://github.com/microsoft/LightGBM/tree/master/examples/python-guide>`_.

Development Guide
-----------------

The code style of Python-package follows `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_. If you would like to make a contribution and not familiar with PEP 8, please check the PEP 8 style guide first. Otherwise, the check won't pass. You should be careful about:

- E1 Indentation (check PEP 8 link above)
- E202 whitespace before and after brackets
- E225 missing whitespace around operator
- E226 missing whitespace around arithmetic operator
- E261 at least two spaces before inline comment
- E301 expected 1 blank line in front of and at the end of a method
- E302 expected 2 blank lines in front of and at the end of a function or a class

E501 (line too long) and W503 (line break occurred before a binary operator) can be ignored.

Documentation strings (docstrings) are written in the NumPy style.

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/microsoft/LightGBM/blob/master/LICENSE
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/lightgbm.svg
   :target: https://pypi.org/project/lightgbm
.. |PyPI Version| image:: https://img.shields.io/pypi/v/lightgbm.svg
   :target: https://pypi.org/project/lightgbm
.. |Downloads| image:: https://pepy.tech/badge/lightgbm
   :target: https://pepy.tech/project/lightgbm
.. _CMake: https://cmake.org/
