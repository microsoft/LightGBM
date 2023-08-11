LightGBM Python-package
=======================

|License| |Python Versions| |PyPI Version| |Downloads| |API Docs|

Installation
------------

Preparation
'''''''''''

32-bit Python is not supported. Please install 64-bit version. If you have a strong need to install with 32-bit Python, refer to `Build 32-bit Version with 32-bit Python section <#build-32-bit-version-with-32-bit-python>`__.

`setuptools <https://pypi.org/project/setuptools>`_ is needed.

Install from `PyPI <https://pypi.org/project/lightgbm>`_
''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: sh

    pip install lightgbm

Compiled library that is included in the wheel file supports both **GPU** and **CPU** versions out of the box. This feature is experimental and available only for **Windows** and **Linux** currently. To use **GPU** version you only need to install OpenCL Runtime libraries. For NVIDIA and AMD GPU they are included in the ordinary drivers for your graphics card, so no action is required. If you would like your AMD or Intel CPU to act like a GPU (for testing and debugging) you can install `AMD APP SDK <https://github.com/microsoft/LightGBM/releases/download/v2.0.12/AMD-APP-SDKInstaller-v3.0.130.135-GA-windows-F-x64.exe>`_ on **Windows** and `PoCL <http://portablecl.org>`_ on **Linux**. Many modern Linux distributions provide packages for PoCL, look for ``pocl-opencl-icd`` on Debian-based distributions and ``pocl`` on RedHat-based distributions.

For **Windows** users, `VC runtime <https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads>`_ is needed if **Visual Studio** (2015 or newer) is not installed.

For **Linux** users, **glibc** >= 2.14 is required for LightGBM ``<=3.3.3`` and **glibc** >= 2.28 is required for newer versions. Also, in some rare cases, when you hit ``OSError: libgomp.so.1: cannot open shared object file: No such file or directory`` error during importing LightGBM, you need to install OpenMP runtime library separately (use your package manager and search for ``lib[g|i]omp`` for doing this).

For **macOS** (we provide wheels for 3 newest macOS versions) users:

- Starting from version 2.2.1, the library file in distribution wheels is built by the **Apple Clang** (Xcode_8.3.3 for versions 2.2.1 - 2.3.1, Xcode_9.4.1 for versions 2.3.2 - 3.3.2 and Xcode_11.7 from version 4.0.0) compiler. This means that you don't need to install the **gcc** compiler anymore. Instead of that you need to install the **OpenMP** library, which is required for running LightGBM on the system with the **Apple Clang** compiler. You can install the **OpenMP** library by the following command: ``brew install libomp``.

- For version smaller than 2.2.1 and not smaller than 2.1.2, **gcc-8** with **OpenMP** support must be installed first. Refer to `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#gcc>`__ for installation of **gcc-8** with **OpenMP** support.

- For version smaller than 2.1.2, **gcc-7** with **OpenMP** is required.

Use LightGBM with Dask
**********************

.. warning::

    Dask-package is only tested on Linux.

To install all dependencies needed to use ``lightgbm.dask``, append ``[dask]``.

.. code:: sh

    pip install 'lightgbm[dask]'

Use LightGBM with pandas
************************

To install all dependencies needed to use ``pandas`` in LightGBM, append ``[pandas]``.

.. code:: sh

    pip install 'lightgbm[pandas]'

Use LightGBM with scikit-learn
******************************

To install all dependencies needed to use ``scikit-learn`` in LightGBM, append ``[scikit-learn]``.

.. code:: sh

    pip install 'lightgbm[scikit-learn]'

Build from Sources
******************

.. code:: sh

    pip install --no-binary lightgbm lightgbm

For **Linux** and **macOS** users, installation from sources requires installed `CMake`_.

For **Linux** users, **glibc** >= 2.28 is required. Also, in some rare cases you may need to install OpenMP runtime library separately (use your package manager and search for ``lib[g|i]omp`` for doing this).

For **macOS** users, you can perform installation either with **Apple Clang** or **gcc**.

- In case you prefer **Apple Clang**, you should install **OpenMP** (details for installation can be found in `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#apple-clang>`__) first and **CMake** version 3.16 or higher is required.

- In case you prefer **gcc**, you need to install it (details for installation can be found in `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#gcc>`__) and specify compilers by running ``export CXX=g++-7 CC=gcc-7`` (replace "7" with version of **gcc** installed on your machine) first.

For **Windows** users, **Visual Studio** (or `VS Build Tools <https://visualstudio.microsoft.com/downloads/>`_) is needed. If you get any errors during installation, you may need to install `CMake`_ (version 3.8 or higher).

Build Threadless Version
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --config-settings=cmake.define.USE_OPENMP=OFF

All requirements, except the **OpenMP** requirement, from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well.

It is **strongly not recommended** to use this version of LightGBM!

Build MPI Version
~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --config-settings=cmake.define.USE_MPI=ON

All requirements from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well.

For **Windows** users, compilation with **MinGW-w64** is not supported and `CMake`_ (version 3.8 or higher) is strongly required.

**MPI** libraries are needed: details for installation can be found in `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-mpi-version>`__.

Build GPU Version
~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --config-settings=cmake.define.USE_GPU=ON

All requirements from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well.

For **Windows** users, `CMake`_ (version 3.8 or higher) is strongly required.

**Boost** and **OpenCL** are needed: details for installation can be found in `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-gpu-version>`__. Almost always you also need to pass ``OpenCL_INCLUDE_DIR``, ``OpenCL_LIBRARY`` options for **Linux** and ``BOOST_ROOT``, ``BOOST_LIBRARYDIR`` options for **Windows** to **CMake** via ``pip`` options, like

.. code:: sh

    pip install lightgbm \
      --config-settings=cmake.define.USE_GPU=ON \
      --config-settings=cmake.define.OpenCL_INCLUDE_DIR="/usr/local/cuda/include/" \
      --config-settings=cmake.define.OpenCL_LIBRARY="/usr/local/cuda/lib64/libOpenCL.so"

All available options that can be passed via ``cmake.define.{option}``.

- Boost_ROOT

- Boost_DIR

- Boost_INCLUDE_DIR

- BOOST_LIBRARYDIR

- OpenCL_INCLUDE_DIR

- OpenCL_LIBRARY

For more details see `FindBoost <https://cmake.org/cmake/help/latest/module/FindBoost.html>`__ and `FindOpenCL <https://cmake.org/cmake/help/latest/module/FindOpenCL.html>`__.

Build CUDA Version
~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON

All requirements from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well, and `CMake`_ (version 3.16 or higher) is strongly required.

**CUDA** library (version 10.0 or higher) is needed: details for installation can be found in `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-cuda-version-experimental>`__.

To use the CUDA version within Python, pass ``{"device": "cuda"}`` respectively in parameters.

Build HDFS Version
~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --config-settings=cmake.define.USE_HDFS=ON

All requirements from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well.

**HDFS** library is needed: details for installation can be found in `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#build-hdfs-version>`__.

Note that the installation process of HDFS version was tested only on **Linux**.

Build with MinGW-w64 on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    # in sh.exe, git bash, or other Unix-like shell
    export CMAKE_GENERATOR='MinGW Makefiles'
    pip install lightgbm --config-settings=cmake.define.CMAKE_SH=CMAKE_SH-NOTFOUND

`CMake`_ and `MinGW-w64 <https://www.mingw-w64.org/>`_ should be installed first.

It is recommended to use **Visual Studio** for its better multithreading efficiency in **Windows** for many-core systems
(see `Question 4 <https://github.com/microsoft/LightGBM/blob/master/docs/FAQ.rst#4-i-am-using-windows-should-i-use-visual-studio-or-mingw-for-compiling-lightgbm>`__ and `Question 8 <https://github.com/microsoft/LightGBM/blob/master/docs/FAQ.rst#8-cpu-usage-is-low-like-10-in-windows-when-using-lightgbm-on-very-large-datasets-with-many-core-systems>`__).

Build 32-bit Version with 32-bit Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    # in sh.exe, git bash, or other Unix-like shell
    export CMAKE_GENERATOR='Visual Studio 17 2022'
    export CMAKE_GENERATOR_PLATFORM='Win32'
    pip install --no-binary lightgbm lightgbm

By default, installation in environment with 32-bit Python is prohibited. However, you can remove this prohibition on your own risk by passing ``bit32`` option.

It is **strongly not recommended** to use this version of LightGBM!

Build with Time Costs Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

    pip install lightgbm --config-settings=cmake.define.USE_TIMETAG=ON

Use this option to make LightGBM output time costs for different internal routines, to investigate and benchmark its performance.

Install from `conda-forge channel <https://anaconda.org/conda-forge/lightgbm>`_
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

If you use ``conda`` to manage Python dependencies, you can install LightGBM using ``conda install``.

We strongly recommend installation from the ``conda-forge`` channel and not from the ``default`` one due to many reasons. The main ones are less time delay for new releases, greater number of supported architectures and better handling of dependency conflicts, especially workaround for OpenMP is crucial for LightGBM. More details can be found in `this comment <https://github.com/microsoft/LightGBM/issues/4948#issuecomment-1013766397>`_.

**Note**: The `lightgbm conda-forge feedstock <https://github.com/conda-forge/lightgbm-feedstock>`_ is not maintained by LightGBM maintainers.

.. code:: sh

    conda install -c conda-forge lightgbm

Install from GitHub
'''''''''''''''''''

All requirements from `Build from Sources section <#build-from-sources>`__ apply for this installation option as well.

For **Windows** users, if you get any errors during installation and there is the warning ``WARNING:LightGBM:Compilation with MSBuild from existing solution file failed.`` in the log, you should install `CMake`_ (version 3.8 or higher).

.. code:: sh

    git clone --recursive https://github.com/microsoft/LightGBM.git
    # export CXX=g++-7 CC=gcc-7  # macOS users, if you decided to compile with gcc, don't forget to specify compilers (replace "7" with version of gcc installed on your machine)
    sh ./build-python.sh install

Note: ``sudo`` (or administrator rights in **Windows**) may be needed to perform the command.

Run ``sh ./build-python.sh install --nomp`` to disable **OpenMP** support. All requirements from `Build Threadless Version section <#build-threadless-version>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --mpi`` to enable **MPI** support. All requirements from `Build MPI Version section <#build-mpi-version>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --mingw``, if you want to use **MinGW-w64** on **Windows** instead of **Visual Studio**. All requirements from `Build with MinGW-w64 on Windows section <#build-with-mingw-w64-on-windows>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --gpu`` to enable GPU support. All requirements from `Build GPU Version section <#build-gpu-version>`__ apply for this installation option as well. To pass additional options to **CMake** use the following syntax: ``sh ./build-python.sh install --gpu --opencl-include-dir="/usr/local/cuda/include/"``, see `Build GPU Version section <#build-gpu-version>`__ for the complete list of them.

Run ``sh ./build-python.sh install --cuda`` to enable CUDA support. All requirements from `Build CUDA Version section <#build-cuda-version>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --hdfs`` to enable HDFS support. All requirements from `Build HDFS Version section <#build-hdfs-version>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --bit32``, if you want to use 32-bit version. All requirements from `Build 32-bit Version with 32-bit Python section <#build-32-bit-version-with-32-bit-python>`__ apply for this installation option as well.

Run ``sh ./build-python.sh install --time-costs``, if you want to output time costs for different internal routines. All requirements from `Build with Time Costs Output section <#build-with-time-costs-output>`__ apply for this installation option as well.

If you get any errors during installation or due to any other reasons, you may want to build dynamic library from sources by any method you prefer (see `Installation Guide <https://github.com/microsoft/LightGBM/blob/master/docs/Installation-Guide.rst>`__) and then just run ``sh ./build-python.sh install --precompile``.

Build Wheel File
****************

You can use ``sh ./build-python.sh install bdist_wheel`` instead of ``sh ./build-python.sh install`` to build wheel file and use it for installation later. This might be useful for systems with restricted or completely without network access.

Build With MSBuild
******************

To use ``MSBuild`` (Windows-only), first build ``lib_lightgbm.dll`` by running the following from the root of the repo.

.. code:: sh

  MSBuild.exe windows/LightGBM.sln /p:Configuration=DLL /p:Platform=x64 /p:PlatformToolset=v143

Then install the Python package using that library.

.. code:: sh

  sh ./build-python.sh install --precompile

Troubleshooting
---------------

In case you are facing any errors during the installation process, you can examine ``$HOME/LightGBM_compilation.log`` file, in which all operations are logged, to get more details about occurred problem. Also, please attach this file to the issue on GitHub to help faster indicate the cause of the error.

Refer to `FAQ <https://github.com/microsoft/LightGBM/tree/master/docs/FAQ.rst>`_.

Examples
--------

Refer to the walk through examples in `Python guide folder <https://github.com/microsoft/LightGBM/tree/master/examples/python-guide>`_.

Development Guide
-----------------

The code style of Python-package follows `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.

The package's documentation strings (docstrings) are written in the `numpydoc style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

To check that a contribution to the package matches its style expectations, run the following from the root of the repo.

.. code:: sh

    sh .ci/lint-python.sh

.. |License| image:: https://img.shields.io/github/license/microsoft/lightgbm.svg
   :target: https://github.com/microsoft/LightGBM/blob/master/LICENSE
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/lightgbm.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/lightgbm
.. |PyPI Version| image:: https://img.shields.io/pypi/v/lightgbm.svg?logo=pypi&logoColor=white
   :target: https://pypi.org/project/lightgbm
.. |Downloads| image:: https://pepy.tech/badge/lightgbm
   :target: https://pepy.tech/project/lightgbm
.. |API Docs| image:: https://readthedocs.org/projects/lightgbm/badge/?version=latest
   :target: https://lightgbm.readthedocs.io/en/latest/Python-API.html
.. _CMake: https://cmake.org/
