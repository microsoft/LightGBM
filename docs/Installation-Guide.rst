Installation Guide
==================

Here is the guide for the build of CLI version.

For the build of Python-package and R-package, please refer to `Python-package`_ and `R-package`_ folders respectively.

Windows
~~~~~~~

LightGBM can use Visual Studio, MSBuild with CMake or MinGW to build in Windows.

Visual Studio (or MSBuild)
^^^^^^^^^^^^^^^^^^^^^^^^^^

With GUI
********

1. Install `Visual Studio`_ (2015 or newer).

2. Download `zip archive`_ and unzip it.

3. Go to ``LightGBM-master/windows`` folder.

4. Open ``LightGBM.sln`` file with Visual Studio, choose ``Release`` configuration and click ``BUILD``->\ ``Build Solution (Ctrl+Shift+B)``.

   If you have errors about **Platform Toolset**, go to ``PROJECT``->\ ``Properties``->\ ``Configuration Properties``->\ ``General`` and select the toolset installed on your machine.

The exe file will be in ``LightGBM-master/windows/x64/Release`` folder.

From Command Line
*****************

1. Install `Git for Windows`_, `CMake`_ (3.8 or higher) and `MSBuild`_ (**MSBuild** is not needed if **Visual Studio** (2015 or newer) is installed).

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
     cmake --build . --target ALL_BUILD --config Release

The exe and dll files will be in ``LightGBM/Release`` folder.

MinGW64
^^^^^^^

1. Install `Git for Windows`_, `CMake`_ and `MinGW-w64`_.

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -G "MinGW Makefiles" ..
     mingw32-make.exe -j4

The exe and dll files will be in ``LightGBM/`` folder.

**Note**: you may need to run the ``cmake -G "MinGW Makefiles" ..`` one more time if met ``sh.exe was found in your PATH`` error.

Also you may want to reed `gcc Tips <./gcc-Tips.rst>`__.

Linux
~~~~~

LightGBM uses **CMake** to build. Run the following commands:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  mkdir build ; cd build
  cmake ..
  make -j4

**Note**: glibc >= 2.14 is required.

Also you may want to reed `gcc Tips <./gcc-Tips.rst>`__.

macOS
~~~~~

LightGBM depends on **OpenMP** for compiling, which isn't supported by Apple Clang.

Please install **gcc/g++** by using the following commands:

.. code::

  brew install cmake
  brew install gcc

Then install LightGBM:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  export CXX=g++-7 CC=gcc-7
  mkdir build ; cd build
  cmake ..
  make -j4

Also you may want to reed `gcc Tips <./gcc-Tips.rst>`__.

Docker
~~~~~~

Refer to `Docker folder <https://github.com/Microsoft/LightGBM/tree/master/docker>`__.

Build MPI Version
~~~~~~~~~~~~~~~~~

The default build version of LightGBM is based on socket. LightGBM also supports `MPI`_.
MPI is a high performance communication approach with `RDMA`_ support.

If you need to run a parallel learning application with high performance communication, you can build the LightGBM with MPI support.

Windows
^^^^^^^

With GUI
********

1. You need to install `MS MPI`_ first. Both ``msmpisdk.msi`` and ``MSMpiSetup.exe`` are needed.

2. Install `Visual Studio`_ (2015 or newer).

3. Download `zip archive`_ and unzip it.

4. Go to ``LightGBM-master/windows`` folder.

5. Open ``LightGBM.sln`` file with Visual Studio, choose ``Release_mpi`` configuration and click ``BUILD``->\ ``Build Solution (Ctrl+Shift+B)``.

   If you have errors about **Platform Toolset**, go to ``PROJECT``->\ ``Properties``->\ ``Configuration Properties``->\ ``General`` and select the toolset installed on your machine.

The exe file will be in ``LightGBM-master/windows/x64/Release_mpi`` folder.

From Command Line
*****************

1. You need to install `MS MPI`_ first. Both ``msmpisdk.msi`` and ``MSMpiSetup.exe`` are needed.

2. Install `Git for Windows`_, `CMake`_ (3.8 or higher) and `MSBuild`_ (MSBuild is not needed if **Visual Studio** (2015 or newer) is installed).

3. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DUSE_MPI=ON ..
     cmake --build . --target ALL_BUILD --config Release

The exe and dll files will be in ``LightGBM/Release`` folder.

**Note**: Build MPI version by **MinGW** is not supported due to the miss of MPI library in it.

Linux
^^^^^

You need to install `Open MPI`_ first.

Then run the following commands:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  mkdir build ; cd build
  cmake -DUSE_MPI=ON ..
  make -j4

**Note**: glibc >= 2.14 is required.

macOS
^^^^^

Install **Open MPI** first:

.. code::

  brew install open-mpi
  brew install cmake

Then run the following commands:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  export CXX=g++-7 CC=gcc-7
  mkdir build ; cd build
  cmake -DUSE_MPI=ON ..
  make -j4

Build GPU Version
~~~~~~~~~~~~~~~~~

Linux
^^^^^

The following dependencies should be installed before compilation:

-  OpenCL 1.2 headers and libraries, which is usually provided by GPU manufacture.

   The generic OpenCL ICD packages (for example, Debian package ``cl-icd-libopencl1`` and ``cl-icd-opencl-dev``) can also be used.

-  libboost 1.56 or later (1.61 or later recommended).

   We use Boost.Compute as the interface to GPU, which is part of the Boost library since version 1.61. However, since we include the source code of Boost.Compute as a submodule, we only require the host has Boost 1.56 or later installed. We also use Boost.Align for memory allocation. Boost.Compute requires Boost.System and Boost.Filesystem to store offline kernel cache.

   The following Debian packages should provide necessary Boost libraries: ``libboost-dev``, ``libboost-system-dev``, ``libboost-filesystem-dev``.

-  CMake 3.2 or later.

To build LightGBM GPU version, run the following commands:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  mkdir build ; cd build
  cmake -DUSE_GPU=1 ..
  # if you have installed the NVIDIA OpenGL, please use following command instead
  # sudo cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -OpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
  make -j4

Windows
^^^^^^^

If you use **MinGW**, the build procedure are similar to the build in Linux. Refer to `GPU Windows Compilation <./GPU-Windows.rst>`__ to get more details.

Following procedure is for the MSVC (Microsoft Visual C++) build.

1. Install `Git for Windows`_, `CMake`_ (3.8 or higher) and `MSBuild`_ (MSBuild is not needed if **Visual Studio** (2015 or newer) is installed).

2. Install **OpenCL** for Windows. The installation depends on the brand (NVIDIA, AMD, Intel) of your GPU card.

   - For running on Intel, get `Intel SDK for OpenCL`_.

   - For running on AMD, get `AMD APP SDK`_.

   - For running on NVIDIA, get `CUDA Toolkit`_.

3. Install `Boost Binary`_.

   **Note**: match your Visual C++ version:
   
   Visual Studio 2015 -> ``msvc-14.0-64.exe``,

   Visual Studio 2017 -> ``msvc-14.1-64.exe``.

4. Run the following commands:

   .. code::

     Set BOOST_ROOT=C:\local\boost_1_64_0\
     Set BOOST_LIBRARYDIR=C:\local\boost_1_64_0\lib64-msvc-14.0
     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DUSE_GPU=1 ..
     cmake --build . --target ALL_BUILD --config Release

   **Note**: ``C:\local\boost_1_64_0\`` and ``C:\local\boost_1_64_0\lib64-msvc-14.0`` are locations of your Boost binaries. You also can set them to the environment variable to avoid ``Set ...`` commands when build.

Docker
^^^^^^

Refer to `GPU Docker folder <https://github.com/Microsoft/LightGBM/tree/master/docker/gpu>`__.

.. _Python-package: https://github.com/Microsoft/LightGBM/tree/master/python-package

.. _R-package: https://github.com/Microsoft/LightGBM/tree/master/R-package

.. _zip archive: https://github.com/Microsoft/LightGBM/archive/master.zip

.. _Visual Studio: https://www.visualstudio.com/downloads/

.. _Git for Windows: https://git-scm.com/download/win

.. _CMake: https://cmake.org/

.. _MSBuild: https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017

.. _MinGW-w64: https://mingw-w64.org/doku.php/download

.. _MPI: https://en.wikipedia.org/wiki/Message_Passing_Interface

.. _RDMA: https://en.wikipedia.org/wiki/Remote_direct_memory_access

.. _MS MPI: https://www.microsoft.com/en-us/download/details.aspx?id=49926

.. _Open MPI: https://www.open-mpi.org/

.. _Intel SDK for OpenCL: https://software.intel.com/en-us/articles/opencl-drivers

.. _AMD APP SDK: http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/

.. _CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

.. _Boost Binary: https://sourceforge.net/projects/boost/files/boost-binaries/1.64.0/
