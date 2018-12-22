Installation Guide
==================

Here is the guide for the build of LightGBM CLI version.

For the build of Python-package and R-package, please refer to `Python-package`_ and `R-package`_ folders respectively.

Also you can download artifacts of the latest successful build in master branch: |download artifacts|.

**Contents**

-  `Windows <#windows>`__

-  `Linux <#linux>`__

-  `macOS <#macos>`__

-  `Docker <#docker>`__

- `Threadless Version (not Recommended) <#build-threadless-version-not-recommended>`__

-  `MPI Version <#build-mpi-version>`__

-  `GPU Version <#build-gpu-version>`__

-  `HDFS Version <#build-hdfs-version>`__

-  `Java Wrapper <#build-java-wrapper>`__

Windows
~~~~~~~

On Windows LightGBM can be built using

- **Visual Studio**;

- **CMake** and **VS Build Tools**;

- **CMake** and **MinGW**.

Visual Studio (or VS Build Tools)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With GUI
********

1. Install `Visual Studio`_ (2015 or newer).

2. Download `zip archive`_ and unzip it.

3. Go to ``LightGBM-master/windows`` folder.

4. Open ``LightGBM.sln`` file with **Visual Studio**, choose ``Release`` configuration and click ``BUILD`` -> ``Build Solution (Ctrl+Shift+B)``.

   If you have errors about **Platform Toolset**, go to ``PROJECT`` -> ``Properties`` -> ``Configuration Properties`` -> ``General`` and select the toolset installed on your machine.

The exe file will be in ``LightGBM-master/windows/x64/Release`` folder.

From Command Line
*****************

1. Install `Git for Windows`_, `CMake`_ (3.8 or higher) and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** (2015 or newer) is already installed).

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
     cmake --build . --target ALL_BUILD --config Release

The exe and dll files will be in ``LightGBM/Release`` folder.

MinGW-w64
^^^^^^^^^

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

**Note**: You may need to run the ``cmake -G "MinGW Makefiles" ..`` one more time if you encounter the ``sh.exe was found in your PATH`` error.

It is recommended to use **Visual Studio** for its better multithreading efficiency in **Windows** for many-core systems (see `FAQ <./FAQ.rst#lightgbm>`__ Question 4 and Question 8).

Also, you may want to read `gcc Tips <./gcc-Tips.rst>`__.

Linux
~~~~~

On Linux LightGBM can be built using **CMake** and **gcc** or **Clang**.

1. Install `CMake`_.

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     mkdir build ; cd build
     cmake ..
     make -j4

**Note**: glibc >= 2.14 is required.

Also, you may want to read `gcc Tips <./gcc-Tips.rst>`__.

macOS
~~~~~

On macOS LightGBM can be built using **CMake** and **Apple Clang** or **gcc**.

Apple Clang
^^^^^^^^^^^

Only **Apple Clang** version 8.1 or higher is supported.

1. Install `CMake`_ (3.12 or higher):

   .. code::

     brew install cmake

2. Install **OpenMP**:

   .. code::

     brew install libomp

3. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     mkdir build ; cd build
     cmake \
       -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
       -DOpenMP_C_LIB_NAMES="omp" \
       -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" \
       -DOpenMP_CXX_LIB_NAMES="omp" \
       -DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib \
       ..
     make -j4

gcc
^^^

1. Install `CMake`_ (3.2 or higher):

   .. code::

     brew install cmake

2. Install **gcc**:

   .. code::

     brew install gcc

3. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     export CXX=g++-7 CC=gcc-7  # replace "7" with version of gcc installed on your machine
     mkdir build ; cd build
     cmake ..
     make -j4

Also, you may want to read `gcc Tips <./gcc-Tips.rst>`__.

Docker
~~~~~~

Refer to `Docker folder <https://github.com/Microsoft/LightGBM/tree/master/docker>`__.

Build Threadless Version (not Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default build version of LightGBM is based on OpenMP.
However, you can build the LightGBM without OpenMP support, but it is **strongly not recommended**.

Windows
^^^^^^^

On Windows version of LightGBM without OpenMP support can be built using

- **Visual Studio**;

- **CMake** and **VS Build Tools**;

- **CMake** and **MinGW**.

Visual Studio (or VS Build Tools)
*********************************

With GUI
--------

1. Install `Visual Studio`_ (2015 or newer).

2. Download `zip archive`_ and unzip it.

3. Go to ``LightGBM-master/windows`` folder.

4. Open ``LightGBM.sln`` file with **Visual Studio**.

5. Go to ``PROJECT`` -> ``Properties`` -> ``Configuration Properties`` -> ``C/C++`` -> ``Language`` and change the ``OpenMP Support`` property to ``No (/openmp-)``.

6. Get back to the project's main screen, then choose ``Release`` configuration and click ``BUILD`` -> ``Build Solution (Ctrl+Shift+B)``.

   If you have errors about **Platform Toolset**, go to ``PROJECT`` -> ``Properties`` -> ``Configuration Properties`` -> ``General`` and select the toolset installed on your machine.

The exe file will be in ``LightGBM-master/windows/x64/Release`` folder.

From Command Line
-----------------

1. Install `Git for Windows`_, `CMake`_ (3.8 or higher) and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** (2015 or newer) is already installed).

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DUSE_OPENMP=OFF ..
     cmake --build . --target ALL_BUILD --config Release

The exe and dll files will be in ``LightGBM/Release`` folder.

MinGW-w64
*********

1. Install `Git for Windows`_, `CMake`_ and `MinGW-w64`_.

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -G "MinGW Makefiles" -DUSE_OPENMP=OFF ..
     mingw32-make.exe -j4

The exe and dll files will be in ``LightGBM/`` folder.

**Note**: You may need to run the ``cmake -G "MinGW Makefiles" -DUSE_OPENMP=OFF ..`` one more time if you encounter the ``sh.exe was found in your PATH`` error.

Linux
^^^^^

On Linux version of LightGBM without OpenMP support can be built using **CMake** and **gcc** or **Clang**.

1. Install `CMake`_.

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     mkdir build ; cd build
     cmake -DUSE_OPENMP=OFF ..
     make -j4

**Note**: glibc >= 2.14 is required.

macOS
^^^^^

On macOS version of LightGBM without OpenMP support can be built using **CMake** and **Apple Clang** or **gcc**.

Apple Clang
***********

Only **Apple Clang** version 8.1 or higher is supported.

1. Install `CMake`_ (3.12 or higher):

   .. code::

     brew install cmake

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     mkdir build ; cd build
     cmake -DUSE_OPENMP=OFF ..
     make -j4

gcc
***

1. Install `CMake`_ (3.2 or higher):

   .. code::

     brew install cmake

2. Install **gcc**:

   .. code::

     brew install gcc

3. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     export CXX=g++-7 CC=gcc-7  # replace "7" with version of gcc installed on your machine
     mkdir build ; cd build
     cmake -DUSE_OPENMP=OFF ..
     make -j4

Build MPI Version
~~~~~~~~~~~~~~~~~

The default build version of LightGBM is based on socket. LightGBM also supports MPI.
`MPI`_ is a high performance communication approach with `RDMA`_ support.

If you need to run a parallel learning application with high performance communication, you can build the LightGBM with MPI support.

Windows
^^^^^^^

On Windows MPI version of LightGBM can be built using

- **MS MPI** and **Visual Studio**;

- **MS MPI**, **CMake** and **VS Build Tools**.

With GUI
********

1. You need to install `MS MPI`_ first. Both ``msmpisdk.msi`` and ``msmpisetup.exe`` are needed.

2. Install `Visual Studio`_ (2015 or newer).

3. Download `zip archive`_ and unzip it.

4. Go to ``LightGBM-master/windows`` folder.

5. Open ``LightGBM.sln`` file with **Visual Studio**, choose ``Release_mpi`` configuration and click ``BUILD`` -> ``Build Solution (Ctrl+Shift+B)``.

   If you have errors about **Platform Toolset**, go to ``PROJECT`` -> ``Properties`` -> ``Configuration Properties`` -> ``General`` and select the toolset installed on your machine.

The exe file will be in ``LightGBM-master/windows/x64/Release_mpi`` folder.

From Command Line
*****************

1. You need to install `MS MPI`_ first. Both ``msmpisdk.msi`` and ``msmpisetup.exe`` are needed.

2. Install `Git for Windows`_, `CMake`_ (3.8 or higher) and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** (2015 or newer) is already installed).

3. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DUSE_MPI=ON ..
     cmake --build . --target ALL_BUILD --config Release

The exe and dll files will be in ``LightGBM/Release`` folder.

**Note**: Building MPI version by **MinGW** is not supported due to the miss of MPI library in it.

Linux
^^^^^

On Linux MPI version of LightGBM can be built using **Open MPI**, **CMake** and **gcc** or **Clang**.

1. Install `Open MPI`_.

2. Install `CMake`_.

3. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     mkdir build ; cd build
     cmake -DUSE_MPI=ON ..
     make -j4

**Note**: glibc >= 2.14 is required.

macOS
^^^^^

On macOS MPI version of LightGBM can be built using **Open MPI**, **CMake** and **Apple Clang** or **gcc**.

Apple Clang
***********

Only **Apple Clang** version 8.1 or higher is supported.

1. Install `CMake`_ (3.12 or higher):

   .. code::

     brew install cmake

2. Install **OpenMP**:

   .. code::

     brew install libomp

3. Install **Open MPI**:

   .. code::

     brew install open-mpi

4. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     mkdir build ; cd build
     cmake -DUSE_MPI=ON ..
     make -j4

gcc
***

1. Install `CMake`_ (3.2 or higher):

   .. code::

     brew install cmake

2. Install **gcc**:

   .. code::

     brew install gcc

3. Install **Open MPI**:

   .. code::

     brew install open-mpi

4. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     export CXX=g++-7 CC=gcc-7  # replace "7" with version of gcc installed on your machine
     mkdir build ; cd build
     cmake -DUSE_MPI=ON ..
     make -j4

Build GPU Version
~~~~~~~~~~~~~~~~~

Linux
^^^^^

On Linux GPU version of LightGBM can be built using **OpenCL**, **Boost**, **CMake** and **gcc** or **Clang**.

The following dependencies should be installed before compilation:

-  **OpenCL** 1.2 headers and libraries, which is usually provided by GPU manufacture.

   The generic OpenCL ICD packages (for example, Debian package ``cl-icd-libopencl1`` and ``cl-icd-opencl-dev``) can also be used.

-  **libboost** 1.56 or later (1.61 or later is recommended).

   We use Boost.Compute as the interface to GPU, which is part of the Boost library since version 1.61. However, since we include the source code of Boost.Compute as a submodule, we only require the host has Boost 1.56 or later installed. We also use Boost.Align for memory allocation. Boost.Compute requires Boost.System and Boost.Filesystem to store offline kernel cache.

   The following Debian packages should provide necessary Boost libraries: ``libboost-dev``, ``libboost-system-dev``, ``libboost-filesystem-dev``.

-  **CMake** 3.2 or later.

To build LightGBM GPU version, run the following commands:

.. code::

  git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
  mkdir build ; cd build
  cmake -DUSE_GPU=1 ..
  # if you have installed NVIDIA CUDA to a customized location, you should specify paths to OpenCL headers and library like the following:
  # cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
  make -j4

Windows
^^^^^^^

On Windows GPU version of LightGBM can be built using **OpenCL**, **Boost**, **CMake** and **VS Build Tools** or **MinGW**.

If you use **MinGW**, the build procedure is similar to the build on Linux. Refer to `GPU Windows Compilation <./GPU-Windows.rst>`__ to get more details.

Following procedure is for the **MSVC** (Microsoft Visual C++) build.

1. Install `Git for Windows`_, `CMake`_ (3.8 or higher) and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** (2015 or newer) is installed).

2. Install **OpenCL** for Windows. The installation depends on the brand (NVIDIA, AMD, Intel) of your GPU card.

   - For running on Intel, get `Intel SDK for OpenCL`_.

   - For running on AMD, get AMD APP SDK.

   - For running on NVIDIA, get `CUDA Toolkit`_.

   Further reading and correspondence table: `GPU SDK Correspondence and Device Targeting Table <./GPU-Targets.rst>`__.

3. Install `Boost Binary`_.

   **Note**: Match your Visual C++ version:
   
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

   **Note**: ``C:\local\boost_1_64_0\`` and ``C:\local\boost_1_64_0\lib64-msvc-14.0`` are locations of your **Boost** binaries. You also can set them to the environment variable to avoid ``Set ...`` commands when build.

Docker
^^^^^^

Refer to `GPU Docker folder <https://github.com/Microsoft/LightGBM/tree/master/docker/gpu>`__.

Build HDFS Version
~~~~~~~~~~~~~~~~~~

**Note**: Installation process of HDFS version is untested.

Linux
^^^^^

On Linux HDFS version of LightGBM can be built using **CMake** and **gcc** or **Clang**.

1. Install `CMake`_.

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     mkdir build ; cd build
     cmake -DUSE_HDFS=ON ..
     make -j4

Build Java Wrapper
~~~~~~~~~~~~~~~~~~

By the following instructions you can generate a JAR file containing the LightGBM `C API <./Development-Guide.rst#c-api>`__ wrapped by **SWIG**.

Windows
^^^^^^^

On Windows Java wrapper of LightGBM can be built using **Java**, **SWIG**, **CMake** and **VS Build Tools** or **MinGW**.

VS Build Tools
**************

1. Install `Git for Windows`_, `CMake`_ (3.8 or higher) and `VS Build Tools`_ (**VS Build Tools** is not needed if **Visual Studio** (2015 or newer) is already installed).

2. Install `SWIG`_ and **Java** (also make sure that ``JAVA_HOME`` is set properly).

3. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DUSE_SWIG=ON ..
     cmake --build . --target ALL_BUILD --config Release

The jar file will be in ``LightGBM/build`` folder and the dll files will be in ``LightGBM/Release`` folder.

MinGW-w64
*********

1. Install `Git for Windows`_, `CMake`_ and `MinGW-w64`_.

2. Install `SWIG`_ and **Java** (also make sure that ``JAVA_HOME`` is set properly).

3. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM
     cd LightGBM
     mkdir build
     cd build
     cmake -G "MinGW Makefiles" -DUSE_SWIG=ON ..
     mingw32-make.exe -j4

The jar file will be in ``LightGBM/build`` folder and the dll files will be in ``LightGBM/`` folder.

**Note**: You may need to run the ``cmake -G "MinGW Makefiles" -DUSE_SWIG=ON ..`` one more time if you encounter the ``sh.exe was found in your PATH`` error.

It is recommended to use **VS Build Tools (Visual Studio)** for its better multithreading efficiency in **Windows** for many-core systems (see `FAQ <./FAQ.rst#lightgbm>`__ Question 4 and Question 8).

Also, you may want to read `gcc Tips <./gcc-Tips.rst>`__.

Linux
^^^^^

On Linux Java wrapper of LightGBM can be built using **Java**, **SWIG**, **CMake** and **gcc** or **Clang**.

1. Install `CMake`_, `SWIG`_ and **Java** (also make sure that ``JAVA_HOME`` is set properly).

2. Run the following commands:

   .. code::

     git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
     mkdir build ; cd build
     cmake -DUSE_SWIG=ON ..
     make -j4

.. |download artifacts| image:: ./_static/images/artifacts-not-available.svg
   :target: https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html

.. _Python-package: https://github.com/Microsoft/LightGBM/tree/master/python-package

.. _R-package: https://github.com/Microsoft/LightGBM/tree/master/R-package

.. _zip archive: https://github.com/Microsoft/LightGBM/archive/master.zip

.. _Visual Studio: https://visualstudio.microsoft.com/downloads/

.. _Git for Windows: https://git-scm.com/download/win

.. _CMake: https://cmake.org/

.. _VS Build Tools: https://visualstudio.microsoft.com/downloads/

.. _MinGW-w64: https://mingw-w64.org/doku.php/download

.. _MPI: https://en.wikipedia.org/wiki/Message_Passing_Interface

.. _RDMA: https://en.wikipedia.org/wiki/Remote_direct_memory_access

.. _MS MPI: https://www.microsoft.com/en-us/download/details.aspx?id=57467

.. _Open MPI: https://www.open-mpi.org/

.. _Intel SDK for OpenCL: https://software.intel.com/en-us/articles/opencl-drivers

.. _CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

.. _Boost Binary: https://sourceforge.net/projects/boost/files/boost-binaries/1.64.0/

.. _SWIG: http://www.swig.org/download.html
