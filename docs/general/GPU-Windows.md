# Install LightGBM GPU version in Windows (CLI / R / Python), using MinGW/gcc

This is for a vanilla installation of Boost, including full compilation steps from source without precompiled libraries.

Installation steps (depends on what you are going to do):

* Install the appropriate OpenCL SDK
* Install MinGW
* Install Boost
* Install Git
* Install cmake
* Create LightGBM binaries
* Debugging LightGBM in CLI (if GPU is crashing or any other crash reason)

If you wish to use another compiler like Visual Studio C++ compiler, you need to adapt the steps to your needs.

For this compilation tutorial, I am using AMD SDK for our OpenCL steps. However, you are free to use any OpenCL SDK you want, you just need to adjust the PATH correctly.

You will also need administrator rights. This will not work without them.

At the end, you can restore your original PATH.

---

## Modifying PATH (for newbies)

To modify PATH, just follow the pictures after going to the `Control Panel`:

![System](https://cloud.githubusercontent.com/assets/9083669/24928495/e3293b12-1f02-11e7-861d-37ec2d086dba.png)

Then, go to `Advanced` > `Environment Variables...`:

![Advanced System Settings](https://cloud.githubusercontent.com/assets/9083669/24928515/00b252ae-1f03-11e7-8ff6-fbf78c503754.png)

Under `System variables`, the variable `Path`:

![Environment Variables](https://cloud.githubusercontent.com/assets/9083669/24928517/00fd8008-1f03-11e7-84e2-7dc8fd50d6ce.png)

---

### Antivirus Performance Impact

Does not apply to you if you do not use a third-party antivirus nor the default preinstalled antivirus on Windows.

**Windows Defender or any other antivirus will have a significant impact on the speed you will be able to perform the steps.** It is recommended to **turn them off temporarily** until you finished with building and setting up everything, then turn them back on, if you are using them.

---

## OpenCL SDK Installation

Installing the appropriate OpenCL SDK requires you to download the correct vendor source SDK. You need to know on what you are going to use LightGBM!:

* For running on Intel, get Intel SDK for OpenCL: https://software.intel.com/en-us/articles/opencl-drivers
* For running on AMD, get AMD APP SDK: http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/
* For running on NVIDIA, get CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

Further reading and correspondnce table (especially if you intend to use cross-platform devices, like Intel CPU with AMD APP SDK): [GPU SDK Correspondence and Device Targeting Table](./GPU-Targets.md).

---

## MinGW correct compiler selection

If you are expecting to use LightGBM without R, you need to install MinGW. Installing MinGW is straightforward, download this: http://iweb.dl.sourceforge.net/project/mingw-w64/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/installer/mingw-w64-install.exe

Make sure you are using the x86_64 architecture, and do not modify anything else. You may choose a version other than the most recent one if you need a previous MinGW version.

![MinGW installation](https://cloud.githubusercontent.com/assets/9083669/25063112/a7374ee2-21db-11e7-89f4-ae6f413a16f1.png)

Then, add to your PATH the following (to adjust to your MinGW version):

```
C:\Program Files\mingw-w64\x86_64-5.3.0-posix-seh-rt_v4-rev0\mingw64\bin
```

**Warning: R users (even if you do not want LightGBM for R)**

**If you have RTools and MinGW installed, and wish to use LightGBM in R, get rid of MinGW from PATH (to keep: `c:\Rtools\bin;c:\Rtools\mingw_32\bin` for 32-bit R installation, `c:\Rtools\bin;c:\Rtools\mingw_64\bin` for 64-bit R installation).**

You can check which MinGW version you are using by running the following in a command prompt: `gcc -v`:

![R MinGW used](https://cloud.githubusercontent.com/assets/9083669/24927803/80b83782-1f00-11e7-961a-068d58d82885.png)

To check whether you need 32-bit or 64-bit MinGW for R, install LightGBM as usual and check for the following:

```r
* installing *source* package 'lightgbm' ...
** libs
c:/Rtools/mingw_64/bin/g++
```

If it says `mingw_64` then you need the 64-bit version (PATH with `c:\Rtools\bin;c:\Rtools\mingw_64\bin`), otherwise you need the 32-bit version (`c:\Rtools\bin;c:\Rtools\mingw_32\bin`), the latter being a very rare and untested case.

Quick installation of LightGBM can be done using:

```r
devtools::install_github("Microsoft/LightGBM", subdir = "R-package")
```

---

## Boost Compilation

Installing Boost requires to download Boost and to install it. It takes about 10 minutes to several hours depending on your CPU speed and network speed.

We will assume an installation in `C:\boost` and a general installation (like in Unix variants: without versioning and without type tags).

There is one mandatory step to check: the compiler.

* **Warning: if you want the R installation**: If you have already MinGW in your PATH variable, get rid of it (you will link to the wrong compiler otherwise).
* **Warning: if you want the CLI installation**: if you have already Rtools in your PATH variable, get rid of it (you will link to the wrong compiler otherwise).

* R installation must have Rtools in PATH
* CLI / Python installation must have MinGW (not Rtools) in PATH

In addition, assuming you are going to use `C:\boost` for the folder path, you should add now already the following to PATH: `C:\boost\boost-build\bin;C:\boost\boost-build\include\boost`. Adjust `C:\boost` if you install it elsewhere.

We can now start downloading and compiling the required Boost libraries:

* Download Boost here: http://www.boost.org/users/history/version_1_63_0.html (boost_1_63_0.zip).
* Extract the archive to `C:\boost`.
* Open a command prompt, and run `cd C:\boost\boost_1_63_0\tools\build`.
* In command prompt, run `bootstrap.bat gcc`.
* In command prompt, run `b2 install --prefix="C:\boost\boost-build" toolset=gcc`.
* In command prompt, run `cd C:\boost\boost_1_63_0`.

To build the Boost libraries, you have two choices for command prompt:

* If you have only one single core, you can use the default `b2 install --build_dir="C:\boost\boost-build" --prefix="C:\boost\boost-build" toolset=gcc --with=filesystem,system threading=multi --layout=system release`.
* If you want to do a multithreaded library building (faster), add -j N by replacing N by the number of cores/threads you have. For instance, for 2 cores, you would do `b2 install --build_dir="C:\boost\boost-build" --prefix="C:\boost\boost-build" toolset=gcc --with=filesystem,system threading=multi --layout=system release -j 2`

Ignore all the errors popping up, like Python, etc., they do not matter for us.

Your folder should look like this at the end (not fully detailed):

```
- C
  |--- boost
  |------ boost_1_63_0
  |--------- some folders and files
  |------ boost-build
  |--------- bin
  |--------- include
  |------------ boost
  |------ lib
  |------ share
```

This is what you should (approximately) get at the end of Boost compilation:

![Boost compiled](https://cloud.githubusercontent.com/assets/9083669/24918623/5152a3c0-1ee1-11e7-9d59-d75fb1193241.png)

---

## Git Installation

Installing Git for Windows is straightforward, use the following link: https://git-for-windows.github.io/

![git for Windows](https://cloud.githubusercontent.com/assets/9083669/24919716/e2612ea6-1ee4-11e7-9eca-d30997b911ff.png)

Then, click on the big Download button, you can't miss it.

Now, we can fetch LightGBM repository for GitHub. Run Git Bash and the following command:

```
cd C:/
mkdir github_repos
cd github_repos
git clone --recursive https://github.com/Microsoft/LightGBM
```

Your LightGBM repository copy should now be under `C:\github_repos\LightGBM`. You are free to use any folder you want, but you have to adapt.

Keep Git Bash open.

---

## cmake Installation, Configuration, Generation

**CLI / Python users only**

Installing cmake requires one download first and then a lot of configuration for LightGBM:

![Downloading cmake](https://cloud.githubusercontent.com/assets/9083669/24919759/fe5f4d90-1ee4-11e7-992e-00f8d9bfe6dd.png)

* Download cmake 3.8.0 here: https://cmake.org/download/.
* Install cmake.
* Run cmake-gui.
* Select the folder where you put LightGBM for `Where is the source code`, default using our steps would be `C:/github_repos/LightGBM`.
* Copy the folder name, and add `/build` for "Where to build the binaries", default using our steps would be `C:/github_repos/LightGBM/build`.
* Click `Configure`.

![Create directory](https://cloud.githubusercontent.com/assets/9083669/24921175/33feee92-1eea-11e7-8330-6d8e519a6177.png)

![MinGW makefiles to use](https://cloud.githubusercontent.com/assets/9083669/24921193/404dd384-1eea-11e7-872e-6220e0f8b321.png)

* Lookup for `USE_GPU` and check the checkbox

![Use GPU](https://cloud.githubusercontent.com/assets/9083669/24921364/d7ccd426-1eea-11e7-8054-d4bd3a39af84.png)

* Click `Configure`

You should get (approximately) the following after clicking Configure:

![Configured LightGBM](https://cloud.githubusercontent.com/assets/9083669/24919175/1301b42e-1ee3-11e7-9823-70a1d4c8c39e.png)

```
Looking for CL_VERSION_2_0
Looking for CL_VERSION_2_0 - found
Found OpenCL: C:/Windows/System32/OpenCL.dll (found version "2.0") 
OpenCL include directory:C:/Program Files (x86)/AMD APP SDK/3.0/include
Boost version: 1.63.0
Found the following Boost libraries:
  filesystem
  system
Configuring done
```

* Click `Generate` to get the following message:

```
Generating done
```

This is straightforward, as cmake is providing a large help into locating the correct elements.

---

## LightGBM Compilation (CLI: final step)

### Installation in CLI

**CLI / Python users**

Creating LightGBM libraries is very simple as all the important and hard steps were done before.

You can do everything in the Git Bash console you left open:

* If you closed Git Bash console previously, run this to get back to the build folder: `cd C:/github_repos/LightGBM/build`
* If you did not close the Git Bash console previously, run this to get to the build folder: `cd LightGBM/build`
* Setup MinGW as make using `alias make='mingw32-make'` (otherwise, beware error and name clash!).
* In Git Bash, run `make` and see LightGBM being installing!

![LightGBM with GPU support compiled](https://cloud.githubusercontent.com/assets/9083669/24923499/0cb90572-1ef2-11e7-8842-371d038fb5e9.png)

If everything was done correctly, you now compiled CLI LightGBM with GPU support!

### Testing in CLI

You can now test LightGBM directly in CLI in a **command prompt** (not Git Bash):

```
cd C:/github_repos/LightGBM/examples/binary_classification
"../../lightgbm.exe" config=train.conf data=binary.train valid=binary.test objective=binary device=gpu
```

![LightGBM in CLI with GPU](https://cloud.githubusercontent.com/assets/9083669/24958722/98021e72-1f90-11e7-80a9-204d56ace395.png)

Congratulations for reaching this stage!

To learn how to target a correct CPU or GPU for training, please see: [GPU SDK Correspondence and Device Targeting Table](./GPU-Targets.md).

---

## LightGBM Setup and Installation for Python (Python: final step)

### Installation in Python

**Python users, extra steps**

Installing in Python is as straightforward as CLI. Assuming you already have `numpy`, `scipy`, `scikit-learn`, and `setuptools`, run the following in the Git Console:

```
cd C:/github_repos/LightGBM/python-package/
python setup.py install
```

![LightGBM with GPU support in Python](https://cloud.githubusercontent.com/assets/9083669/24957399/f14d0da2-1f8b-11e7-8f90-e8a606266265.png)

### Testing in Python

You can try to run the following demo script in Python to test if it works:

```python
import lightgbm as lgb
import pandas as pd
import os

# load or create your dataset
print('Load data...')
os.chdir('C:/github_repos/LightGBM/examples/regression')
df_train = pd.read_csv('regression.train', header=None, sep='\t')
df_test = pd.read_csv('regression.test', header=None, sep='\t')

y_train = df_train[0].values
y_test = df_test[0].values
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'objective': 'regression',
    'metric': 'l2',
    'verbose': 2,
    'device': 'gpu'
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
```

![LightGBM GPU in Python](https://cloud.githubusercontent.com/assets/9083669/24959269/9202a670-1f92-11e7-94a1-a7c062eaf91c.png)

Congratulations for reaching this stage!

To learn how to target a correct CPU or GPU for training, please see: [GPU SDK Correspondence and Device Targeting Table](./GPU-Targets.md).

---

## LightGBM Setup and Installation for R (R: final step)

### Preparation for R

**R users**

This gets a bit complicated for this step.

First of all, you need to to find the correct paths for the following, and keep them in a notepad:

* `BOOST_INCLUDE_DIR = "C:/boost/boost-build/include"`: if you followed the instructions, it is `C:/boost/boost-build/include`.
* `BOOST_LIBRARY = "C:/boost/boost-build/lib"`: if you followed the instructions, it is `C:/boost/boost-build/lib`.
* `OpenCL_INCLUDE_DIR = "C:/Program Files (x86)/AMD APP SDK/3.0/include"`: this varies, it must be the OpenCL SDK folder containing the file `CL/CL.h` (caps do not matter). For instance, using AMD APP SDK, it becomes `C:/Program Files (x86)/AMD APP SDK/3.0/include`.
* `OpenCL_LIBRARY = "C:/Program Files (x86)/AMD APP SDK/3.0/lib/x86_64"`: this varies, it must be the OpenCL SDK folder containing the file `OpenCL.lib` (caps do not matter). For instance, using AMD APP SDK, it becomes `C:/Program Files (x86)/AMD APP SDK/3.0/lib/x86_64`.

Second, you need to find out where is `Makeconf`, as it is the essential file you will need to use to specify the PATH for R. Run the following code to get the file path to your `Makeconf` file:

```r
file.path(R.home("etc"), "Makeconf"))
```

For instance, `"C:/PROGRA~1/MIE74D~1/RCLIEN~1/R_SERVER/etc/Makeconf"` means `"C:\Program Files\Microsoft\R_Client\R_SERVER\etc\Makeconf"`.

Third, edit the `Makeconf` file **as an Administrator**. Remember the first step we had to do where we store four different values in a notepad? We apply them right now.

For instance, for this installation and using AMD OpenCL SDK, we are doing the following below `LINKFLAGS`:

```r
BOOST_INCLUDE_DIR = "C:/boost/boost-build/include"
BOOST_LIBRARY = "C:/boost/boost-build/lib"
OpenCL_INCLUDE_DIR = "C:/Program Files (x86)/AMD APP SDK/3.0/include"
OpenCL_LIBRARY = "C:/Program Files (x86)/AMD APP SDK/3.0/lib/x86_64"
```

![Getting R Makeconf](https://cloud.githubusercontent.com/assets/9083669/24978322/8f374ac0-1fd0-11e7-9164-ace708d600cc.png)

From there, you have two solutions:

* Installation Method 1 (hard): Use your local LightGBM repository with the latest and recent development features
* Installation Method 2 (easy): Use ez_lgb, [Laurae2/LightGBM 's repository](https://github.com/Laurae2/LightGBM) for installing LightGBM easily, but it might not be up to date. It uses compute to patch boostorg/compute#704 (boostorg/compute@6de7f64)

### Installation Method 1

Edit 1 to do: you need to include proper GPU compilation support to the R package by adding the following to `R-package\src\lightgbm-all.cpp`:

```r
// gpu support
#include "../../src/treelearner/gpu_tree_learner.cpp"

```

The `lightgbm-all.cpp` becomes:

```r
// application
#include "../../src/application/application.cpp"

// boosting
#include "../../src/boosting/boosting.cpp"
#include "../../src/boosting/gbdt.cpp"

// io
#include "../../src/io/bin.cpp"
#include "../../src/io/config.cpp"
#include "../../src/io/dataset.cpp"
#include "../../src/io/dataset_loader.cpp"
#include "../../src/io/metadata.cpp"
#include "../../src/io/parser.cpp"
#include "../../src/io/tree.cpp"

// metric
#include "../../src/metric/dcg_calculator.cpp"
#include "../../src/metric/metric.cpp"

// network
#include "../../src/network/linker_topo.cpp"
#include "../../src/network/linkers_socket.cpp"
#include "../../src/network/network.cpp"

// objective
#include "../../src/objective/objective_function.cpp"

// treelearner
#include "../../src/treelearner/data_parallel_tree_learner.cpp"
#include "../../src/treelearner/feature_parallel_tree_learner.cpp"
#include "../../src/treelearner/serial_tree_learner.cpp"
#include "../../src/treelearner/tree_learner.cpp"
#include "../../src/treelearner/voting_parallel_tree_learner.cpp"

// c_api
#include "../../src/c_api.cpp"

// gpu support
#include "../../src/treelearner/gpu_tree_learner.cpp"

```

Edit 2 to do: you need to edit the `Makevars.win` in `R-package\src` appropriately by overwriting the following flags (`LGBM_RFLAGS`, `PKG_CPPFLAGS`, `PKG_LIBS`) with the following:

```r
LGBM_RFLAGS = -DUSE_SOCKET -DUSE_GPU=1

PKG_CPPFLAGS= -I$(PKGROOT)/include -I$(BOOST_INCLUDE_DIR) -I$(OpenCL_INCLUDE_DIR) -I../compute/include $(LGBM_RFLAGS)

PKG_LIBS = $(SHLIB_OPENMP_CFLAGS) $(SHLIB_PTHREAD_FLAGS) -lws2_32 -liphlpapi -L$(BOOST_LIBRARY) -lboost_filesystem -lboost_system -L$(OpenCL_LIBRARY) -lOpenCL
```

Your `Makevars.win` will look like this:

![Makevars look](https://cloud.githubusercontent.com/assets/9083669/24978371/b9124674-1fd0-11e7-8e3d-4ebb3d6340dd.png)

Or, copy & paste this:

```r
# package root
PKGROOT=../../

ENABLE_STD_THREAD=1
CXX_STD = CXX11

LGBM_RFLAGS = -DUSE_SOCKET -DUSE_GPU=1

PKG_CPPFLAGS= -I$(PKGROOT)/include -I$(BOOST_INCLUDE_DIR) -I$(OpenCL_INCLUDE_DIR) -I../compute/include $(LGBM_RFLAGS)
PKG_CXXFLAGS= $(SHLIB_OPENMP_CFLAGS) $(SHLIB_PTHREAD_FLAGS) -std=c++11
PKG_LIBS = $(SHLIB_OPENMP_CFLAGS) $(SHLIB_PTHREAD_FLAGS) -lws2_32 -liphlpapi -L$(BOOST_LIBRARY) -lboost_filesystem -lboost_system -L$(OpenCL_LIBRARY) -lOpenCL
OBJECTS = ./lightgbm-all.o ./lightgbm_R.o

```

Now, we need to install LightGBM as usual:

* Open an interactive R console.
* Assuming you have the LightGBM folder in `C:/LightGBM`, run `devtools::install("C:/github_repos/LightGBM/R-package")`.

![LightGBM installed with GPU support](https://cloud.githubusercontent.com/assets/9083669/24955074/40179df8-1f82-11e7-909b-d64e62e92641.png)

### Installation Method 2

This is very simple, as you only need to open an R interactive console and run:

```r
devtools::install_github("Laurae2/LightGBM", subdir = "R-package")
```

It will install automatically LightGBM for R with GPU support, without the need to edit manually the `Makevars.win` and `lightgbm-all.cpp`.

Laurae's LightGBM has all the steps of installation method 1 done for you. Therefore, this is a GPU-only version. You can check how many days it is behind Microsoft/LightGBM master branch and the latest master branch commit made here:

![image](https://cloud.githubusercontent.com/assets/9083669/25041428/3cacfc00-2110-11e7-8000-a783cde6f124.png)

Self-contained packages are not provided are untested. It might work but it was untested, and requires to modify the `fullcode` files instead of the regular files (`lightgbm-fullcode.cpp` and `Makevars_fullcode.win`).

### Testing in R

When you run LightGBM with a specific amount of bins, it will create the appropriate kernels. This will be obviously leading to poor performance during the first usages of LightGBM. But once the kernels are built for the number of bins you are using, you do not have to care about building them again.

Test GPU support with the following:

```r
library(lightgbm)
data(agaricus.train, package = "lightgbm")
train <- agaricus.train
dtrain <- lgb.Dataset(train$data, label = train$label)
data(agaricus.test, package = "lightgbm")
test <- agaricus.test
dtest <- lgb.Dataset.create.valid(dtrain, test$data, label = test$label)
params <- list(objective = "regression", metric = "l2", device = "gpu")
valids <- list(test = dtest)
model <- lgb.train(params,
                   dtrain,
                   100,
                   valids,
                   min_data = 1,
                   learning_rate = 1,
                   early_stopping_rounds = 10)
```

![LightGBM with GPU support running](https://cloud.githubusercontent.com/assets/9083669/24955029/0c7dc17a-1f82-11e7-8b75-89a5173f7276.png)

Congratulations for reaching this stage!

To learn how to target a correct CPU or GPU for training, please see: [GPU SDK Correspondence and Device Targeting Table](./GPU-Targets.md).

## Debugging LightGBM crashes in CLI

Now that you compiled LightGBM, you try it... and you always see a segmentation fault or an undocumented crash with GPU support:

![Segmentation Fault](https://cloud.githubusercontent.com/assets/9083669/25015529/7326860a-207c-11e7-8fc3-320b2be619a6.png)

Please check you are using the right device and whether it works with the default `gpu_device_id = 0` and `gpu_platform_id = 0`. If it still does not work with the default values, then you should follow all the steps below.

You will have to redo the compilation steps for LightGBM to add debugging mode. This involves:

* Deleting `C:/github_repos/LightGBM/build` folder
* Deleting `lightgbm.exe`, `lib_lightgbm.dll`, and `lib_lightgbm.dll.a` files

![Files to remove](https://cloud.githubusercontent.com/assets/9083669/25051307/3b7dd084-214c-11e7-9758-c338c8cacb1e.png)

Once you removed the file, go into cmake, and follow the usual steps. Before clicking "Generate", click on "Add Entry":

![Added manual entry in cmake](https://cloud.githubusercontent.com/assets/9083669/25051323/508969ca-214c-11e7-884a-20882cd3936a.png)

In addition, click on Configure and Generate:

![Configured and Generated cmake](https://cloud.githubusercontent.com/assets/9083669/25051236/e71237ce-214b-11e7-8faa-d885d7826fe1.png)

And then, follow the regular LightGBM CLI installation from there.

Once you have installed LightGBM CLI, assuming your LightGBM is in `C:\github_repos\LightGBM`, open a command prompt and run the following:

`gdb --args "../../lightgbm.exe" config=train.conf data=binary.train valid=binary.test objective=binary device=gpu`

![Debug run](https://cloud.githubusercontent.com/assets/9083669/25041067/8fdbee66-210d-11e7-8adb-79b688c051d5.png)

Type `run` and Enter key.

You will probably get something similar to this:

```
[LightGBM] [Info] This is the GPU trainer!!
[LightGBM] [Info] Total Bins 6143
[LightGBM] [Info] Number of data: 7000, number of used features: 28
[New Thread 105220.0x1a62c]
[LightGBM] [Info] Using GPU Device: Oland, Vendor: Advanced Micro Devices, Inc.
[LightGBM] [Info] Compiling OpenCL Kernel with 256 bins...

Program received signal SIGSEGV, Segmentation fault.
0x00007ffbb37c11f1 in strlen () from C:\Windows\system32\msvcrt.dll
(gdb) 
```

There, write `backtrace` and Enter key as many times as gdb requests two choices:

```
Program received signal SIGSEGV, Segmentation fault.
0x00007ffbb37c11f1 in strlen () from C:\Windows\system32\msvcrt.dll
(gdb) backtrace
#0  0x00007ffbb37c11f1 in strlen () from C:\Windows\system32\msvcrt.dll
#1  0x000000000048bbe5 in std::char_traits<char>::length (__s=0x0)
    at C:/PROGRA~1/MINGW-~1/X86_64~1.0-P/mingw64/x86_64-w64-mingw32/include/c++/bits/char_traits.h:267
#2  std::operator+<char, std::char_traits<char>, std::allocator<char> > (__rhs="\\", __lhs=0x0)
    at C:/PROGRA~1/MINGW-~1/X86_64~1.0-P/mingw64/x86_64-w64-mingw32/include/c++/bits/basic_string.tcc:1157
#3  boost::compute::detail::appdata_path[abi:cxx11]() () at C:/boost/boost-build/include/boost/compute/detail/path.hpp:38
#4  0x000000000048eec3 in boost::compute::detail::program_binary_path (hash="d27987d5bd61e2d28cd32b8d7a7916126354dc81", create=create@entry=false)
    at C:/boost/boost-build/include/boost/compute/detail/path.hpp:46
#5  0x00000000004913de in boost::compute::program::load_program_binary (hash="d27987d5bd61e2d28cd32b8d7a7916126354dc81", ctx=...)
    at C:/boost/boost-build/include/boost/compute/program.hpp:605
#6  0x0000000000490ece in boost::compute::program::build_with_source (
    source="\n#ifndef _HISTOGRAM_256_KERNEL_\n#define _HISTOGRAM_256_KERNEL_\n\n#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable\n#pragma OPENC
L EXTENSION cl_khr_global_int32_base_atomics : enable\n\n//"..., context=...,
    options=" -D POWER_FEATURE_WORKGROUPS=5 -D USE_CONSTANT_BUF=0 -D USE_DP_FLOAT=0 -D CONST_HESSIAN=0 -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -c
l-fast-relaxed-math") at C:/boost/boost-build/include/boost/compute/program.hpp:549
#7  0x0000000000454339 in LightGBM::GPUTreeLearner::BuildGPUKernels () at C:\LightGBM\src\treelearner\gpu_tree_learner.cpp:583
#8  0x00000000636044f2 in libgomp-1!GOMP_parallel () from C:\Program Files\mingw-w64\x86_64-5.3.0-posix-seh-rt_v4-rev0\mingw64\bin\libgomp-1.dll
#9  0x0000000000455e7e in LightGBM::GPUTreeLearner::BuildGPUKernels (this=this@entry=0x3b9cac0)
    at C:\LightGBM\src\treelearner\gpu_tree_learner.cpp:569
#10 0x0000000000457b49 in LightGBM::GPUTreeLearner::InitGPU (this=0x3b9cac0, platform_id=<optimized out>, device_id=<optimized out>)
    at C:\LightGBM\src\treelearner\gpu_tree_learner.cpp:720
#11 0x0000000000410395 in LightGBM::GBDT::ResetTrainingData (this=0x1f26c90, config=<optimized out>, train_data=0x1f28180, objective_function=0x1f280e0,
    training_metrics=std::vector of length 2, capacity 2 = {...}) at C:\LightGBM\src\boosting\gbdt.cpp:98
#12 0x0000000000402e93 in LightGBM::Application::InitTrain (this=this@entry=0x23f9d0) at C:\LightGBM\src\application\application.cpp:213
---Type <return> to continue, or q <return> to quit---
#13 0x00000000004f0b55 in LightGBM::Application::Run (this=0x23f9d0) at C:/LightGBM/include/LightGBM/application.h:84
#14 main (argc=6, argv=0x1f21e90) at C:\LightGBM\src\main.cpp:7
```

Right-click the command prompt, click "Mark", and select all the text from the first line (with the command prompt containing gdb) to the last line printed, containing all the log, such as:

```
C:\LightGBM\examples\binary_classification>gdb --args "../../lightgbm.exe" config=train.conf data=binary.train valid=binary.test objective=binary device
=gpu
GNU gdb (GDB) 7.10.1
Copyright (C) 2015 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
and "show warranty" for details.
This GDB was configured as "x86_64-w64-mingw32".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
<http://www.gnu.org/software/gdb/documentation/>.
For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ../../lightgbm.exe...done.
(gdb) run
Starting program: C:\LightGBM\lightgbm.exe "config=train.conf" "data=binary.train" "valid=binary.test" "objective=binary" "device=gpu"
[New Thread 105220.0x199b8]
[New Thread 105220.0x783c]
[Thread 105220.0x783c exited with code 0]
[LightGBM] [Info] Finished loading parameters
[New Thread 105220.0x19490]
[New Thread 105220.0x1a71c]
[New Thread 105220.0x19a24]
[New Thread 105220.0x4fb0]
[Thread 105220.0x4fb0 exited with code 0]
[LightGBM] [Info] Loading weights...
[New Thread 105220.0x19988]
[Thread 105220.0x19988 exited with code 0]
[New Thread 105220.0x1a8fc]
[Thread 105220.0x1a8fc exited with code 0]
[LightGBM] [Info] Loading weights...
[New Thread 105220.0x1a90c]
[Thread 105220.0x1a90c exited with code 0]
[LightGBM] [Info] Finished loading data in 1.011408 seconds
[LightGBM] [Info] Number of positive: 3716, number of negative: 3284
[LightGBM] [Info] This is the GPU trainer!!
[LightGBM] [Info] Total Bins 6143
[LightGBM] [Info] Number of data: 7000, number of used features: 28
[New Thread 105220.0x1a62c]
[LightGBM] [Info] Using GPU Device: Oland, Vendor: Advanced Micro Devices, Inc.
[LightGBM] [Info] Compiling OpenCL Kernel with 256 bins...

Program received signal SIGSEGV, Segmentation fault.
0x00007ffbb37c11f1 in strlen () from C:\Windows\system32\msvcrt.dll
(gdb) backtrace
#0  0x00007ffbb37c11f1 in strlen () from C:\Windows\system32\msvcrt.dll
#1  0x000000000048bbe5 in std::char_traits<char>::length (__s=0x0)
    at C:/PROGRA~1/MINGW-~1/X86_64~1.0-P/mingw64/x86_64-w64-mingw32/include/c++/bits/char_traits.h:267
#2  std::operator+<char, std::char_traits<char>, std::allocator<char> > (__rhs="\\", __lhs=0x0)
    at C:/PROGRA~1/MINGW-~1/X86_64~1.0-P/mingw64/x86_64-w64-mingw32/include/c++/bits/basic_string.tcc:1157
#3  boost::compute::detail::appdata_path[abi:cxx11]() () at C:/boost/boost-build/include/boost/compute/detail/path.hpp:38
#4  0x000000000048eec3 in boost::compute::detail::program_binary_path (hash="d27987d5bd61e2d28cd32b8d7a7916126354dc81", create=create@entry=false)
    at C:/boost/boost-build/include/boost/compute/detail/path.hpp:46
#5  0x00000000004913de in boost::compute::program::load_program_binary (hash="d27987d5bd61e2d28cd32b8d7a7916126354dc81", ctx=...)
    at C:/boost/boost-build/include/boost/compute/program.hpp:605
#6  0x0000000000490ece in boost::compute::program::build_with_source (
    source="\n#ifndef _HISTOGRAM_256_KERNEL_\n#define _HISTOGRAM_256_KERNEL_\n\n#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable\n#pragma OPENC
L EXTENSION cl_khr_global_int32_base_atomics : enable\n\n//"..., context=...,
    options=" -D POWER_FEATURE_WORKGROUPS=5 -D USE_CONSTANT_BUF=0 -D USE_DP_FLOAT=0 -D CONST_HESSIAN=0 -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -c
l-fast-relaxed-math") at C:/boost/boost-build/include/boost/compute/program.hpp:549
#7  0x0000000000454339 in LightGBM::GPUTreeLearner::BuildGPUKernels () at C:\LightGBM\src\treelearner\gpu_tree_learner.cpp:583
#8  0x00000000636044f2 in libgomp-1!GOMP_parallel () from C:\Program Files\mingw-w64\x86_64-5.3.0-posix-seh-rt_v4-rev0\mingw64\bin\libgomp-1.dll
#9  0x0000000000455e7e in LightGBM::GPUTreeLearner::BuildGPUKernels (this=this@entry=0x3b9cac0)
    at C:\LightGBM\src\treelearner\gpu_tree_learner.cpp:569
#10 0x0000000000457b49 in LightGBM::GPUTreeLearner::InitGPU (this=0x3b9cac0, platform_id=<optimized out>, device_id=<optimized out>)
    at C:\LightGBM\src\treelearner\gpu_tree_learner.cpp:720
#11 0x0000000000410395 in LightGBM::GBDT::ResetTrainingData (this=0x1f26c90, config=<optimized out>, train_data=0x1f28180, objective_function=0x1f280e0,
    training_metrics=std::vector of length 2, capacity 2 = {...}) at C:\LightGBM\src\boosting\gbdt.cpp:98
#12 0x0000000000402e93 in LightGBM::Application::InitTrain (this=this@entry=0x23f9d0) at C:\LightGBM\src\application\application.cpp:213
---Type <return> to continue, or q <return> to quit---
#13 0x00000000004f0b55 in LightGBM::Application::Run (this=0x23f9d0) at C:/LightGBM/include/LightGBM/application.h:84
#14 main (argc=6, argv=0x1f21e90) at C:\LightGBM\src\main.cpp:7
```

And open an issue in GitHub here with that log: https://github.com/Microsoft/LightGBM/issues
