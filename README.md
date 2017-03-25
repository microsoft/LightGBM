GPU Accelerated LightGBM for Histogram-based GBDT Training
=========================

This is the development repository of GPU-accelerated LightGBM. 
LightGBM is a popular Gradient Boosted Decision Tree (GBDT) training system,
and has been shown to be faster than XGBoost on large-scale datasets.
Our aim is to accelerate the feature histogram construction in LightGBM on GPUs,
and we propose an efficient algorithm on GPU to accelerate this process.
Our implementation is highly modular and does not affect existing features 
of LightGBM.

Please consider this repository experimental. If you find any problems during using
GPU acceleration, please send an email to [Huan Zhang](ecezhang@ucdavis.edu) or 
open an GitHub issue.

Build LightGBM with GPU support
-------------------------

The following dependencies should be installed before compilation:

- OpenCL 1.2 headers and libraries, which is usually provided by GPU manufacture.  
  The generic OpenCL ICD packages (for example, ocl-icd-libopencl1,
  ocl-icd-opencl-dev) can also be used.

- libboost 1.56 or later (1.61 or later recommended). We use Boost.Compute as
  the interface to GPU, which is part of the Boost library since version 1.61.
  However, since we include the source code of Boost.Compute as a submodule, we
  only require the host has Boost 1.56 or later installed. We also use
  Boost.Align for memory allocation. Boost.Compute also replies on Boost.System
  and Boost.Filesystem to store offline kernel cache. The following Debian 
  packages should provide necessary libraries: 
  `libboost-dev, libboost-system-dev, libboost-filesystem-dev`.

- CMake 3.2 or later

Currently only building on Linux has been tested, but it should also work with
MinGW on Windows as long as dependencies are available. To build LightGBM-GPU,
use the following procedure:

First clone this repository:

```
git clone --recursive https://github.com/huanzhang12/lightgbm-gpu.git
```

Then run `cmake` and `make`:

```
cd lightgbm-gpu
mkdir build ; cd build
cmake -DUSE_GPU=1 .. 
make -j$(nproc) 
```

GPU Related Configurations
--------------------------

To enable the GPU tree trainer, simply passing the parameter `tree_learner=gpu` to LightGBM.

The following new parameters are added:

- `gpu_platform_id`: OpenCL Platform ID (default: -1, selecting the default OpenCL platform).
This is useful only when you have OpenCL devices from different vendors.

- `gpu_device_id`: OpenCL Device ID (default: -1, selecting the default device).
Specify which GPU to run on if you have multiple GPUs installed.

- `gpu_use_dp`: When setting to `true`, double precision GPU kernels will be used 
(default: `false`, using single precision). When setting to `true`, the GPU tree
trainer should generate (almost) identical results as the CPU trainer.

- `sparse_threshold`: The threshold of zero elements percentage for
  treating a feature as dense feature. When setting to 1, all features are
  processed as dense features (default: 0.8).

To get good speedup with GPU, it is suggested to use a smaller number of bins.
Setting `max_bin=64` is recommended, as it usually does not noticeably affect
training accuracy on large datasets, but GPU training can be significantly
faster than the original version using the default bin size of 255. 
Also, try to use single precision training (`gpu_use_dp=false`) when possible, 
because most GPUs (especially NVIDIA consumer GPUs) have poor double-precision 
performance.

Supported Hardware
--------------------------

Our GPU code targets AMD Graphics Core Next (GCN) architecture and NVIDIA
Maxwell and Pascal architectures. Most AMD GPUs released after 2012 and NVIDIA
GPUs released after 2014 should be supported. We have tested the GPU
implementation on the following GPUs:

- AMD RX 480 with AMDGPU-pro driver 16.60 on Ubuntu 16.10
- AMD R9 280X with fglrx driver 15.302.2301 on Ubuntu 16.10
- NVIDIA GTX 1080 with driver 375.39 and CUDA 8.0 on Ubuntu 16.10 
- NVIDIA Titan X (Pascal) with driver 367.48 and CUDA 8.0 on Ubuntu 16.04

The use of the following hardware is discouraged:

- NVIDIA Kepler (K80, K40, K20, most GeForce GTX 700 series GPUs) or earilier
  NVIDIA GPUs. They don't support hardware atomic operations in local memory space
  and thus histogram construction will be slow.

- AMD VLIW4-based GPUs, including Radeon HD 6xxx series and earlier GPUs. These
  GPUs have been discontinued for years and are rarely seen nowadays.

Datasets
--------------------------

Datasets HIGGS, Yahoo LTR, Microsoft LTR and Expo that are used for 
[LightGBM benchmarks](https://github.com/Microsoft/LightGBM/wiki/Experiments#parallel-experiment) 
work on GPUs. To prepare these datasets, follow the instructions in
[this repo](https://github.com/guolinke/boosting_tree_benchmarks).

We also tested our implementation with the `epsilon` dataset, available at 
[LibSVM Datasets](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).

Other large (preferably dense) datasets are also good targets for GPU
acceleration. Currently, we construct histograms for dense features on GPU
and construct histograms for sparse features on CPU simultaneously. The
parameter `sparse_threshold` can be used to balance the work on GPU and CPU.

The following configuration can be used for training these large-scale
datasets on GPU:

```
max_bin = 64
num_leaves = 255
num_iterations = 500
learning_rate = 0.1
tree_learner = gpu
task = train
is_train_metric = false
min_data_in_leaf = 1
min_sum_hessian_in_leaf = 100
ndcg_eval_at = 1,3,5,10

gpu_platform_id = 0
gpu_device_id = 0
num_thread = 28
```

The last three parameters should be customized based on your machine configuration;
`num_thread` should match the total number of cores in your system, `gpu_platform_id`
and `gpu_device_id` select the GPU to use. If you have a hybrid GPU setting,
make sure to select the high-performance discrete GPU, not the integrated GPU.
The OpenCL platform ID and device ID can be looked up using the `clinfo` utility.

Examples
--------------------------

The following example shows how to run the `higgs` dataset with GPU acceleration.

```
# build LightGBM with GPU support enabled
git clone --recursive https://github.com/huanzhang12/lightgbm-gpu.git
cd lightgbm-gpu
mkdir build ; cd build
cmake -DUSE_GPU=1 .. 
make -j$(nproc) 
# Executable lightgbm should be generated here
cd ..
# Now clone the LightGBM benchmark repo for data preparation
git clone https://github.com/guolinke/boosting_tree_benchmarks.git
cd boosting_tree_benchmarks/data
# Download data and unzip
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
gunzip HIGGS.csv.gz
# Prepare the dataset. This will take some time. At the same time you can read how to prepare other datasets.
cat readme.md && python higgs2libsvm.py
cd ../..
ln -s boosting_tree_benchmarks/data/higgs.train
ln -s boosting_tree_benchmarks/data/higgs.test
# Now we have higgs.train and higgs.test ready
# Generate a configuration file. Remember to check GPU platform ID and device ID if you have multiple GPUs
cat > lightgbm_gpu.conf <<EOF
max_bin = 64
num_leaves = 255
num_iterations = 500
learning_rate = 0.1
tree_learner = gpu
task = train
is_train_metric = false
min_data_in_leaf = 1
min_sum_hessian_in_leaf = 100
ndcg_eval_at = 1,3,5,10
gpu_platform_id = 0
gpu_device_id = 0
EOF
echo "num_threads=$(nproc)" >> lightgbm_gpu.conf
# Now we are ready to run GPU accelerated LightGBM!
# Accuracy test (make sure to verify the "Using GPU Device" line in output):
./lightgbm config=lightgbm_gpu.conf data=higgs.train valid=higgs.test objective=binary metric=auc
# Speed test:
./lightgbm config=lightgbm_gpu.conf data=higgs.train objective=binary metric=auc
# Accuracy reference (on CPU):
./lightgbm config=lightgbm_gpu.conf tree_learner=serial data=higgs.train valid=higgs.test objective=binary metric=auc
# Speed reference (on CPU):
./lightgbm config=lightgbm_gpu.conf tree_learner=serial data=higgs.train objective=binary metric=auc
```

