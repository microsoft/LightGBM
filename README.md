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
  The generic OpenCL ICD packages (for example, Debian package
  `ocl-icd-libopencl1` and `ocl-icd-opencl-dev`) can also be used.

- libboost 1.56 or later (1.61 or later recommended). We use Boost.Compute as
  the interface to GPU, which is part of the Boost library since version 1.61.
  However, since we include the source code of Boost.Compute as a submodule, we
  only require the host has Boost 1.56 or later installed. We also use
  Boost.Align for memory allocation. Boost.Compute requires Boost.System
  and Boost.Filesystem to store offline kernel cache. The following Debian 
  packages should provide necessary Boost libraries: 
  `libboost-dev, libboost-system-dev, libboost-filesystem-dev`.

- CMake 3.2 or later

Currently only building on Linux has been tested, but it should also work with
MinGW on Windows as long as all dependencies are available. To build LightGBM-GPU,
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
trainer should generate (almost) identical results as the CPU trainer, at least
for the first a few iterations.

- `sparse_threshold`: The threshold of zero elements percentage for
  treating a feature as dense feature. When setting to 1, all features are
  processed as dense features (default: 0.8).


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

- NVIDIA Kepler (K80, K40, K20, most GeForce GTX 700 series GPUs) or earlier
  NVIDIA GPUs. They don't support hardware atomic operations in local memory space
  and thus histogram construction will be slow.

- AMD VLIW4-based GPUs, including Radeon HD 6xxx series and earlier GPUs. These
  GPUs have been discontinued for years and are rarely seen nowadays.

Datasets
--------------------------

Datasets HIGGS, Yahoo LTR, Microsoft LTR and Expo that are used for 
[LightGBM benchmarks](https://github.com/Microsoft/LightGBM/wiki/Experiments#parallel-experiment) 
work on GPUs with good speedup. To prepare these datasets, follow the instructions in
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
max_bin = 63
num_leaves = 255
num_iterations = 500
learning_rate = 0.1
tree_learner = gpu
task = train
is_train_metric = false
min_data_in_leaf = 1
min_sum_hessian_in_leaf = 100
ndcg_eval_at = 1,3,5,10
sparse_threshold=1.0

gpu_platform_id = 0
gpu_device_id = 0
num_thread = 28
```

The last three parameters should be customized based on your machine configuration;
`num_thread` should match the total number of cores in your system, `gpu_platform_id`
and `gpu_device_id` select the GPU to use. If you have a hybrid GPU setting,
make sure to select the high-performance discrete GPU, not the integrated GPU.
The OpenCL platform ID and device ID can be looked up using the `clinfo` utility.

How to Achieve Good Speedup on GPU
--------------------------

1. You want to run a few datasets that we have verified with good speedup
   (including Higgs, epsilon, Microsoft Learning to Rank, etc) to ensure your
   setup is correct. Make sure your system is idle (especially when using a
   shared computer) to get accuracy performance measurements. 

2. GPU works best on large scale and dense datasets. If dataset is too small,
   computing it on GPU is inefficient as the data transfer overhead can be
   significant.  For dataset with a mixture of sparse and dense features, you
   can control the `sparse_threshold` parameter to make sure there are enough
   dense features to process on the GPU. If you have categorical features, use
   the `categorical_column` option and input them into LightGBM directly; do
   not convert them into one-hot variables. Make sure to check the run log and
   look at the reported number of sparse and dense features.


3. To get good speedup with GPU, it is suggested to use a smaller number of
   bins.  Setting `max_bin=63` is recommended, as it usually does not
   noticeably affect training accuracy on large datasets, but GPU training can
   be significantly faster than using the default bin size of 255.  For some
   dataset, even using 15 bins is enough (`max_bin=15`); using 15 bins will
   maximize GPU performance. Make sure to check the run log and verify that the
   desired number of bins is used.

4. Try to use single precision training (`gpu_use_dp=false`) when possible,
   because most GPUs (especially NVIDIA consumer GPUs) have poor
   double-precision performance.

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
max_bin = 63
num_leaves = 255
num_iterations = 500
learning_rate = 0.1
tree_learner = gpu
task = train
is_train_metric = false
min_data_in_leaf = 1
min_sum_hessian_in_leaf = 100
ndcg_eval_at = 1,3,5,10
sparse_threshold = 1.0
gpu_platform_id = 0
gpu_device_id = 0
EOF
echo "num_threads=$(nproc)" >> lightgbm_gpu.conf
# Now we are ready to run GPU accelerated LightGBM!
# Accuracy test on GPU (make sure to verify the "Using GPU Device" line in output):
./lightgbm config=lightgbm_gpu.conf data=higgs.train valid=higgs.test objective=binary metric=auc
# Speed test on GPU:
./lightgbm config=lightgbm_gpu.conf data=higgs.train objective=binary metric=auc
# Accuracy reference (on CPU):
./lightgbm config=lightgbm_gpu.conf tree_learner=serial data=higgs.train valid=higgs.test objective=binary metric=auc
# Speed reference (on CPU):
./lightgbm config=lightgbm_gpu.conf tree_learner=serial data=higgs.train objective=binary metric=auc
```

Now let's try the `epsilon` dataset:

```
# assume we are in the same directory as the lightgbm binary
# download the dataset and extract them
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
bunzip2 epsilon_normalized.bz2 
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2
bunzip2 epsilon_normalized.t.bz2
mv epsilon_normalized epsilon.train
mv epsilon_normalized.t epsilon.test
# datasets are ready, now start training
# Accuracy test on GPU (make sure to verify the "Using GPU Device" line in output):
./lightgbm config=lightgbm_gpu.conf data=epsilon.train valid=epsilon.test objective=binary metric=auc
# Speed test on GPU (without calculating validation set AUC after each iteration):
./lightgbm config=lightgbm_gpu.conf data=epsilon.train objective=binary metric=auc
# Accuracy reference (on CPU):
./lightgbm config=lightgbm_gpu.conf tree_learner=serial data=epsilon.train valid=epsilon.test objective=binary metric=auc
# Speed reference (on CPU):
./lightgbm config=lightgbm_gpu.conf tree_learner=serial data=epsilon.train objective=binary metric=auc

```

Try to change the number of bins and see how that affacts training speed:

```
# Speed test on GPU with max_bin size of 15:
./lightgbm config=lightgbm_gpu.conf data=epsilon.train objective=binary metric=auc max_bin=15
# Speed test on GPU with max_bin size of 63:
./lightgbm config=lightgbm_gpu.conf data=epsilon.train objective=binary metric=auc max_bin=63
# Speed test on GPU with max_bin size of 255:
./lightgbm config=lightgbm_gpu.conf data=epsilon.train objective=binary metric=auc max_bin=255
```

Further Reading
--------------------------

If you are interested in more details about our algorithm and benchmarks,
please see our paper:

```
GPU Acceleration for Large-scale Tree Boosting
Huan Zhang, Si Si and Cho-Jui Hsieh, 2017.
```

