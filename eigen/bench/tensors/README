The tensor benchmark suite is made of several parts.

The first part is a generic suite, in which each benchmark comes in 2 flavors: one that runs on CPU, and one that runs on GPU.

To compile the floating point CPU benchmarks, simply call:
g++ tensor_benchmarks_cpu.cc benchmark_main.cc -I ../../ -std=c++11 -O3 -DNDEBUG -pthread -mavx -o benchmarks_cpu

To compile the floating point GPU benchmarks, simply call:
nvcc tensor_benchmarks_gpu.cu benchmark_main.cc -I ../../ -std=c++11 -O2 -DNDEBUG -use_fast_math -ftz=true -arch compute_35 -o benchmarks_gpu

We also provide a version of the generic GPU tensor benchmarks that uses half floats (aka fp16) instead of regular floats. To compile these benchmarks, simply call the command line below. You'll need a recent GPU that supports compute capability 5.3 or higher to run them and nvcc 7.5 or higher to compile the code.
nvcc tensor_benchmarks_fp16_gpu.cu benchmark_main.cc -I ../../ -std=c++11 -O2 -DNDEBUG -use_fast_math -ftz=true -arch compute_53 -o benchmarks_fp16_gpu

To compile and run the benchmark for SYCL, using ComputeCpp, simply run the
following commands:
1. export COMPUTECPP_PACKAGE_ROOT_DIR={PATH TO COMPUTECPP ROOT DIRECTORY}
2. bash eigen_sycl_bench.sh

Last but not least, we also provide a suite of benchmarks to measure the scalability of the contraction code on CPU. To compile these benchmarks, call
g++ contraction_benchmarks_cpu.cc benchmark_main.cc -I ../../ -std=c++11 -O3 -DNDEBUG -pthread -mavx -o benchmarks_cpu
