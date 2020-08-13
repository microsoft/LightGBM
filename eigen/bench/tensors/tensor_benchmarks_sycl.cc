#ifdef EIGEN_USE_SYCL

#include <CL/sycl.hpp>
#include <iostream>

#include "tensor_benchmarks.h"

cl::sycl::gpu_selector selector;
Eigen::QueueInterface queue(selector);
#define BM_FuncWithInput2DimsGPU(FUNC, D1, D2)                      \
  static void BM_##FUNC##_##D1##x##D2(int iters, int N) {           \
    StopBenchmarkTiming();                                          \
    Eigen::SyclDevice device(&queue);                               \
    BenchmarkSuite<Eigen::SyclDevice, float> suite(device, D1, D2); \
    suite.FUNC(iters);                                              \
  }                                                                 \
  BENCHMARK_RANGE(BM_##FUNC##_##D1##x##D2, 10, 10);

BM_FuncWithInput2DimsGPU(rowReduction, 256, 100352);
BM_FuncWithInput2DimsGPU(rowReduction, 64, 100352);
BM_FuncWithInput2DimsGPU(rowReduction, 512, 25088);
BM_FuncWithInput2DimsGPU(rowReduction, 128, 25088);
BM_FuncWithInput2DimsGPU(rowReduction, 102, 6272);
BM_FuncWithInput2DimsGPU(rowReduction, 256, 6272);
BM_FuncWithInput2DimsGPU(rowReduction, 204, 1568);
BM_FuncWithInput2DimsGPU(rowReduction, 512, 1568);
BM_FuncWithInput2DimsGPU(rowReduction, 1024, 1568);
BM_FuncWithInput2DimsGPU(rowReduction, 2048, 1568);

BM_FuncWithInput2DimsGPU(colReduction, 100352, 256);
BM_FuncWithInput2DimsGPU(colReduction, 100352, 64);
BM_FuncWithInput2DimsGPU(colReduction, 25088, 512);
BM_FuncWithInput2DimsGPU(colReduction, 6272, 102);
BM_FuncWithInput2DimsGPU(colReduction, 25088, 128);
BM_FuncWithInput2DimsGPU(colReduction, 6272, 256);
BM_FuncWithInput2DimsGPU(colReduction, 1568, 204);
BM_FuncWithInput2DimsGPU(colReduction, 1568, 512);
BM_FuncWithInput2DimsGPU(colReduction, 1568, 1024);
BM_FuncWithInput2DimsGPU(colReduction, 1568, 2048);
BM_FuncWithInput2DimsGPU(fullReduction, 1001, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 2050048, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 2097152, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 2048, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 262144, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 256, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 589824, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 1024, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 524288, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 512, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 2359296, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 1048576, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 131072, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 16384, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 9408, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 64, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 4096, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 36864, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 32768, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 128, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 147456, 1);
BM_FuncWithInput2DimsGPU(fullReduction, 65536, 1);
#define BM_FuncGPU(FUNC)                                       \
  static void BM_##FUNC(int iters, int N) {                    \
    StopBenchmarkTiming();                                     \
    Eigen::SyclDevice device(&queue);                          \
    BenchmarkSuite<Eigen::SyclDevice, float> suite(device, N); \
    suite.FUNC(iters);                                         \
  }                                                            \
  BENCHMARK_RANGE(BM_##FUNC, 10, 5000);

BM_FuncGPU(rowReduction);
BM_FuncGPU(colReduction);
BM_FuncGPU(fullReduction);

BM_FuncGPU(memcpy);
BM_FuncGPU(typeCasting);
BM_FuncGPU(random);
BM_FuncGPU(slicing);
BM_FuncGPU(rowChip);
BM_FuncGPU(colChip);
BM_FuncGPU(shuffling);
BM_FuncGPU(padding);
BM_FuncGPU(striding);
BM_FuncGPU(broadcasting);
BM_FuncGPU(coeffWiseOp);
BM_FuncGPU(algebraicFunc);
BM_FuncGPU(transcendentalFunc);
// Contractions
#define BM_FuncWithInputDimsGPU(FUNC, D1, D2, D3)                       \
  static void BM_##FUNC##_##D1##x##D2##x##D3(int iters, int N) {        \
    StopBenchmarkTiming();                                              \
    Eigen::SyclDevice device(&queue);                                   \
    BenchmarkSuite<Eigen::SyclDevice, float> suite(device, D1, D2, D3); \
    suite.FUNC(iters);                                                  \
  }                                                                     \
  BENCHMARK_RANGE(BM_##FUNC##_##D1##x##D2##x##D3, 10, 5000);

BM_FuncWithInputDimsGPU(contraction, N, N, N);
BM_FuncWithInputDimsGPU(contraction, 64, N, N);
BM_FuncWithInputDimsGPU(contraction, N, 64, N);
BM_FuncWithInputDimsGPU(contraction, N, N, 64);

BM_FuncWithInputDimsGPU(contractionRowMajor, N, N, N);
BM_FuncWithInputDimsGPU(contractionRowMajor, 64, N, N);
BM_FuncWithInputDimsGPU(contractionRowMajor, N, 64, N);
BM_FuncWithInputDimsGPU(contractionRowMajor, N, N, 64);

BM_FuncWithInputDimsGPU(contractionRowMajorAT, N, N, N);
BM_FuncWithInputDimsGPU(contractionRowMajorAT, 64, N, N);
BM_FuncWithInputDimsGPU(contractionRowMajorAT, N, 64, N);
BM_FuncWithInputDimsGPU(contractionRowMajorAT, N, N, 64);

BM_FuncWithInputDimsGPU(contractionRowMajorBT, N, N, N);
BM_FuncWithInputDimsGPU(contractionRowMajorBT, 64, N, N);
BM_FuncWithInputDimsGPU(contractionRowMajorBT, N, 64, N);
BM_FuncWithInputDimsGPU(contractionRowMajorBT, N, N, 64);


BM_FuncWithInputDimsGPU(contractionRowMajorABT, N, N, N);
BM_FuncWithInputDimsGPU(contractionRowMajorABT, 64, N, N);
BM_FuncWithInputDimsGPU(contractionRowMajorABT, N, 64, N);
BM_FuncWithInputDimsGPU(contractionRowMajorABT, N, N, 64);

// Convolutions
#define BM_FuncWithKernelDimsGPU(FUNC, DIM1, DIM2)             \
  static void BM_##FUNC##_##DIM1##x##DIM2(int iters, int N) {  \
    StopBenchmarkTiming();                                     \
    Eigen::SyclDevice device(&queue);                          \
    BenchmarkSuite<Eigen::SyclDevice, float> suite(device, N); \
    suite.FUNC(iters, DIM1, DIM2);                             \
  }                                                            \
  BENCHMARK_RANGE(BM_##FUNC##_##DIM1##x##DIM2, 128, 5000);

BM_FuncWithKernelDimsGPU(convolution, 7, 1);
BM_FuncWithKernelDimsGPU(convolution, 1, 7);
BM_FuncWithKernelDimsGPU(convolution, 7, 4);
BM_FuncWithKernelDimsGPU(convolution, 4, 7);
BM_FuncWithKernelDimsGPU(convolution, 7, 64);
BM_FuncWithKernelDimsGPU(convolution, 64, 7);
#endif
