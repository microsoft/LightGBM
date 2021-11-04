/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_utils.h>

#ifdef USE_CUDA

namespace LightGBM {

void SynchronizeCUDADevice(const char* file, const int line) {
  gpuAssert(cudaDeviceSynchronize(), file, line);
}

void PrintLastCUDAError() {
  const char* error_name = cudaGetErrorName(cudaGetLastError());
  Log::Warning(error_name); 
}

}  // namespace LightGBM

#endif  // USE_CUDA
