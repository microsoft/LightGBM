/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_utils.h>

//#ifdef USE_CUDA

namespace LightGBM {

void SynchronizeCUDADeviceOuter(const char* file, const int line) {
  CUDASUCCESS_OR_FATAL_OUTER(cudaDeviceSynchronize());
}

void PrintLastCUDAErrorOuter(const char* /*file*/, const int /*line*/) {
  const char* error_name = cudaGetErrorName(cudaGetLastError());
  Log::Warning(error_name); 
}

}  // namespace LightGBM

//#endif  // USE_CUDA
