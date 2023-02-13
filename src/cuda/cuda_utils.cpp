/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_utils.h>

namespace LightGBM {

void SynchronizeCUDADevice(const char* file, const int line) {
  gpuAssert(cudaDeviceSynchronize(), file, line);
}

void PrintLastCUDAError() {
  const char* error_name = cudaGetErrorName(cudaGetLastError());
  Log::Fatal(error_name);
}

void SetCUDADevice(int gpu_device_id, const char* file, int line) {
  int cur_gpu_device_id = 0;
  CUDASUCCESS_OR_FATAL_OUTER(cudaGetDevice(&cur_gpu_device_id));
  if (cur_gpu_device_id != gpu_device_id) {
    CUDASUCCESS_OR_FATAL_OUTER(cudaSetDevice(gpu_device_id));
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
