/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "new_cuda_utils.hpp"

namespace LightGBM {

void SynchronizeCUDADevice() {
  CUDASUCCESS_OR_FATAL(cudaDeviceSynchronize());
}

void PrintLastCUDAError() {
  const char* error_name = cudaGetErrorName(cudaGetLastError());
  Log::Warning(error_name);
}

}  // namespace LightGBM

#endif  // USE_CUDA
