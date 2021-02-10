/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_CUDA_CUDA_UTILS_H_
#define LIGHTGBM_CUDA_CUDA_UTILS_H_

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDASUCCESS_OR_FATAL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    LightGBM::Log::Fatal("[CUDA] %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#endif  // USE_CUDA

#endif  // LIGHTGBM_CUDA_CUDA_UTILS_H_
