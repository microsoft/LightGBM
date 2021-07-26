/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
#define LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <LightGBM/utils/log.h>

namespace LightGBM {

template <typename T>
__device__ void ReduceSum(T* values, size_t n);

template <typename T>
__device__ void ReduceMax(T* values, size_t n);

template <typename T>
__device__ void PrefixSum(T* values, size_t n);

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
