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

template <typenamce T>
__device__ ReduceSum(T* values, size_t n);

template <typenamce T>
__device__ ReduceMax(T* values, size_t n);

template <typenamce T>
__device__ PrefixSum(T* values, size_t n);

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
