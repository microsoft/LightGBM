/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_NEW_CUDA_UTILS_HPP_
#define LIGHTGBM_NEW_CUDA_UTILS_HPP_

#ifdef USE_CUDA

#include <LightGBM/utils/log.h>
#include <LightGBM/meta.h>
#include <LightGBM/cuda/cuda_utils.h>

namespace LightGBM {

template <typename T>
void AllocateCUDAMemory(size_t size, T** out_ptr) {
  void* tmp_ptr = nullptr;
  CUDASUCCESS_OR_FATAL(cudaMalloc(&tmp_ptr, size * sizeof(T)));
  *out_ptr = reinterpret_cast<T*>(tmp_ptr);
}

template <typename T>
void CopyFromHostToCUDADevice(T* dst_ptr, const T* src_ptr, size_t size) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL(cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyHostToDevice));
}

template <typename T>
void CopyFromCUDADeviceToHost(T* dst_ptr, const T* src_ptr, size_t size) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL(cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToHost));
}

template <typename T>
void CopyFromCUDADeviceToCUDADevice(T* dst_ptr, const T* src_ptr, size_t size) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL(cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToDevice));
}

void SynchronizeCUDADevice();

template <typename T>
void SetCUDAMemory(T* dst_ptr, int value, size_t size) {
  CUDASUCCESS_OR_FATAL(cudaMemset(reinterpret_cast<void*>(dst_ptr), value, size * sizeof(T)));
}

void PrintLastCUDAError();

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_NEW_CUDA_UTILS_HPP_
