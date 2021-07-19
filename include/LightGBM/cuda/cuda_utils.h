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

#include <LightGBM/utils/log.h>

namespace LightGBM {

#define CUDASUCCESS_OR_FATAL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    LightGBM::Log::Fatal("[CUDA] %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define CUDASUCCESS_OR_FATAL_OUTER(ans) { gpuAssert((ans), file, line); }

template <typename T>
void AllocateCUDAMemoryOuter(T** out_ptr, size_t size, const char* file, const int line) {
  void* tmp_ptr = nullptr;
  CUDASUCCESS_OR_FATAL_OUTER(cudaMalloc(&tmp_ptr, size * sizeof(T)));
  *out_ptr = reinterpret_cast<T*>(tmp_ptr);
}

template <typename T>
void CopyFromHostToCUDADeviceOuter(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyHostToDevice));
}

template <typename T>
void CopyFromHostToCUDADeviceAsyncOuter(T* dst_ptr, const T* src_ptr, size_t size, cudaStream_t stream, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpyAsync(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyHostToDevice, stream));
}

template <typename T>
void InitCUDAMemoryFromHostMemoryOuter(T** dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  AllocateCUDAMemoryOuter<T>(dst_ptr, size, file, line);
  CopyFromHostToCUDADeviceOuter<T>(*dst_ptr, src_ptr, size, file, line);
}

template <typename T>
void InitCUDAValueFromConstantOuter(T** dst_ptr, const T value, const char* file, const int line) {
  AllocateCUDAMemoryOuter<T>(1, dst_ptr, file, line);
  CopyFromHostToCUDADeviceOuter<T>(*dst_ptr, &value, 1, file, line);
}

template <typename T>
void CopyFromCUDADeviceToHostOuter(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToHost));
}

template <typename T>
void CopyFromCUDADeviceToHostAsyncOuter(T* dst_ptr, const T* src_ptr, size_t size, cudaStream_t stream, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpyAsync(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void CopyFromCUDADeviceToCUDADeviceOuter(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToDevice));
}

template <typename T>
void CopyFromCUDADeviceToCUDADeviceAsyncOuter(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpyAsync(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToDevice));
}

void SynchronizeCUDADeviceOuter(const char* file, const int line);

template <typename T>
void SetCUDAMemoryOuter(T* dst_ptr, int value, size_t size, const char* file, const int line) {
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemset(reinterpret_cast<void*>(dst_ptr), value, size * sizeof(T)));
}

void PrintLastCUDAErrorOuter(const char* /*file*/, const int /*line*/);

template <typename T>
void DeallocateCUDAMemoryOuter(T** ptr, const char* file, const int line) {
  CUDASUCCESS_OR_FATAL_OUTER(cudaFree(reinterpret_cast<void*>(*ptr)));
  *ptr = nullptr;
}

}

#endif  // USE_CUDA

#endif  // LIGHTGBM_CUDA_CUDA_UTILS_H_
