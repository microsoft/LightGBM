/*!
 * Copyright (c) 2020-2021 IBM Corporation, Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_CUDA_CUDA_UTILS_H_
#define LIGHTGBM_CUDA_CUDA_UTILS_H_

#if defined(USE_CUDA) || defined(USE_CUDA_EXP)
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#endif  // USE_CUDA || USE_CUDA_EXP

#ifdef USE_CUDA_EXP
#include <LightGBM/utils/log.h>
#include <vector>
#endif  // USE_CUDA_EXP

namespace LightGBM {

#if defined(USE_CUDA) || defined(USE_CUDA_EXP)
#define CUDASUCCESS_OR_FATAL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    LightGBM::Log::Fatal("[CUDA] %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#endif  // USE_CUDA || USE_CUDA_EXP

#ifdef USE_CUDA_EXP
#define CUDASUCCESS_OR_FATAL_OUTER(ans) { gpuAssert((ans), file, line); }

void SetCUDADevice(int gpu_device_id, const char* file, int line);

template <typename T>
void AllocateCUDAMemory(T** out_ptr, size_t size, const char* file, const int line) {
  void* tmp_ptr = nullptr;
  CUDASUCCESS_OR_FATAL_OUTER(cudaMalloc(&tmp_ptr, size * sizeof(T)));
  *out_ptr = reinterpret_cast<T*>(tmp_ptr);
}

template <typename T>
void CopyFromHostToCUDADevice(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyHostToDevice));
}

template <typename T>
void InitCUDAMemoryFromHostMemory(T** dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  AllocateCUDAMemory<T>(dst_ptr, size, file, line);
  CopyFromHostToCUDADevice<T>(*dst_ptr, src_ptr, size, file, line);
}

template <typename T>
void CopyFromCUDADeviceToHost(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToHost));
}

template <typename T>
void CopyFromCUDADeviceToHostAsync(T* dst_ptr, const T* src_ptr, size_t size, cudaStream_t stream, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpyAsync(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void CopyFromCUDADeviceToCUDADevice(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToDevice));
}

template <typename T>
void CopyFromCUDADeviceToCUDADeviceAsync(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemcpyAsync(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToDevice));
}

void SynchronizeCUDADevice(const char* file, const int line);

template <typename T>
void SetCUDAMemory(T* dst_ptr, int value, size_t size, const char* file, const int line) {
  CUDASUCCESS_OR_FATAL_OUTER(cudaMemset(reinterpret_cast<void*>(dst_ptr), value, size * sizeof(T)));
  SynchronizeCUDADevice(file, line);
}

template <typename T>
void DeallocateCUDAMemory(T** ptr, const char* file, const int line) {
  if (*ptr != nullptr) {
    CUDASUCCESS_OR_FATAL_OUTER(cudaFree(reinterpret_cast<void*>(*ptr)));
    *ptr = nullptr;
  }
}

void PrintLastCUDAError();

template <typename T>
class CUDAVector {
 public:
  CUDAVector() {
    size_ = 0;
    data_ = nullptr;
  }

  explicit CUDAVector(size_t size) {
    size_ = size;
    AllocateCUDAMemory<T>(&data_, size_, __FILE__, __LINE__);
  }

  void Resize(size_t size) {
    if (size == 0) {
      Clear();
    }
    T* new_data = nullptr;
    AllocateCUDAMemory<T>(&new_data, size, __FILE__, __LINE__);
    if (size_ > 0 && data_ != nullptr) {
      CopyFromCUDADeviceToCUDADevice<T>(new_data, data_, size, __FILE__, __LINE__);
    }
    DeallocateCUDAMemory<T>(&data_, __FILE__, __LINE__);
    data_ = new_data;
    size_ = size;
  }

  void Clear() {
    if (size_ > 0 && data_ != nullptr) {
      DeallocateCUDAMemory<T>(&data_, __FILE__, __LINE__);
    }
    size_ = 0;
  }

  void PushBack(const T* values, size_t len) {
    T* new_data = nullptr;
    AllocateCUDAMemory<T>(&new_data, size_ + len, __FILE__, __LINE__);
    if (size_ > 0 && data_ != nullptr) {
      CopyFromCUDADeviceToCUDADevice<T>(new_data, data_, size_, __FILE__, __LINE__);
    }
    CopyFromCUDADeviceToCUDADevice<T>(new_data + size_, values, len, __FILE__, __LINE__);
    DeallocateCUDAMemory<T>(&data_, __FILE__, __LINE__);
    size_ += len;
    data_ = new_data;
  }

  size_t Size() {
    return size_;
  }

  ~CUDAVector() {
    DeallocateCUDAMemory<T>(&data_, __FILE__, __LINE__);
  }

  std::vector<T> ToHost() {
    std::vector<T> host_vector(size_);
    if (size_ > 0 && data_ != nullptr) {
      CopyFromCUDADeviceToHost(host_vector.data(), data_, size_, __FILE__, __LINE__);
    }
    return host_vector;
  }

  T* RawData() const {
    return data_;
  }

  const T* RawDataReadOnly() const {
    return data_;
  }

 private:
  T* data_;
  size_t size_;
};

#endif  // USE_CUDA_EXP

}  // namespace LightGBM

#endif  // LIGHTGBM_CUDA_CUDA_UTILS_H_
