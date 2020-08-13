// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef EIGEN_BENCH_CONTRACT_SYCL
#define EIGEN_BENCH_CONTRACT_SYCL
#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
#include <SYCL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>

#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;
std::ofstream out("Result.txt");

std::chrono::time_point<std::chrono::system_clock> get_time(){
  std::chrono::time_point<std::chrono::system_clock> start, end;
  return std::chrono::system_clock::now();
}

template<typename Start, typename End, typename TensorIndex>
void finalizeBenchmark(Start start, End end, TensorIndex m_, TensorIndex k_, TensorIndex n_ , TensorIndex num_iters, std::string name){

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout <<"Kernel Name : " << name << ", M : " << m_ << ",  N : " << n_ << ", K : " << k_ << " GFLOP/s : " <<
  static_cast<float>((static_cast<int64_t>(2) * m_ * n_ * k_ * num_iters)/ elapsed_seconds.count()) * 1e-9 << "\n";
    out <<"Kernel Name : " << name << ", M : " << m_ << ",  N : " << n_ << ", K : " << k_ << " GFLOP/s : " <<
    static_cast<float>((static_cast<int64_t>(2) * m_ * n_ * k_ * num_iters)/ elapsed_seconds.count()) * 1e-9 << "\n";
}

// do a contraction which is equivalent to a matrix multiplication
template<typename T, typename Device, typename TensorIndex>
void contraction(const Device& device_, TensorIndex num_iters, TensorIndex m_, TensorIndex k_, TensorIndex n_) {
  T* a_;
  T* b_;
  T* c_;
  a_ = (T *) device_.allocate(m_ * k_ * sizeof(T));
  b_ = (T *) device_.allocate(k_ * n_ * sizeof(T));
  c_ = (T *) device_.allocate(m_ * n_ * sizeof(T));

  // Initialize the content of the memory pools to prevent asan from
  // complaining.
  device_.memset(a_, 12, m_ * k_ * sizeof(T));
  device_.memset(b_, 23, k_ * n_ * sizeof(T));
  device_.memset(c_, 31, m_ * n_ * sizeof(T));

  Eigen::array<TensorIndex, 2> sizeA;
  sizeA[0] = m_;
  sizeA[1] = k_;
  Eigen::array<TensorIndex, 2> sizeB;
  sizeB[0] = k_;
  sizeB[1] = n_;
  Eigen::array<TensorIndex, 2> sizeC;
  sizeC[0] = m_;
  sizeC[1] = n_;

  const TensorMap<Tensor<T, 2>, Eigen::Aligned> A(a_, sizeA);
  const TensorMap<Tensor<T, 2>, Eigen::Aligned> B(b_, sizeB);
  TensorMap<Tensor<T, 2>, Eigen::Aligned> C(c_, sizeC);

  typedef typename Tensor<T, 2>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims;
  dims[0] = DimPair(1, 0);
#ifdef EIGEN_USE_SYCL // warmup for sycl
  for (int iter = 0; iter < 10; ++iter) {
    C.device(device_) = A.contract(B, dims);
   }
#endif
  auto start = get_time();
  for (int iter = 0; iter < num_iters; ++iter) {
    C.device(device_) = A.contract(B, dims);
  }
 auto end = get_time();
  // Record the number of FLOPs executed per second (size_ multiplications and
  // additions for each value in the resulting tensor)
  finalizeBenchmark(start, end, m_, k_, n_, num_iters, "contraction");
  device_.deallocate(a_);
  device_.deallocate(b_);
  device_.deallocate(c_);
  device_.synchronize();
}



// do a contraction which is equivalent to a matrix multiplication
template<typename T, typename Device, typename TensorIndex>
void contractionRowMajor(const Device& device_, TensorIndex num_iters, TensorIndex m_, TensorIndex k_, TensorIndex n_) {
  T* a_;
  T* b_;
  T* c_;
  a_ = (T *) device_.allocate(m_ * k_ * sizeof(T));
  b_ = (T *) device_.allocate(k_ * n_ * sizeof(T));
  c_ = (T *) device_.allocate(m_ * n_ * sizeof(T));

  // Initialize the content of the memory pools to prevent asan from
  // complaining.
  device_.memset(a_, 12, m_ * k_ * sizeof(T));
  device_.memset(b_, 23, k_ * n_ * sizeof(T));
  device_.memset(c_, 31, m_ * n_ * sizeof(T));

  Eigen::array<TensorIndex, 2> sizeA;
  sizeA[0] = m_;
  sizeA[1] = k_;
  Eigen::array<TensorIndex, 2> sizeB;
  sizeB[0] = k_;
  sizeB[1] = n_;
  Eigen::array<TensorIndex, 2> sizeC;
  sizeC[0] = m_;
  sizeC[1] = n_;

  const TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> A(a_, sizeA);
  const TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> B(b_, sizeB);
  TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> C(c_, sizeC);

  typedef typename Tensor<T, 2>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims;
  dims[0] = DimPair(1, 0);
#ifdef EIGEN_USE_SYCL // warmup for sycl
  for (int iter = 0; iter < 10; ++iter) {
    C.device(device_) = A.contract(B, dims);
   }
#endif
  auto start = get_time();
  for (int iter = 0; iter < num_iters; ++iter) {
    C.device(device_) = A.contract(B, dims);
  }
  auto end = get_time();
  // Record the number of FLOPs executed per second (size_ multiplications and
  // additions for each value in the resulting tensor)
  finalizeBenchmark(start, end, m_, k_, n_, num_iters, "contractionRowMajor");
  device_.deallocate(a_);
  device_.deallocate(b_);
  device_.deallocate(c_);
  device_.synchronize();
}


template<typename T, typename Device, typename TensorIndex>
void contractionAT(const Device& device_, TensorIndex num_iters, TensorIndex m_, TensorIndex k_, TensorIndex n_) {
  T* a_;
  T* b_;
  T* c_;
  a_ = (T *) device_.allocate(m_ * k_ * sizeof(T));
  b_ = (T *) device_.allocate(k_ * n_ * sizeof(T));
  c_ = (T *) device_.allocate(m_ * n_ * sizeof(T));

  // Initialize the content of the memory pools to prevent asan from
  // complaining.
  device_.memset(a_, 12, m_ * k_ * sizeof(T));
  device_.memset(b_, 23, k_ * n_ * sizeof(T));
  device_.memset(c_, 31, m_ * n_ * sizeof(T));
  Eigen::array<TensorIndex, 2> sizeA;
  sizeA[0] = k_;
  sizeA[1] = m_;
  Eigen::array<TensorIndex, 2> sizeB;
  sizeB[0] = k_;
  sizeB[1] = n_;
  Eigen::array<TensorIndex, 2> sizeC;
  sizeC[0] = m_;
  sizeC[1] = n_;

  const TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> A(a_, sizeA);
  const TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> B(b_, sizeB);
  TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> C(c_, sizeC);

  typedef typename Tensor<T, 2>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims;
  dims[0] = DimPair(0, 0);
#ifdef EIGEN_USE_SYCL // warmup for sycl
  for (int iter = 0; iter < 10; ++iter) {
    C.device(device_) = A.contract(B, dims);
   }
#endif
  auto start = get_time();
  for (int iter = 0; iter < num_iters; ++iter) {
    C.device(device_) = A.contract(B, dims);
  }
  auto end = get_time();
  // Record the number of FLOPs executed per second (size_ multiplications and
  // additions for each value in the resulting tensor)
  finalizeBenchmark(start, end, m_, k_, n_, num_iters, "contractionAT");
  device_.deallocate(a_);
  device_.deallocate(b_);
  device_.deallocate(c_);
  device_.synchronize();

}

template<typename T, typename Device, typename TensorIndex>
void contractionBT(const Device& device_, TensorIndex num_iters, TensorIndex m_, TensorIndex k_, TensorIndex n_) {
  T* a_;
  T* b_;
  T* c_;
  a_ = (T *) device_.allocate(m_ * k_ * sizeof(T));
  b_ = (T *) device_.allocate(k_ * n_ * sizeof(T));
  c_ = (T *) device_.allocate(m_ * n_ * sizeof(T));

  // Initialize the content of the memory pools to prevent asan from
  // complaining.
  device_.memset(a_, 12, m_ * k_ * sizeof(T));
  device_.memset(b_, 23, k_ * n_ * sizeof(T));
  device_.memset(c_, 31, m_ * n_ * sizeof(T));

  Eigen::array<TensorIndex, 2> sizeA;
  sizeA[0] = m_;
  sizeA[1] = k_;
  Eigen::array<TensorIndex, 2> sizeB;
  sizeB[0] = n_;
  sizeB[1] = k_;
  Eigen::array<TensorIndex, 2> sizeC;
  sizeC[0] = m_;
  sizeC[1] = n_;

  const TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> A(a_, sizeA);
  const TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> B(b_, sizeB);
  TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> C(c_, sizeC);

  typedef typename Tensor<T, 2>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims;
  dims[0] = DimPair(1, 1);
#ifdef EIGEN_USE_SYCL // warmup for sycl
  for (int iter = 0; iter < 10; ++iter) {
    C.device(device_) = A.contract(B, dims);
   }
#endif
  auto start = get_time();
  for (int iter = 0; iter < num_iters; ++iter) {
    C.device(device_) = A.contract(B, dims);
  }
  auto end = get_time();
  // Record the number of FLOPs executed per second (size_ multiplications and
  // additions for each value in the resulting tensor)
  finalizeBenchmark(start, end, m_, k_, n_, num_iters, "contractionBT");
  device_.deallocate(a_);
  device_.deallocate(b_);
  device_.deallocate(c_);
  device_.synchronize();

}

template<typename T, typename Device, typename TensorIndex>
void contractionABT(const Device& device_, TensorIndex num_iters, TensorIndex m_, TensorIndex k_, TensorIndex n_) {
  T* a_;
  T* b_;
  T* c_;
  a_ = (T *) device_.allocate(m_ * k_ * sizeof(T));
  b_ = (T *) device_.allocate(k_ * n_ * sizeof(T));
  c_ = (T *) device_.allocate(m_ * n_ * sizeof(T));

  // Initialize the content of the memory pools to prevent asan from
  // complaining.
  device_.memset(a_, 12, m_ * k_ * sizeof(T));
  device_.memset(b_, 23, k_ * n_ * sizeof(T));
  device_.memset(c_, 31, m_ * n_ * sizeof(T));

  Eigen::array<TensorIndex, 2> sizeA;
  sizeA[0] = k_;
  sizeA[1] = m_;
  Eigen::array<TensorIndex, 2> sizeB;
  sizeB[0] = n_;
  sizeB[1] = k_;
  Eigen::array<TensorIndex, 2> sizeC;
  sizeC[0] = m_;
  sizeC[1] = n_;

  const TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> A(a_, sizeA);
  const TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> B(b_, sizeB);
  TensorMap<Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> C(c_, sizeC);

  typedef typename Tensor<T, 2>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims;
  dims[0] = DimPair(0, 1);
#ifdef EIGEN_USE_SYCL // warmup for sycl
  for (int iter = 0; iter < 10; ++iter) {
    C.device(device_) = A.contract(B, dims);
   }
#endif
  auto start = get_time();
  for (int iter = 0; iter < num_iters; ++iter) {
    C.device(device_) = A.contract(B, dims);
  }
  auto end = get_time();
  // Record the number of FLOPs executed per second (size_ multiplications and
  // additions for each value in the resulting tensor)
  finalizeBenchmark(start, end, m_, k_, n_, num_iters, "contractionABT");
  device_.deallocate(a_);
  device_.deallocate(b_);
  device_.deallocate(c_);
  device_.synchronize();
}

int main() {
  cl::sycl::gpu_selector selector;
  Eigen::QueueInterface queue(selector);
  Eigen::SyclDevice device(&queue);
  int64_t num_iters =20;
  for(int64_t m = 32; m <= 4096; m *= 2)
    for(int64_t k = 32; k <= 4096; k *= 2)
      for(int64_t n = 32; n <= 4096; n*= 2){
        (contraction<float>(device, num_iters, m, k, n));
        (contractionRowMajor<float>(device, num_iters, m, k, n));
        (contractionAT<float>(device, num_iters, m, k, n));
        (contractionBT<float>(device, num_iters, m, k, n));
        (contractionABT<float>(device, num_iters, m, k, n));
      }
  return 0;
  }

#endif // EIGEN_BENCH_CONTRACT_SYCL
