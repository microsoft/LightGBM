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

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
#define EIGEN_USE_SYCL

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>

#include "main.h"

#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void static test_sycl_contraction(const Device &sycl_device, IndexType m_size,
                                  IndexType k_size, IndexType n_size) {
  typedef typename Tensor<DataType, 1, DataLayout, IndexType>::DimensionPair
      DimPair;
  static const DataType error_threshold = DataType(1e-4);
  // with these dimensions, the output has 300 * 140 elements, which is
  // more than 30 * 1024, which is the number of threads in blocks on
  // a 15 SM GK110 GPU
  Tensor<DataType, 2, DataLayout, IndexType> t_left(m_size, k_size);
  Tensor<DataType, 2, DataLayout, IndexType> t_right(k_size, n_size);
  Tensor<DataType, 2, DataLayout, IndexType> t_result(m_size, n_size);
  Tensor<DataType, 2, DataLayout, IndexType> t_result_gpu(m_size, n_size);
  Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
  Eigen::array<IndexType, 2> left_dims = {{m_size, k_size}};
  Eigen::array<IndexType, 2> right_dims = {{k_size, n_size}};
  Eigen::array<IndexType, 2> result_dims = {{m_size, n_size}};

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size() * sizeof(DataType);
  std::size_t t_right_bytes = t_right.size() * sizeof(DataType);
  std::size_t t_result_bytes = t_result.size() * sizeof(DataType);

  DataType *d_t_left =
      static_cast<DataType *>(sycl_device.allocate(t_left_bytes));
  DataType *d_t_right =
      static_cast<DataType *>(sycl_device.allocate(t_right_bytes));
  DataType *d_t_result =
      static_cast<DataType *>(sycl_device.allocate(t_result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_left(d_t_left, left_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_right(d_t_right, right_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_result(d_t_result, result_dims);

  sycl_device.memcpyHostToDevice(d_t_left, t_left.data(), t_left_bytes);
  sycl_device.memcpyHostToDevice(d_t_right, t_right.data(), t_right_bytes);

  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result,
                                 t_result_bytes);

  t_result = t_left.contract(t_right, dims);

  for (IndexType i = 0; i < t_result.size(); i++) {
    if (static_cast<DataType>(std::fabs(static_cast<DataType>(
            t_result(i) - t_result_gpu(i)))) < error_threshold) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i),
                                  error_threshold)) {
      continue;
    }

    std::cout << "M : " << m_size << ", N : " << n_size << ", K : " << k_size
              << ", mismatch detected at IndexType " << i << ": " << t_result(i)
              << " vs " << t_result_gpu(i) << std::endl;
    VERIFY_IS_APPROX(t_result_gpu(i), t_result(i));
  }
  sycl_device.deallocate(d_t_left);
  sycl_device.deallocate(d_t_right);
  sycl_device.deallocate(d_t_result);
}

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void test_sycl_contraction_m(const Device &sycl_device) {
  for (IndexType k = 32; k < 256; k++) {
    test_sycl_contraction<DataLayout, DataType, IndexType>(sycl_device, k, 128,
                                                           128);
  }
}

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void test_sycl_contraction_k(const Device &sycl_device) {
  for (IndexType k = 32; k < 256; k++) {
    test_sycl_contraction<DataLayout, DataType, IndexType>(sycl_device, 128, k,
                                                           128);
  }
}

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void test_sycl_contraction_n(const Device &sycl_device) {
  for (IndexType k = 32; k < 256; k++) {
    test_sycl_contraction<DataLayout, DataType, IndexType>(sycl_device, 128,
                                                           128, k);
  }
}

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void test_sycl_contraction_sizes(const Device &sycl_device) {
  IndexType m_sizes[] = {31,  39,  63,  64,  65,   127,  129, 255,
                         257, 511, 512, 513, 1023, 1024, 1025};

  IndexType n_sizes[] = {31,  39,  63,  64,  65,   127,  129, 255,
                         257, 511, 512, 513, 1023, 1024, 1025};

  IndexType k_sizes[] = {31,  39,  63,  64,  65,  95,   96,   127, 129,
                         255, 257, 511, 512, 513, 1023, 1024, 1025};

  for (IndexType i = 0; i < 15; i++) {
    for (IndexType j = 0; j < 15; j++) {
      for (IndexType k = 0; k < 17; k++) {
        test_sycl_contraction<DataLayout, DataType, IndexType>(
            sycl_device, m_sizes[i], n_sizes[j], k_sizes[k]);
      }
    }
  }
}

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void static test_no_out_of_bounds(const Device &sycl_device, IndexType m_size,
                                  IndexType k_size, IndexType n_size) {
  typedef typename Tensor<DataType, 1, DataLayout, IndexType>::DimensionPair
      DimPair;
  static const DataType error_threshold = DataType(1e-4);
  Tensor<DataType, 2, DataLayout, IndexType> t_left(m_size, k_size);
  Tensor<DataType, 2, DataLayout, IndexType> t_right(k_size, n_size);
  Tensor<DataType, 2, DataLayout, IndexType> t_result(m_size, n_size);

  Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
  Eigen::array<IndexType, 2> left_dims = {{m_size, k_size}};
  Eigen::array<IndexType, 2> right_dims = {{k_size, n_size}};
  Eigen::array<IndexType, 2> result_dims = {{m_size, n_size}};

  t_left.setRandom();
  t_right.setRandom();

  // Allocate buffers twice as big to check for invalid read and write
  auto padded_left_size = 2 * t_left.size();
  auto padded_right_size = 2 * t_right.size();
  auto padded_result_size = 2 * t_result.size();

  std::size_t t_left_bytes = padded_left_size * sizeof(DataType);
  std::size_t t_right_bytes = padded_right_size * sizeof(DataType);
  std::size_t t_result_bytes = padded_result_size * sizeof(DataType);

  DataType *d_t_left =
      static_cast<DataType *>(sycl_device.allocate(t_left_bytes));
  DataType *d_t_right =
      static_cast<DataType *>(sycl_device.allocate(t_right_bytes));
  DataType *d_t_result =
      static_cast<DataType *>(sycl_device.allocate(t_result_bytes));

  // TensorMaps are still of the same size than the Tensors
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_left(d_t_left, left_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_right(d_t_right, right_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_result(d_t_result, result_dims);

  // Write nan after the actual buffer to propagate nans everywhere in case of
  // invalid reads
  DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  auto host_left_data = new DataType[padded_left_size];
  std::copy_n(t_left.data(), t_left.size(), host_left_data);
  std::fill_n(host_left_data + t_left.size(), t_left.size(), nan);
  auto host_right_data = new DataType[padded_right_size];
  std::copy_n(t_right.data(), t_right.size(), host_right_data);
  std::fill_n(host_right_data + t_right.size(), t_right.size(), nan);
  auto host_result_data = new DataType[padded_result_size];
  std::fill_n(host_result_data, padded_result_size, nan);

  sycl_device.memcpyHostToDevice(d_t_left, host_left_data, t_left_bytes);
  sycl_device.memcpyHostToDevice(d_t_right, host_right_data, t_right_bytes);
  sycl_device.memcpyHostToDevice(d_t_result, host_result_data, t_result_bytes);

  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
  sycl_device.memcpyDeviceToHost(host_result_data, d_t_result, t_result_bytes);

  t_result = t_left.contract(t_right, dims);

  for (IndexType i = 0; i < t_result.size(); i++) {
    if (static_cast<DataType>(std::fabs(static_cast<DataType>(
            t_result(i) - host_result_data[i]))) < error_threshold) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), host_result_data[i],
                                  error_threshold)) {
      continue;
    }
    if (std::isnan(host_result_data[i])) {
      std::cout << "M : " << m_size << ", N : " << n_size << ", K : " << k_size
                << ", invalid read detected at IndexType " << i << ": "
                << t_result(i) << " vs " << host_result_data[i] << std::endl;
    } else {
      std::cout << "M : " << m_size << ", N : " << n_size << ", K : " << k_size
                << ", mismatch detected at IndexType " << i << ": "
                << t_result(i) << " vs " << host_result_data[i] << std::endl;
    }
    VERIFY_IS_APPROX(host_result_data[i], t_result(i));
  }
  // Make sure that the rest of the result is still nans
  for (IndexType i = t_result.size(); i < padded_result_size; i++) {
    if (std::isnan(host_result_data[i])) {
      continue;
    }
    std::cout << "M : " << m_size << ", N : " << n_size << ", K : " << k_size
              << ", invalid write detected at IndexType " << i << ": "
              << host_result_data[i] << std::endl;
    VERIFY_IS_APPROX(host_result_data[i], t_result(i));
  }
  sycl_device.deallocate(d_t_left);
  sycl_device.deallocate(d_t_right);
  sycl_device.deallocate(d_t_result);

  delete[] host_left_data;
  delete[] host_right_data;
  delete[] host_result_data;
}

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void test_scalar(const Device &sycl_device, IndexType m_size, IndexType k_size,
                 IndexType n_size) {
  // std::cout << "Testing for (" << m_size << "," << k_size << "," << n_size <<
  // ")" << std::endl;
  // with these dimensions, the output has 300 * 140 elements, which is
  // more than 30 * 1024, which is the number of threads in blocks on
  // a 15 SM GK110 GPU
  typedef typename Tensor<DataType, 1, DataLayout, IndexType>::DimensionPair
      DimPair;
  static const DataType error_threshold = DataType(1e-4);
  Tensor<DataType, 2, DataLayout, IndexType> t_left(m_size, k_size);
  Tensor<DataType, 2, DataLayout, IndexType> t_right(k_size, n_size);
  Tensor<DataType, 0, DataLayout, IndexType> t_result;
  Tensor<DataType, 0, DataLayout, IndexType> t_result_gpu;
  Eigen::array<DimPair, 2> dims = {{DimPair(0, 0), DimPair(1, 1)}};
  Eigen::array<IndexType, 2> left_dims = {{m_size, k_size}};
  Eigen::array<IndexType, 2> right_dims = {{k_size, n_size}};
  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size() * sizeof(DataType);
  std::size_t t_right_bytes = t_right.size() * sizeof(DataType);
  std::size_t t_result_bytes = sizeof(DataType);

  DataType *d_t_left =
      static_cast<DataType *>(sycl_device.allocate(t_left_bytes));
  DataType *d_t_right =
      static_cast<DataType *>(sycl_device.allocate(t_right_bytes));
  DataType *d_t_result =
      static_cast<DataType *>(sycl_device.allocate(t_result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_left(d_t_left, left_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_right(d_t_right, right_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 0, DataLayout, IndexType>>
      gpu_t_result(d_t_result);

  sycl_device.memcpyHostToDevice(d_t_left, t_left.data(), t_left_bytes);
  sycl_device.memcpyHostToDevice(d_t_right, t_right.data(), t_right_bytes);

  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result,
                                 t_result_bytes);

  t_result = t_left.contract(t_right, dims);

  if (static_cast<DataType>(std::fabs(static_cast<DataType>(
          t_result() - t_result_gpu()))) > error_threshold &&
      !Eigen::internal::isApprox(t_result(), t_result_gpu(), error_threshold)) {
    std::cout << "K: " << k_size << ", N: " << n_size << ", M: " << m_size
              << " : mismatch detected: " << t_result() << " vs "
              << t_result_gpu() << std::endl;
    VERIFY_IS_APPROX(t_result_gpu(), t_result());
  }

  sycl_device.deallocate(d_t_left);
  sycl_device.deallocate(d_t_right);
  sycl_device.deallocate(d_t_result);
}

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void contraction_batch(const Device &sycl_device, IndexType m_size,
                       IndexType k_size, IndexType n_size, IndexType m_batch,
                       IndexType start, IndexType limit) {
  typedef typename Tensor<DataType, 1, DataLayout, IndexType>::DimensionPair
      DimPair;
  static const DataType error_threshold = DataType(1e-4);
  typedef Eigen::array<IndexType, 3> TensorDim;
  typedef Eigen::Tensor<DataType, 3, DataLayout, IndexType> TensorType;
  TensorDim left_dims = {{m_batch, k_size, m_size}};
  TensorDim right_dims = {{m_batch, n_size, k_size}};
  TensorDim res_dims = {{m_batch, m_size, n_size}};
  Eigen::array<DimPair, 1> contract_pairs = {{DimPair(0, 1)}};

  TensorType t_left(left_dims);
  TensorType t_right(right_dims);
  TensorType t_result_gpu(res_dims);
  TensorType t_result(res_dims);

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size() * sizeof(DataType);
  std::size_t t_right_bytes = t_right.size() * sizeof(DataType);
  std::size_t t_result_bytes = t_result.size() * sizeof(DataType);

  DataType *d_t_left =
      static_cast<DataType *>(sycl_device.allocate(t_left_bytes));
  DataType *d_t_right =
      static_cast<DataType *>(sycl_device.allocate(t_right_bytes));
  DataType *d_t_result =
      static_cast<DataType *>(sycl_device.allocate(t_result_bytes));

  Eigen::TensorMap<TensorType> gpu_t_left(d_t_left, left_dims);
  Eigen::TensorMap<TensorType> gpu_t_right(d_t_right, right_dims);
  Eigen::TensorMap<TensorType> gpu_t_result(d_t_result, res_dims);

  sycl_device.memcpyHostToDevice(d_t_left, t_left.data(), t_left_bytes);
  sycl_device.memcpyHostToDevice(d_t_right, t_right.data(), t_right_bytes);
  for (int i = start; i < limit; ++i) {
    auto x = gpu_t_left.template chip<0>(i);
    auto y = gpu_t_right.template chip<0>(i);
    auto z = gpu_t_result.template chip<0>(i);
    z.device(sycl_device) = x.contract(y, contract_pairs);
  }
  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result,
                                 t_result_bytes);

  for (int i = start; i < limit; ++i) {
    auto x = t_left.template chip<0>(i);
    auto y = t_right.template chip<0>(i);
    auto z = t_result.template chip<0>(i);
    z = x.contract(y, contract_pairs);
  }

  for (IndexType i = 0; i < t_result.size(); i++) {
    if (static_cast<DataType>(std::fabs(static_cast<DataType>(
            t_result(i) - t_result_gpu(i)))) < error_threshold) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i),
                                  error_threshold)) {
      continue;
    }
    std::cout << "mismatch detected at IndexType " << i << ": " << t_result(i)
              << " vs " << t_result_gpu(i) << std::endl;
    VERIFY_IS_APPROX(t_result_gpu(i), t_result(i));
  }
  sycl_device.deallocate(d_t_left);
  sycl_device.deallocate(d_t_right);
  sycl_device.deallocate(d_t_result);
}

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void contraction_rhs_transposed(const Device &sycl_device, IndexType m_size,
                                IndexType k_size, IndexType n_size) {
  typedef typename Tensor<DataType, 1, DataLayout, IndexType>::DimensionPair
      DimPair;
  static const DataType error_threshold = DataType(1e-4);
  Eigen::array<IndexType, 2> left_dims = {{m_size, k_size}};
  Eigen::array<IndexType, 2> right_dims = {{n_size, k_size}};
  Eigen::array<IndexType, 2> res_dims = {{m_size, n_size}};
  Eigen::array<DimPair, 1> dims = {{DimPair(1, 1)}};

  Tensor<DataType, 2, DataLayout, IndexType> t_left(left_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_right(right_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_result_gpu(res_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_result(res_dims);

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size() * sizeof(DataType);
  std::size_t t_right_bytes = t_right.size() * sizeof(DataType);
  std::size_t t_result_bytes = t_result.size() * sizeof(DataType);

  DataType *d_t_left =
      static_cast<DataType *>(sycl_device.allocate(t_left_bytes));
  DataType *d_t_right =
      static_cast<DataType *>(sycl_device.allocate(t_right_bytes));
  DataType *d_t_result =
      static_cast<DataType *>(sycl_device.allocate(t_result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_left(d_t_left, left_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_right(d_t_right, right_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_result(d_t_result, res_dims);

  sycl_device.memcpyHostToDevice(d_t_left, t_left.data(), t_left_bytes);
  sycl_device.memcpyHostToDevice(d_t_right, t_right.data(), t_right_bytes);

  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result,
                                 t_result_bytes);

  t_result = t_left.contract(t_right, dims);

  for (IndexType j = 0; j < m_size; j++) {
    for (IndexType i = 0; i < n_size; i++) {
      if (static_cast<DataType>(std::fabs(static_cast<DataType>(
              t_result(j, i) - t_result_gpu(j, i)))) < error_threshold) {
        continue;
      }
      if (Eigen::internal::isApprox(t_result(j, i), t_result_gpu(j, i),
                                    error_threshold)) {
        continue;
      }
      std::cout << "M : " << m_size << ", N : " << n_size << ", K : " << k_size
                << ", mismatch detected at IndexType m: " << j << " n: " << i
                << " CPU : " << t_result(j, i)
                << " vs SYCL:" << t_result_gpu(j, i) << std::endl;
      VERIFY_IS_APPROX(t_result_gpu(j, i), t_result(j, i));
    }
  }
  sycl_device.deallocate(d_t_left);
  sycl_device.deallocate(d_t_right);
  sycl_device.deallocate(d_t_result);
}

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void contraction_lhs_transposed(const Device &sycl_device, IndexType m_size,
                                IndexType k_size, IndexType n_size) {
  typedef typename Tensor<DataType, 1, DataLayout, IndexType>::DimensionPair
      DimPair;
  static const DataType error_threshold = DataType(1e-4);
  Eigen::array<IndexType, 2> left_dims = {{k_size, m_size}};
  Eigen::array<IndexType, 2> right_dims = {{k_size, n_size}};
  Eigen::array<IndexType, 2> res_dims = {{m_size, n_size}};
  Eigen::array<DimPair, 1> dims = {{DimPair(0, 0)}};

  Tensor<DataType, 2, DataLayout, IndexType> t_left(left_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_right(right_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_result_gpu(res_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_result(res_dims);

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size() * sizeof(DataType);
  std::size_t t_right_bytes = t_right.size() * sizeof(DataType);
  std::size_t t_result_bytes = t_result.size() * sizeof(DataType);

  DataType *d_t_left =
      static_cast<DataType *>(sycl_device.allocate(t_left_bytes));
  DataType *d_t_right =
      static_cast<DataType *>(sycl_device.allocate(t_right_bytes));
  DataType *d_t_result =
      static_cast<DataType *>(sycl_device.allocate(t_result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_left(d_t_left, left_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_right(d_t_right, right_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_result(d_t_result, res_dims);

  sycl_device.memcpyHostToDevice(d_t_left, t_left.data(), t_left_bytes);
  sycl_device.memcpyHostToDevice(d_t_right, t_right.data(), t_right_bytes);

  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result,
                                 t_result_bytes);

  t_result = t_left.contract(t_right, dims);

  for (IndexType i = 0; i < t_result.size(); i++) {
    if (static_cast<DataType>(std::fabs(static_cast<DataType>(
            t_result(i) - t_result_gpu(i)))) < error_threshold) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i),
                                  error_threshold)) {
      continue;
    }
    std::cout << "M : " << m_size << ", N : " << n_size << ", K : " << k_size
              << ", mismatch detected at IndexType " << i << ": " << t_result(i)
              << " vs " << t_result_gpu(i) << std::endl;
    VERIFY_IS_APPROX(t_result_gpu(i), t_result(i));
  }
  sycl_device.deallocate(d_t_left);
  sycl_device.deallocate(d_t_right);
  sycl_device.deallocate(d_t_result);
}

template <int DataLayout, typename DataType, typename IndexType,
          typename Device>
void contraction_both_transposed(const Device &sycl_device, IndexType m_size,
                                 IndexType k_size, IndexType n_size) {
  typedef typename Tensor<DataType, 1, DataLayout, IndexType>::DimensionPair
      DimPair;
  static const DataType error_threshold = DataType(1e-4);
  Eigen::array<IndexType, 2> left_dims = {{k_size, m_size}};
  Eigen::array<IndexType, 2> right_dims = {{n_size, k_size}};
  Eigen::array<IndexType, 2> res_dims = {{m_size, n_size}};
  Eigen::array<DimPair, 1> dims = {{DimPair(0, 1)}};

  Tensor<DataType, 2, DataLayout, IndexType> t_left(left_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_right(right_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_result_gpu(res_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_result(res_dims);

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size() * sizeof(DataType);
  std::size_t t_right_bytes = t_right.size() * sizeof(DataType);
  std::size_t t_result_bytes = t_result.size() * sizeof(DataType);

  DataType *d_t_left =
      static_cast<DataType *>(sycl_device.allocate(t_left_bytes));
  DataType *d_t_right =
      static_cast<DataType *>(sycl_device.allocate(t_right_bytes));
  DataType *d_t_result =
      static_cast<DataType *>(sycl_device.allocate(t_result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_left(d_t_left, left_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_right(d_t_right, right_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType>>
      gpu_t_result(d_t_result, res_dims);

  sycl_device.memcpyHostToDevice(d_t_left, t_left.data(), t_left_bytes);
  sycl_device.memcpyHostToDevice(d_t_right, t_right.data(), t_right_bytes);

  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result,
                                 t_result_bytes);

  t_result = t_left.contract(t_right, dims);

  for (IndexType i = 0; i < t_result.size(); i++) {
    if (static_cast<DataType>(std::fabs(static_cast<DataType>(
            t_result(i) - t_result_gpu(i)))) < error_threshold) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i),
                                  error_threshold)) {
      continue;
    }
    std::cout << "M : " << m_size << ", N : " << n_size << ", K : " << k_size
              << ", mismatch detected at IndexType " << i << ": " << t_result(i)
              << " vs " << t_result_gpu(i) << std::endl;

    VERIFY_IS_APPROX(t_result_gpu(i), t_result(i));
  }
  sycl_device.deallocate(d_t_left);
  sycl_device.deallocate(d_t_right);
  sycl_device.deallocate(d_t_result);
}

template <typename Dev>
void inline tensorOutofBound(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // Test out of bound for Tensor-Tensor
  test_no_out_of_bounds<RowMajor, DataType, IndexType>(sycl_device, 10, 1024,
                                                       1024);
  test_no_out_of_bounds<RowMajor, DataType, IndexType>(sycl_device, 1024, 1024,
                                                       4096);
  test_no_out_of_bounds<RowMajor, DataType, IndexType>(sycl_device, 4096, 1024,
                                                       2048);
  test_no_out_of_bounds<ColMajor, DataType, IndexType>(sycl_device, 784, 2048,
                                                       1024);
  test_no_out_of_bounds<ColMajor, DataType, IndexType>(sycl_device, 2048, 1024,
                                                       784);
  test_no_out_of_bounds<RowMajor, DataType, IndexType>(sycl_device, 10, 1024,
                                                       10);
  test_no_out_of_bounds<RowMajor, DataType, IndexType>(sycl_device, 513, 4096,
                                                       513);
  test_no_out_of_bounds<RowMajor, DataType, IndexType>(sycl_device, 783, 1024,
                                                       783);
  test_no_out_of_bounds<ColMajor, DataType, IndexType>(sycl_device, 784, 2048,
                                                       784);
  test_no_out_of_bounds<ColMajor, DataType, IndexType>(sycl_device, 11, 1024,
                                                       11);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "tensor out of bound tests finished computation at "
            << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensorTensor(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // Tensor Tensor Contraction
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 128, 128,
                                                       128);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 128, 128,
                                                       128);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "tensor tensor tests finished computation at "
            << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensorTensor_m(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // Tensor Tensor Contraction
  test_sycl_contraction_m<ColMajor, DataType, IndexType>(sycl_device);
  test_sycl_contraction_m<RowMajor, DataType, IndexType>(sycl_device);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "tensor tensor tests finished computation at "
            << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensorTensor_n(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // Tensor Tensor Contraction
  test_sycl_contraction_n<ColMajor, DataType, IndexType>(sycl_device);
  test_sycl_contraction_n<RowMajor, DataType, IndexType>(sycl_device);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "tensor tensor tests finished computation at "
            << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensorTensor_k(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  test_sycl_contraction_k<ColMajor, DataType, IndexType>(sycl_device);
  test_sycl_contraction_k<RowMajor, DataType, IndexType>(sycl_device);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "tensor tensor tests finished computation at "
            << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensorTensor_sizes(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // Tensor Tensor Contraction
  test_sycl_contraction_sizes<ColMajor, DataType, IndexType>(sycl_device);
  test_sycl_contraction_sizes<RowMajor, DataType, IndexType>(sycl_device);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "tensor tensor tests finished computation at "
            << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}
template <typename Dev>
void inline vectorVector(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // VECTOR-VECTOR
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1025, 1,
                                                       1025);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1025, 1,
                                                       1025);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1024, 1,
                                                       1024);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1024, 1,
                                                       1024);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1023, 1,
                                                       1023);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1023, 1,
                                                       1023);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "contracted tensor tests finished computation at "
            << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline vectorTensor(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // Vector-Tensor
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1, 1025,
                                                       1025);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1, 1025,
                                                       1025);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1, 1024,
                                                       1024);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1, 1024,
                                                       1024);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1, 1023,
                                                       1023);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1, 1023,
                                                       1023);

  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1, 4097,
                                                       4097);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1, 4097,
                                                       4097);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1, 4096,
                                                       4096);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1, 4096,
                                                       4096);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1, 4095,
                                                       4095);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1, 4095,
                                                       4095);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1, 802816,
                                                       32);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensorVector(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // Matrix-Vector
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1025, 1025,
                                                       1);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1125, 1025,
                                                       1);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1224, 1024,
                                                       1);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1024, 1024,
                                                       1);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 1023, 1023,
                                                       1);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 1023, 1023,
                                                       1);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 4097, 4197,
                                                       1);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 4097, 4097,
                                                       1);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 4096, 4096,
                                                       1);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 4096, 8196,
                                                       1);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 4095, 4095,
                                                       1);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 4095, 4095,
                                                       1);
// If the GEMV disabled it will creates one kernel to calculate the contraction.
// Therefore the acumuation of float number will overflow the precision
// threshold for float and cause the test to fail. While it the GMV multiple
// kernel will be created and each one run the overflow of accumutation breaks
// among the kernels.
#ifndef EIGEN_SYCL_DISABLE_GEMV
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 32, 802032,
                                                       1);
#endif

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensorScalar(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // SCALAR Contraction
  test_scalar<ColMajor, DataType, IndexType>(sycl_device, 127, 127, 127);
  test_scalar<RowMajor, DataType, IndexType>(sycl_device, 127, 127, 127);
  test_scalar<ColMajor, DataType, IndexType>(sycl_device, 128, 128, 128);
  test_scalar<RowMajor, DataType, IndexType>(sycl_device, 128, 128, 128);
  test_scalar<ColMajor, DataType, IndexType>(sycl_device, 129, 129, 129);
  test_scalar<RowMajor, DataType, IndexType>(sycl_device, 129, 129, 129);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline skinnyTensor_row(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // Tensor Tensor Contraction
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 16, 4, 16);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 257, 131073,
                                                       257);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 256, 131072,
                                                       256);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 16, 131073,
                                                       16);
  test_sycl_contraction<RowMajor, DataType, IndexType>(sycl_device, 17, 131072,
                                                       17);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline skinnyTensor_col(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  // Tensor Tensor Contraction
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 16, 4, 16);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 257, 131073,
                                                       257);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 256, 131072,
                                                       256);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 16, 131073,
                                                       16);
  test_sycl_contraction<ColMajor, DataType, IndexType>(sycl_device, 17, 131072,
                                                       17);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensor_contraction_batch_per_device(const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  contraction_batch<RowMajor, DataType, IndexType>(sycl_device, 64, 75, 30, 4,
                                                   0, 4);
  contraction_batch<ColMajor, DataType, IndexType>(sycl_device, 64, 75, 30, 4,
                                                   0, 4);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensor_contraction_lhs_transposed_per_device(
    const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  contraction_lhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 8, 4,
                                                            8);
  contraction_lhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 32, 8,
                                                            32);
  contraction_lhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 64, 16,
                                                            64);
  contraction_lhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 784,
                                                            2048, 1024);
  contraction_lhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 1024,
                                                            10, 1024);
  contraction_lhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 4096,
                                                            1024, 1024);
  contraction_lhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 2048,
                                                            4096, 1024);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensor_contraction_rhs_transposed_per_device(
    const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  contraction_rhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 16, 4,
                                                            16);
  contraction_rhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 17, 5,
                                                            17);
  contraction_rhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 32, 8,
                                                            32);
  contraction_rhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 64, 16,
                                                            64);
  contraction_rhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 10,
                                                            1024, 1024);
  contraction_rhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 1024,
                                                            1024, 4096);
  contraction_rhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 4096,
                                                            1024, 2048);
  contraction_rhs_transposed<RowMajor, DataType, IndexType>(sycl_device, 2048,
                                                            1024, 784);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

template <typename Dev>
void inline tensor_contraction_both_transposed_per_device(
    const Dev &sycl_device) {
  typedef float DataType;
  typedef int64_t IndexType;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  contraction_both_transposed<RowMajor, DataType, IndexType>(sycl_device, 17, 5,
                                                             17);
  contraction_both_transposed<RowMajor, DataType, IndexType>(sycl_device, 32, 8,
                                                             32);
  contraction_both_transposed<RowMajor, DataType, IndexType>(sycl_device, 64,
                                                             16, 64);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

EIGEN_DECLARE_TEST(cxx11_tensor_contract_sycl) {
  for (const auto &device : Eigen::get_sycl_supported_devices()) {
    std::cout << "Running on "
              << device.template get_info<cl::sycl::info::device::name>()
              << std::endl;
    QueueInterface queueInterface(device);
    auto sycl_device = Eigen::SyclDevice(&queueInterface);
    CALL_SUBTEST_1(tensorOutofBound(sycl_device));
    CALL_SUBTEST_2(tensorTensor(sycl_device));
    CALL_SUBTEST_2(tensorTensor_m(sycl_device));
    CALL_SUBTEST_2(tensorTensor_n(sycl_device));
    CALL_SUBTEST_2(tensorTensor_k(sycl_device));
    CALL_SUBTEST_2(tensorTensor_sizes(sycl_device));
    CALL_SUBTEST_3(vectorVector(sycl_device));
    CALL_SUBTEST_4(vectorTensor(sycl_device));
    CALL_SUBTEST_5(tensorVector(sycl_device));
    CALL_SUBTEST_6(tensorScalar(sycl_device));
    CALL_SUBTEST_7(skinnyTensor_row(sycl_device));
    CALL_SUBTEST_7(skinnyTensor_col(sycl_device));
    CALL_SUBTEST_8(tensor_contraction_batch_per_device(sycl_device));
    CALL_SUBTEST_9(tensor_contraction_lhs_transposed_per_device(sycl_device));
    CALL_SUBTEST_10(tensor_contraction_rhs_transposed_per_device(sycl_device));
    CALL_SUBTEST_11(tensor_contraction_both_transposed_per_device(sycl_device));
  }
}
