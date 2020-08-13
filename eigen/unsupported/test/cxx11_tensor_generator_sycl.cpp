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
static const float error_threshold =1e-8f;

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;
struct Generator1D {
  Generator1D() { }

  float operator()(const array<Eigen::DenseIndex, 1>& coordinates) const {
    return coordinates[0];
  }
};

template <typename DataType, int DataLayout, typename IndexType>
static void test_1D_sycl(const Eigen::SyclDevice& sycl_device)
{

  IndexType sizeDim1 = 6;
  array<IndexType, 1> tensorRange = {{sizeDim1}};
  Tensor<DataType, 1, DataLayout,IndexType> vec(tensorRange);
  Tensor<DataType, 1, DataLayout,IndexType> result(tensorRange);

  const size_t tensorBuffSize =vec.size()*sizeof(DataType);
  DataType* gpu_data_vec  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  DataType* gpu_data_result  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));

  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> gpu_vec(gpu_data_vec, tensorRange);
  TensorMap<Tensor<DataType, 1, DataLayout,IndexType>> gpu_result(gpu_data_result, tensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_vec, vec.data(), tensorBuffSize);
  gpu_result.device(sycl_device)=gpu_vec.generate(Generator1D());
  sycl_device.memcpyDeviceToHost(result.data(), gpu_data_result, tensorBuffSize);

  for (IndexType i = 0; i < 6; ++i) {
    VERIFY_IS_EQUAL(result(i), i);
  }
}


struct Generator2D {
  Generator2D() { }

  float operator()(const array<Eigen::DenseIndex, 2>& coordinates) const {
    return 3 * coordinates[0] + 11 * coordinates[1];
  }
};

template <typename DataType, int DataLayout, typename IndexType>
static void test_2D_sycl(const Eigen::SyclDevice& sycl_device)
{
  IndexType sizeDim1 = 5;
  IndexType sizeDim2 = 7;
  array<IndexType, 2> tensorRange = {{sizeDim1, sizeDim2}};
  Tensor<DataType, 2, DataLayout,IndexType> matrix(tensorRange);
  Tensor<DataType, 2, DataLayout,IndexType> result(tensorRange);

  const size_t tensorBuffSize =matrix.size()*sizeof(DataType);
  DataType* gpu_data_matrix  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  DataType* gpu_data_result  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));

  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_matrix(gpu_data_matrix, tensorRange);
  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_result(gpu_data_result, tensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_matrix, matrix.data(), tensorBuffSize);
  gpu_result.device(sycl_device)=gpu_matrix.generate(Generator2D());
  sycl_device.memcpyDeviceToHost(result.data(), gpu_data_result, tensorBuffSize);

  for (IndexType i = 0; i < 5; ++i) {
    for (IndexType j = 0; j < 5; ++j) {
      VERIFY_IS_EQUAL(result(i, j), 3*i + 11*j);
    }
  }
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_gaussian_sycl(const Eigen::SyclDevice& sycl_device)
{
  IndexType rows = 32;
  IndexType cols = 48;
  array<DataType, 2> means;
  means[0] = rows / 2.0f;
  means[1] = cols / 2.0f;
  array<DataType, 2> std_devs;
  std_devs[0] = 3.14f;
  std_devs[1] = 2.7f;
  internal::GaussianGenerator<DataType, Eigen::DenseIndex, 2> gaussian_gen(means, std_devs);

  array<IndexType, 2> tensorRange = {{rows, cols}};
  Tensor<DataType, 2, DataLayout,IndexType> matrix(tensorRange);
  Tensor<DataType, 2, DataLayout,IndexType> result(tensorRange);

  const size_t tensorBuffSize =matrix.size()*sizeof(DataType);
  DataType* gpu_data_matrix  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  DataType* gpu_data_result  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));

  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_matrix(gpu_data_matrix, tensorRange);
  TensorMap<Tensor<DataType, 2, DataLayout,IndexType>> gpu_result(gpu_data_result, tensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_matrix, matrix.data(), tensorBuffSize);
  gpu_result.device(sycl_device)=gpu_matrix.generate(gaussian_gen);
  sycl_device.memcpyDeviceToHost(result.data(), gpu_data_result, tensorBuffSize);

  for (IndexType i = 0; i < rows; ++i) {
    for (IndexType j = 0; j < cols; ++j) {
      DataType g_rows = powf(rows/2.0f - i, 2) / (3.14f * 3.14f) * 0.5f;
      DataType g_cols = powf(cols/2.0f - j, 2) / (2.7f * 2.7f) * 0.5f;
      DataType gaussian = expf(-g_rows - g_cols);
      Eigen::internal::isApprox(result(i, j), gaussian, error_threshold);
    }
  }
}

template<typename DataType, typename dev_Selector> void sycl_generator_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_1D_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_1D_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_2D_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_2D_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_gaussian_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_gaussian_sycl<DataType, ColMajor, int64_t>(sycl_device);
}
EIGEN_DECLARE_TEST(cxx11_tensor_generator_sycl)
{
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_generator_test_per_device<float>(device));
  }
}
