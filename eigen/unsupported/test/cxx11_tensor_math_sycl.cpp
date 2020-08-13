// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
// Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

using Eigen::Tensor;
using Eigen::RowMajor;
template <typename DataType, int DataLayout, typename IndexType>
static void test_tanh_sycl(const Eigen::SyclDevice &sycl_device)
{

  IndexType sizeDim1 = 4;
  IndexType sizeDim2 = 4;
  IndexType sizeDim3 = 1;
  array<IndexType, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  Tensor<DataType, 3, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 3, DataLayout, IndexType> out(tensorRange);
  Tensor<DataType, 3, DataLayout, IndexType> out_cpu(tensorRange);

  in = in.random();

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(in.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(out.size()*sizeof(DataType)));

  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu1(gpu_data1, tensorRange);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu2(gpu_data2, tensorRange);

  sycl_device.memcpyHostToDevice(gpu_data1, in.data(),(in.size())*sizeof(DataType));
  gpu2.device(sycl_device) = gpu1.tanh();
  sycl_device.memcpyDeviceToHost(out.data(), gpu_data2,(out.size())*sizeof(DataType));

  out_cpu=in.tanh();

  for (int i = 0; i < in.size(); ++i) {
    VERIFY_IS_APPROX(out(i), out_cpu(i));
  }
}
template <typename DataType, int DataLayout, typename IndexType>
static void test_sigmoid_sycl(const Eigen::SyclDevice &sycl_device)
{

  IndexType sizeDim1 = 4;
  IndexType sizeDim2 = 4;
  IndexType sizeDim3 = 1;
  array<IndexType, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  Tensor<DataType, 3, DataLayout, IndexType> in(tensorRange);
  Tensor<DataType, 3, DataLayout, IndexType> out(tensorRange);
  Tensor<DataType, 3, DataLayout, IndexType> out_cpu(tensorRange);

  in = in.random();

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(in.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(out.size()*sizeof(DataType)));

  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu1(gpu_data1, tensorRange);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu2(gpu_data2, tensorRange);

  sycl_device.memcpyHostToDevice(gpu_data1, in.data(),(in.size())*sizeof(DataType));
  gpu2.device(sycl_device) = gpu1.sigmoid();
  sycl_device.memcpyDeviceToHost(out.data(), gpu_data2,(out.size())*sizeof(DataType));

  out_cpu=in.sigmoid();

  for (int i = 0; i < in.size(); ++i) {
    VERIFY_IS_APPROX(out(i), out_cpu(i));
  }
}


template<typename DataType, typename dev_Selector> void sycl_computing_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_tanh_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_tanh_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_sigmoid_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_sigmoid_sycl<DataType, ColMajor, int64_t>(sycl_device);
}

EIGEN_DECLARE_TEST(cxx11_tensor_math_sycl) {
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_computing_test_per_device<float>(device));
  }
}
