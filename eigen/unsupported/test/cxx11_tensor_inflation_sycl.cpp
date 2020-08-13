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

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

// Inflation Definition for each dimension the inflated val would be
//((dim-1)*strid[dim] +1)

// for 1 dimension vector of size 3 with value (4,4,4) with the inflated stride value of 3 would be changed to
// tensor of size (2*3) +1 = 7 with the value of
// (4, 0, 0, 4, 0, 0, 4).

template <typename DataType, int DataLayout, typename IndexType>
void test_simple_inflation_sycl(const Eigen::SyclDevice &sycl_device) {


  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  array<IndexType, 4> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  Tensor<DataType, 4, DataLayout,IndexType> tensor(tensorRange);
  Tensor<DataType, 4, DataLayout,IndexType> no_stride(tensorRange);
  tensor.setRandom();

  array<IndexType, 4> strides;
  strides[0] = 1;
  strides[1] = 1;
  strides[2] = 1;
  strides[3] = 1;


  const size_t tensorBuffSize =tensor.size()*sizeof(DataType);
  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  DataType* gpu_data_no_stride  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));

  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_no_stride(gpu_data_no_stride, tensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
  gpu_no_stride.device(sycl_device)=gpu_tensor.inflate(strides);
  sycl_device.memcpyDeviceToHost(no_stride.data(), gpu_data_no_stride, tensorBuffSize);

  VERIFY_IS_EQUAL(no_stride.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(no_stride.dimension(1), sizeDim2);
  VERIFY_IS_EQUAL(no_stride.dimension(2), sizeDim3);
  VERIFY_IS_EQUAL(no_stride.dimension(3), sizeDim4);

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 3; ++j) {
      for (IndexType k = 0; k < 5; ++k) {
        for (IndexType l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), no_stride(i,j,k,l));
        }
      }
    }
  }


  strides[0] = 2;
  strides[1] = 4;
  strides[2] = 2;
  strides[3] = 3;

  IndexType inflatedSizeDim1 = 3;
  IndexType inflatedSizeDim2 = 9;
  IndexType inflatedSizeDim3 = 9;
  IndexType inflatedSizeDim4 = 19;
  array<IndexType, 4> inflatedTensorRange = {{inflatedSizeDim1, inflatedSizeDim2, inflatedSizeDim3, inflatedSizeDim4}};

  Tensor<DataType, 4, DataLayout, IndexType> inflated(inflatedTensorRange);

  const size_t inflatedTensorBuffSize =inflated.size()*sizeof(DataType);
  DataType* gpu_data_inflated  = static_cast<DataType*>(sycl_device.allocate(inflatedTensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu_inflated(gpu_data_inflated, inflatedTensorRange);
  gpu_inflated.device(sycl_device)=gpu_tensor.inflate(strides);
  sycl_device.memcpyDeviceToHost(inflated.data(), gpu_data_inflated, inflatedTensorBuffSize);

  VERIFY_IS_EQUAL(inflated.dimension(0), inflatedSizeDim1);
  VERIFY_IS_EQUAL(inflated.dimension(1), inflatedSizeDim2);
  VERIFY_IS_EQUAL(inflated.dimension(2), inflatedSizeDim3);
  VERIFY_IS_EQUAL(inflated.dimension(3), inflatedSizeDim4);

  for (IndexType i = 0; i < inflatedSizeDim1; ++i) {
    for (IndexType j = 0; j < inflatedSizeDim2; ++j) {
      for (IndexType k = 0; k < inflatedSizeDim3; ++k) {
        for (IndexType l = 0; l < inflatedSizeDim4; ++l) {
          if (i % strides[0] == 0 &&
              j % strides[1] == 0 &&
              k % strides[2] == 0 &&
              l % strides[3] == 0) {
            VERIFY_IS_EQUAL(inflated(i,j,k,l),
                            tensor(i/strides[0], j/strides[1], k/strides[2], l/strides[3]));
          } else {
            VERIFY_IS_EQUAL(0, inflated(i,j,k,l));
          }
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data_tensor);
  sycl_device.deallocate(gpu_data_no_stride);
  sycl_device.deallocate(gpu_data_inflated);
}

template<typename DataType, typename dev_Selector> void sycl_inflation_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_simple_inflation_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_simple_inflation_sycl<DataType, ColMajor, int64_t>(sycl_device);
}
EIGEN_DECLARE_TEST(cxx11_tensor_inflation_sycl)
{
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_inflation_test_per_device<float>(device));
  }
}
