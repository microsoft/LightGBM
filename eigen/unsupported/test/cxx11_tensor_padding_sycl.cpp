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


template<typename DataType, int DataLayout, typename IndexType>
static void test_simple_padding(const Eigen::SyclDevice& sycl_device)
{

  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  array<IndexType, 4> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};

  Tensor<DataType, 4, DataLayout, IndexType> tensor(tensorRange);
  tensor.setRandom();

  array<std::pair<IndexType, IndexType>, 4> paddings;
  paddings[0] = std::make_pair(0, 0);
  paddings[1] = std::make_pair(2, 1);
  paddings[2] = std::make_pair(3, 4);
  paddings[3] = std::make_pair(0, 0);

  IndexType padedSizeDim1 = 2;
  IndexType padedSizeDim2 = 6;
  IndexType padedSizeDim3 = 12;
  IndexType padedSizeDim4 = 7;
  array<IndexType, 4> padedtensorRange = {{padedSizeDim1, padedSizeDim2, padedSizeDim3, padedSizeDim4}};

  Tensor<DataType, 4, DataLayout, IndexType> padded(padedtensorRange);


  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(padded.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu2(gpu_data2, padedtensorRange);

  VERIFY_IS_EQUAL(padded.dimension(0), 2+0);
  VERIFY_IS_EQUAL(padded.dimension(1), 3+3);
  VERIFY_IS_EQUAL(padded.dimension(2), 5+7);
  VERIFY_IS_EQUAL(padded.dimension(3), 7+0);
  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
  gpu2.device(sycl_device)=gpu1.pad(paddings);
  sycl_device.memcpyDeviceToHost(padded.data(), gpu_data2,(padded.size())*sizeof(DataType));
  for (IndexType i = 0; i < padedSizeDim1; ++i) {
    for (IndexType j = 0; j < padedSizeDim2; ++j) {
      for (IndexType k = 0; k < padedSizeDim3; ++k) {
        for (IndexType l = 0; l < padedSizeDim4; ++l) {
          if (j >= 2 && j < 5 && k >= 3 && k < 8) {
            VERIFY_IS_EQUAL(padded(i,j,k,l), tensor(i,j-2,k-3,l));
          } else {
            VERIFY_IS_EQUAL(padded(i,j,k,l), 0.0f);
          }
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
}

template<typename DataType, int DataLayout, typename IndexType>
static void test_padded_expr(const Eigen::SyclDevice& sycl_device)
{
  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  array<IndexType, 4> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};

  Tensor<DataType, 4, DataLayout, IndexType> tensor(tensorRange);
  tensor.setRandom();

  array<std::pair<IndexType, IndexType>, 4> paddings;
  paddings[0] = std::make_pair(0, 0);
  paddings[1] = std::make_pair(2, 1);
  paddings[2] = std::make_pair(3, 4);
  paddings[3] = std::make_pair(0, 0);

  Eigen::DSizes<IndexType, 2> reshape_dims;
  reshape_dims[0] = 12;
  reshape_dims[1] = 84;


  Tensor<DataType, 2, DataLayout, IndexType>  result(reshape_dims);

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(result.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 4,DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu2(gpu_data2, reshape_dims);


  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
  gpu2.device(sycl_device)=gpu1.pad(paddings).reshape(reshape_dims);
  sycl_device.memcpyDeviceToHost(result.data(), gpu_data2,(result.size())*sizeof(DataType));

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 6; ++j) {
      for (IndexType k = 0; k < 12; ++k) {
        for (IndexType l = 0; l < 7; ++l) {
          const float result_value = DataLayout == ColMajor ?
              result(i+2*j,k+12*l) : result(j+6*i,l+7*k);
          if (j >= 2 && j < 5 && k >= 3 && k < 8) {
            VERIFY_IS_EQUAL(result_value, tensor(i,j-2,k-3,l));
          } else {
            VERIFY_IS_EQUAL(result_value, 0.0f);
          }
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
}

template<typename DataType, typename dev_Selector> void sycl_padding_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_simple_padding<DataType, RowMajor, int64_t>(sycl_device);
  test_simple_padding<DataType, ColMajor, int64_t>(sycl_device);
  test_padded_expr<DataType, RowMajor, int64_t>(sycl_device);
  test_padded_expr<DataType, ColMajor, int64_t>(sycl_device);

}
EIGEN_DECLARE_TEST(cxx11_tensor_padding_sycl)
{
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_padding_test_per_device<float>(device));
  }
}
