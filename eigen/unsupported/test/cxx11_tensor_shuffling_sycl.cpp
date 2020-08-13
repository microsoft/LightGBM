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

template <typename DataType, int DataLayout, typename IndexType>
static void test_simple_shuffling_sycl(const Eigen::SyclDevice& sycl_device) {
  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  array<IndexType, 4> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  Tensor<DataType, 4, DataLayout, IndexType> tensor(tensorRange);
  Tensor<DataType, 4, DataLayout, IndexType> no_shuffle(tensorRange);
  tensor.setRandom();

  const size_t buffSize = tensor.size() * sizeof(DataType);
  array<IndexType, 4> shuffles;
  shuffles[0] = 0;
  shuffles[1] = 1;
  shuffles[2] = 2;
  shuffles[3] = 3;
  DataType* gpu_data1 = static_cast<DataType*>(sycl_device.allocate(buffSize));
  DataType* gpu_data2 = static_cast<DataType*>(sycl_device.allocate(buffSize));

  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu1(gpu_data1,
                                                             tensorRange);
  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu2(gpu_data2,
                                                             tensorRange);

  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(), buffSize);

  gpu2.device(sycl_device) = gpu1.shuffle(shuffles);
  sycl_device.memcpyDeviceToHost(no_shuffle.data(), gpu_data2, buffSize);
  sycl_device.synchronize();

  VERIFY_IS_EQUAL(no_shuffle.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(no_shuffle.dimension(1), sizeDim2);
  VERIFY_IS_EQUAL(no_shuffle.dimension(2), sizeDim3);
  VERIFY_IS_EQUAL(no_shuffle.dimension(3), sizeDim4);

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        for (IndexType l = 0; l < sizeDim4; ++l) {
          VERIFY_IS_EQUAL(tensor(i, j, k, l), no_shuffle(i, j, k, l));
        }
      }
    }
  }

  shuffles[0] = 2;
  shuffles[1] = 3;
  shuffles[2] = 1;
  shuffles[3] = 0;
  array<IndexType, 4> tensorrangeShuffle = {
      {sizeDim3, sizeDim4, sizeDim2, sizeDim1}};
  Tensor<DataType, 4, DataLayout, IndexType> shuffle(tensorrangeShuffle);
  DataType* gpu_data3 = static_cast<DataType*>(sycl_device.allocate(buffSize));
  TensorMap<Tensor<DataType, 4, DataLayout, IndexType>> gpu3(
      gpu_data3, tensorrangeShuffle);

  gpu3.device(sycl_device) = gpu1.shuffle(shuffles);
  sycl_device.memcpyDeviceToHost(shuffle.data(), gpu_data3, buffSize);
  sycl_device.synchronize();

  VERIFY_IS_EQUAL(shuffle.dimension(0), sizeDim3);
  VERIFY_IS_EQUAL(shuffle.dimension(1), sizeDim4);
  VERIFY_IS_EQUAL(shuffle.dimension(2), sizeDim2);
  VERIFY_IS_EQUAL(shuffle.dimension(3), sizeDim1);

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        for (IndexType l = 0; l < sizeDim4; ++l) {
          VERIFY_IS_EQUAL(tensor(i, j, k, l), shuffle(k, l, j, i));
        }
      }
    }
  }
}

template <typename DataType, typename dev_Selector>
void sycl_shuffling_test_per_device(dev_Selector s) {
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_simple_shuffling_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_simple_shuffling_sycl<DataType, ColMajor, int64_t>(sycl_device);
}
EIGEN_DECLARE_TEST(cxx11_tensor_shuffling_sycl) {
  for (const auto& device : Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_shuffling_test_per_device<float>(device));
  }
}
