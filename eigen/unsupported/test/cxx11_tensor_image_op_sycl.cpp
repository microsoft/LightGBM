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
static void test_image_op_sycl(const Eigen::SyclDevice &sycl_device)
{
  IndexType sizeDim1 = 245;
  IndexType sizeDim2 = 343;
  IndexType sizeDim3 = 577;

  array<IndexType, 3> input_range ={{sizeDim1, sizeDim2, sizeDim3}};
  array<IndexType, 3> slice_range ={{sizeDim1-1, sizeDim2, sizeDim3}};

  Tensor<DataType, 3,DataLayout, IndexType> tensor1(input_range);
  Tensor<DataType, 3,DataLayout, IndexType> tensor2(input_range);
  Tensor<DataType, 3, DataLayout, IndexType> tensor3(slice_range);
  Tensor<DataType, 3, DataLayout, IndexType> tensor3_cpu(slice_range);



  typedef Eigen::DSizes<IndexType, 3> Index3;
  Index3 strides1(1L,1L, 1L);
  Index3 indicesStart1(1L, 0L, 0L);
  Index3 indicesStop1(sizeDim1, sizeDim2, sizeDim3);

  Index3 strides2(1L,1L, 1L);
  Index3 indicesStart2(0L, 0L, 0L);
  Index3 indicesStop2(sizeDim1-1, sizeDim2, sizeDim3);
  Eigen::DSizes<IndexType, 3> sizes(sizeDim1-1,sizeDim2,sizeDim3);

  tensor1.setRandom();
  tensor2.setRandom();


  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor1.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(tensor3.size()*sizeof(DataType)));

  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu1(gpu_data1, input_range);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu2(gpu_data2, input_range);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu3(gpu_data3, slice_range);

  sycl_device.memcpyHostToDevice(gpu_data1, tensor1.data(),(tensor1.size())*sizeof(DataType));
  sycl_device.memcpyHostToDevice(gpu_data2, tensor2.data(),(tensor2.size())*sizeof(DataType));
  gpu3.device(sycl_device)= gpu1.slice(indicesStart1, sizes) - gpu2.slice(indicesStart2, sizes);
  sycl_device.memcpyDeviceToHost(tensor3.data(), gpu_data3,(tensor3.size())*sizeof(DataType));

  tensor3_cpu = tensor1.stridedSlice(indicesStart1,indicesStop1,strides1) - tensor2.stridedSlice(indicesStart2,indicesStop2,strides2);


  for (IndexType i = 0; i <slice_range[0] ; ++i) {
    for (IndexType j = 0; j < slice_range[1]; ++j) {
      for (IndexType k = 0; k < slice_range[2]; ++k) {
        VERIFY_IS_EQUAL(tensor3_cpu(i,j,k), tensor3(i,j,k));
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
  sycl_device.deallocate(gpu_data3);
}


template<typename DataType, typename dev_Selector> void sycl_computing_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_image_op_sycl<DataType, RowMajor, int64_t>(sycl_device);
}

EIGEN_DECLARE_TEST(cxx11_tensor_image_op_sycl) {
  for (const auto& device :Eigen::get_sycl_supported_devices()) { 
   CALL_SUBTEST(sycl_computing_test_per_device<float>(device));
#ifdef EIGEN_SYCL_DOUBLE_SUPPORT
   CALL_SUBTEST(sycl_computing_test_per_device<double>(device));
#endif
  }
}
