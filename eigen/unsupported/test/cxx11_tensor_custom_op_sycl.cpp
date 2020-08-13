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
template<typename TensorType>
struct InsertZeros {
  DSizes<DenseIndex, 2> dimensions(const TensorType& input) const {
    DSizes<DenseIndex, 2> result;
    result[0] = input.dimension(0) * 2;
    result[1] = input.dimension(1) * 2;
    return result;
  }

  template <typename Output, typename Device>
  void eval(const TensorType& input, Output& output, const Device& device) const
  {
    array<DenseIndex, 2> strides;
    strides[0] = 2;
    strides[1] = 2;
    output.stride(strides).device(device) = input;

    Eigen::DSizes<DenseIndex, 2> offsets(1,1);
    Eigen::DSizes<DenseIndex, 2> extents(output.dimension(0)-1, output.dimension(1)-1);
    output.slice(offsets, extents).stride(strides).device(device) = input.constant(0.0f);
  }
};

template<typename DataType, int DataLayout, typename IndexType>
static void test_custom_unary_op_sycl(const Eigen::SyclDevice &sycl_device)
{
  IndexType sizeDim1 = 3;
  IndexType sizeDim2 = 5;
  Eigen::array<IndexType, 2> tensorRange = {{sizeDim1, sizeDim2}};
  Eigen::array<IndexType, 2> tensorResultRange = {{6, 10}};

  Eigen::Tensor<DataType, 2, DataLayout, IndexType> in1(tensorRange);
  Eigen::Tensor<DataType, 2, DataLayout, IndexType> out(tensorResultRange);

  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(DataType)));
  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));

  typedef Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > TensorType;
  TensorType gpu_in1(gpu_in1_data, tensorRange);
  TensorType gpu_out(gpu_out_data, tensorResultRange);

  in1.setRandom();
  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(DataType));
  gpu_out.device(sycl_device) = gpu_in1.customOp(InsertZeros<TensorType>());
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));

  VERIFY_IS_EQUAL(out.dimension(0), 6);
  VERIFY_IS_EQUAL(out.dimension(1), 10);

  for (int i = 0; i < 6; i+=2) {
    for (int j = 0; j < 10; j+=2) {
      VERIFY_IS_EQUAL(out(i, j), in1(i/2, j/2));
    }
  }
  for (int i = 1; i < 6; i+=2) {
    for (int j = 1; j < 10; j+=2) {
      VERIFY_IS_EQUAL(out(i, j), 0);
    }
  }
  sycl_device.deallocate(gpu_in1_data);
sycl_device.deallocate(gpu_out_data);
}

template<typename TensorType>
struct BatchMatMul {
  DSizes<DenseIndex, 3> dimensions(const TensorType& input1, const TensorType& input2) const {
    DSizes<DenseIndex, 3> result;
    result[0] = input1.dimension(0);
    result[1] = input2.dimension(1);
    result[2] = input2.dimension(2);
    return result;
  }

  template <typename Output, typename Device>
  void eval(const TensorType& input1, const TensorType& input2,
            Output& output, const Device& device) const
  {
    typedef typename TensorType::DimensionPair DimPair;
    array<DimPair, 1> dims;
    dims[0] = DimPair(1, 0);
    for (int64_t i = 0; i < output.dimension(2); ++i) {
      output.template chip<2>(i).device(device) = input1.template chip<2>(i).contract(input2.template chip<2>(i), dims);
    }
  }
};

template<typename DataType, int DataLayout, typename IndexType>
static void test_custom_binary_op_sycl(const Eigen::SyclDevice &sycl_device)
{

  Eigen::array<IndexType, 3> tensorRange1 = {{2, 3, 5}};
  Eigen::array<IndexType, 3> tensorRange2 = {{3,7,5}};
  Eigen::array<IndexType, 3> tensorResultRange  = {{2, 7, 5}};

  Eigen::Tensor<DataType, 3, DataLayout, IndexType> in1(tensorRange1);
  Eigen::Tensor<DataType, 3, DataLayout, IndexType> in2(tensorRange2);
  Eigen::Tensor<DataType, 3, DataLayout, IndexType> out(tensorResultRange);

  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.dimensions().TotalSize()*sizeof(DataType)));
  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(in2.dimensions().TotalSize()*sizeof(DataType)));
  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.dimensions().TotalSize()*sizeof(DataType)));

  typedef Eigen::TensorMap<Eigen::Tensor<DataType, 3, DataLayout, IndexType> > TensorType;
  TensorType gpu_in1(gpu_in1_data, tensorRange1);
  TensorType gpu_in2(gpu_in2_data, tensorRange2);
  TensorType gpu_out(gpu_out_data, tensorResultRange);

  in1.setRandom();
  in2.setRandom();

  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.dimensions().TotalSize())*sizeof(DataType));
  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.dimensions().TotalSize())*sizeof(DataType));

  gpu_out.device(sycl_device) = gpu_in1.customOp(gpu_in2, BatchMatMul<TensorType>());
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.dimensions().TotalSize())*sizeof(DataType));

  for (IndexType i = 0; i < 5; ++i) {
    typedef typename Eigen::Tensor<DataType, 3, DataLayout, IndexType>::DimensionPair DimPair;
    array<DimPair, 1> dims;
    dims[0] = DimPair(1, 0);
    Eigen::Tensor<DataType, 2, DataLayout, IndexType> reference = in1.template chip<2>(i).contract(in2.template chip<2>(i), dims);
    TensorRef<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > val = out.template chip<2>(i);
    for (IndexType j = 0; j < 2; ++j) {
      for (IndexType k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(val(j, k), reference(j, k));
      }
    }
  }
  sycl_device.deallocate(gpu_in1_data);
  sycl_device.deallocate(gpu_in2_data);
  sycl_device.deallocate(gpu_out_data);
}

template <typename DataType, typename Dev_selector> void custom_op_perDevice(Dev_selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_custom_unary_op_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_custom_unary_op_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_custom_binary_op_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_custom_binary_op_sycl<DataType, RowMajor, int64_t>(sycl_device);

}
EIGEN_DECLARE_TEST(cxx11_tensor_custom_op_sycl) {
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(custom_op_perDevice<float>(device));
  }
}
