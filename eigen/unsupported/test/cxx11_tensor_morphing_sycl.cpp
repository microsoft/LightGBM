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
static void test_simple_reshape(const Eigen::SyclDevice& sycl_device)
{
  typename Tensor<DataType, 5 ,DataLayout, IndexType>::Dimensions dim1(2,3,1,7,1);
  typename Tensor<DataType, 3 ,DataLayout, IndexType>::Dimensions dim2(2,3,7);
  typename Tensor<DataType, 2 ,DataLayout, IndexType>::Dimensions dim3(6,7);
  typename Tensor<DataType, 2 ,DataLayout, IndexType>::Dimensions dim4(2,21);

  Tensor<DataType, 5, DataLayout, IndexType> tensor1(dim1);
  Tensor<DataType, 3, DataLayout, IndexType> tensor2(dim2);
  Tensor<DataType, 2, DataLayout, IndexType> tensor3(dim3);
  Tensor<DataType, 2, DataLayout, IndexType> tensor4(dim4);

  tensor1.setRandom();

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor1.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(tensor3.size()*sizeof(DataType)));
  DataType* gpu_data4  = static_cast<DataType*>(sycl_device.allocate(tensor4.size()*sizeof(DataType)));

  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu1(gpu_data1, dim1);
  TensorMap<Tensor<DataType, 3,DataLayout, IndexType>> gpu2(gpu_data2, dim2);
  TensorMap<Tensor<DataType, 2,DataLayout, IndexType>> gpu3(gpu_data3, dim3);
  TensorMap<Tensor<DataType, 2,DataLayout, IndexType>> gpu4(gpu_data4, dim4);

  sycl_device.memcpyHostToDevice(gpu_data1, tensor1.data(),(tensor1.size())*sizeof(DataType));

  gpu2.device(sycl_device)=gpu1.reshape(dim2);
  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor1.size())*sizeof(DataType));

  gpu3.device(sycl_device)=gpu1.reshape(dim3);
  sycl_device.memcpyDeviceToHost(tensor3.data(), gpu_data3,(tensor3.size())*sizeof(DataType));

  gpu4.device(sycl_device)=gpu1.reshape(dim2).reshape(dim4);
  sycl_device.memcpyDeviceToHost(tensor4.data(), gpu_data4,(tensor4.size())*sizeof(DataType));
  for (IndexType i = 0; i < 2; ++i){
    for (IndexType j = 0; j < 3; ++j){
      for (IndexType k = 0; k < 7; ++k){
        VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor2(i,j,k));      ///ColMajor
        if (static_cast<int>(DataLayout) == static_cast<int>(ColMajor)) {
          VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor3(i+2*j,k));    ///ColMajor
          VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor4(i,j+3*k));    ///ColMajor
        }
        else{
          //VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor2(i,j,k));      /// RowMajor
          VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor4(i,j*7 +k));   /// RowMajor
          VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor3(i*3 +j,k));   /// RowMajor
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
  sycl_device.deallocate(gpu_data3);
  sycl_device.deallocate(gpu_data4);
}


template<typename DataType, int DataLayout, typename IndexType>
static void test_reshape_as_lvalue(const Eigen::SyclDevice& sycl_device)
{
  typename Tensor<DataType, 3, DataLayout, IndexType>::Dimensions dim1(2,3,7);
  typename Tensor<DataType, 2, DataLayout, IndexType>::Dimensions dim2(6,7);
  typename Tensor<DataType, 5, DataLayout, IndexType>::Dimensions dim3(2,3,1,7,1);
  Tensor<DataType, 3, DataLayout, IndexType> tensor(dim1);
  Tensor<DataType, 2, DataLayout, IndexType> tensor2d(dim2);
  Tensor<DataType, 5, DataLayout, IndexType> tensor5d(dim3);

  tensor.setRandom();

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2d.size()*sizeof(DataType)));
  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(tensor5d.size()*sizeof(DataType)));

  TensorMap< Tensor<DataType, 3, DataLayout, IndexType> > gpu1(gpu_data1, dim1);
  TensorMap< Tensor<DataType, 2, DataLayout, IndexType> > gpu2(gpu_data2, dim2);
  TensorMap< Tensor<DataType, 5, DataLayout, IndexType> > gpu3(gpu_data3, dim3);

  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));

  gpu2.reshape(dim1).device(sycl_device)=gpu1;
  sycl_device.memcpyDeviceToHost(tensor2d.data(), gpu_data2,(tensor2d.size())*sizeof(DataType));

  gpu3.reshape(dim1).device(sycl_device)=gpu1;
  sycl_device.memcpyDeviceToHost(tensor5d.data(), gpu_data3,(tensor5d.size())*sizeof(DataType));


  for (IndexType i = 0; i < 2; ++i){
    for (IndexType j = 0; j < 3; ++j){
      for (IndexType k = 0; k < 7; ++k){
        VERIFY_IS_EQUAL(tensor5d(i,j,0,k,0), tensor(i,j,k));
        if (static_cast<int>(DataLayout) == static_cast<int>(ColMajor)) {
          VERIFY_IS_EQUAL(tensor2d(i+2*j,k), tensor(i,j,k));    ///ColMajor
        }
        else{
          VERIFY_IS_EQUAL(tensor2d(i*3 +j,k),tensor(i,j,k));   /// RowMajor
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
  sycl_device.deallocate(gpu_data3);
}


template <typename DataType, int DataLayout, typename IndexType>
static void test_simple_slice(const Eigen::SyclDevice &sycl_device)
{
  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  IndexType sizeDim5 = 11;
  array<IndexType, 5> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4, sizeDim5}};
  Tensor<DataType, 5,DataLayout, IndexType> tensor(tensorRange);
  tensor.setRandom();
  array<IndexType, 5> slice1_range ={{1, 1, 1, 1, 1}};
  Tensor<DataType, 5,DataLayout, IndexType> slice1(slice1_range);

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(slice1.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu1(gpu_data1, tensorRange);
  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu2(gpu_data2, slice1_range);
  Eigen::DSizes<IndexType, 5> indices(1,2,3,4,5);
  Eigen::DSizes<IndexType, 5> sizes(1,1,1,1,1);
  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
  gpu2.device(sycl_device)=gpu1.slice(indices, sizes);
  sycl_device.memcpyDeviceToHost(slice1.data(), gpu_data2,(slice1.size())*sizeof(DataType));
  VERIFY_IS_EQUAL(slice1(0,0,0,0,0), tensor(1,2,3,4,5));


  array<IndexType, 5> slice2_range ={{1,1,2,2,3}};
  Tensor<DataType, 5,DataLayout, IndexType> slice2(slice2_range);
  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(slice2.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu3(gpu_data3, slice2_range);
  Eigen::DSizes<IndexType, 5> indices2(1,1,3,4,5);
  Eigen::DSizes<IndexType, 5> sizes2(1,1,2,2,3);
  gpu3.device(sycl_device)=gpu1.slice(indices2, sizes2);
  sycl_device.memcpyDeviceToHost(slice2.data(), gpu_data3,(slice2.size())*sizeof(DataType));
  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 2; ++j) {
      for (IndexType k = 0; k < 3; ++k) {
        VERIFY_IS_EQUAL(slice2(0,0,i,j,k), tensor(1,1,3+i,4+j,5+k));
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
  sycl_device.deallocate(gpu_data3);
}


template <typename DataType, int DataLayout, typename IndexType>
static void test_strided_slice_as_rhs_sycl(const Eigen::SyclDevice &sycl_device)
{
  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  IndexType sizeDim5 = 11;
  typedef Eigen::DSizes<IndexType, 5> Index5;
  Index5 strides(1L,1L,1L,1L,1L);
  Index5 indicesStart(1L,2L,3L,4L,5L);
  Index5 indicesStop(2L,3L,4L,5L,6L);
  Index5 lengths(1L,1L,1L,1L,1L);

  array<IndexType, 5> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4, sizeDim5}};
  Tensor<DataType, 5, DataLayout, IndexType> tensor(tensorRange);
  tensor.setRandom();

  array<IndexType, 5> slice1_range ={{1, 1, 1, 1, 1}};
  Tensor<DataType, 5,DataLayout, IndexType> slice1(slice1_range);
  Tensor<DataType, 5, DataLayout, IndexType> slice_stride1(slice1_range);

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(slice1.size()*sizeof(DataType)));
  DataType* gpu_data_stride2  = static_cast<DataType*>(sycl_device.allocate(slice_stride1.size()*sizeof(DataType)));

  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu1(gpu_data1, tensorRange);
  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu2(gpu_data2, slice1_range);
  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu_stride2(gpu_data_stride2, slice1_range);

  Eigen::DSizes<IndexType, 5> indices(1,2,3,4,5);
  Eigen::DSizes<IndexType, 5> sizes(1,1,1,1,1);
  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
  gpu2.device(sycl_device)=gpu1.slice(indices, sizes);
  sycl_device.memcpyDeviceToHost(slice1.data(), gpu_data2,(slice1.size())*sizeof(DataType));

  gpu_stride2.device(sycl_device)=gpu1.stridedSlice(indicesStart,indicesStop,strides);
  sycl_device.memcpyDeviceToHost(slice_stride1.data(), gpu_data_stride2,(slice_stride1.size())*sizeof(DataType));

  VERIFY_IS_EQUAL(slice1(0,0,0,0,0), tensor(1,2,3,4,5));
  VERIFY_IS_EQUAL(slice_stride1(0,0,0,0,0), tensor(1,2,3,4,5));

  array<IndexType, 5> slice2_range ={{1,1,2,2,3}};
  Tensor<DataType, 5,DataLayout, IndexType> slice2(slice2_range);
  Tensor<DataType, 5, DataLayout, IndexType> strideSlice2(slice2_range);

  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(slice2.size()*sizeof(DataType)));
  DataType* gpu_data_stride3  = static_cast<DataType*>(sycl_device.allocate(strideSlice2.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu3(gpu_data3, slice2_range);
  TensorMap<Tensor<DataType, 5,DataLayout, IndexType>> gpu_stride3(gpu_data_stride3, slice2_range);
  Eigen::DSizes<IndexType, 5> indices2(1,1,3,4,5);
  Eigen::DSizes<IndexType, 5> sizes2(1,1,2,2,3);
  Index5 strides2(1L,1L,1L,1L,1L);
  Index5 indicesStart2(1L,1L,3L,4L,5L);
  Index5 indicesStop2(2L,2L,5L,6L,8L);

  gpu3.device(sycl_device)=gpu1.slice(indices2, sizes2);
  sycl_device.memcpyDeviceToHost(slice2.data(), gpu_data3,(slice2.size())*sizeof(DataType));

  gpu_stride3.device(sycl_device)=gpu1.stridedSlice(indicesStart2,indicesStop2,strides2);
  sycl_device.memcpyDeviceToHost(strideSlice2.data(), gpu_data_stride3,(strideSlice2.size())*sizeof(DataType));

  for (IndexType i = 0; i < 2; ++i) {
    for (IndexType j = 0; j < 2; ++j) {
      for (IndexType k = 0; k < 3; ++k) {
        VERIFY_IS_EQUAL(slice2(0,0,i,j,k), tensor(1,1,3+i,4+j,5+k));
        VERIFY_IS_EQUAL(strideSlice2(0,0,i,j,k), tensor(1,1,3+i,4+j,5+k));
      }
    }
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
  sycl_device.deallocate(gpu_data3);
}

template<typename DataType, int DataLayout, typename IndexType>
static void test_strided_slice_write_sycl(const Eigen::SyclDevice& sycl_device)
{
  typedef Tensor<DataType, 2, DataLayout, IndexType> Tensor2f;
  typedef Eigen::DSizes<IndexType, 2> Index2;
  IndexType sizeDim1 = 7L;
  IndexType sizeDim2 = 11L;
  array<IndexType, 2> tensorRange = {{sizeDim1, sizeDim2}};
  Tensor<DataType, 2, DataLayout, IndexType> tensor(tensorRange),tensor2(tensorRange);
  IndexType sliceDim1 = 2;
  IndexType sliceDim2 = 3;
  array<IndexType, 2> sliceRange = {{sliceDim1, sliceDim2}};
  Tensor2f slice(sliceRange);
  Index2 strides(1L,1L);
  Index2 indicesStart(3L,4L);
  Index2 indicesStop(5L,7L);
  Index2 lengths(2L,3L);

  DataType* gpu_data1  = static_cast<DataType*>(sycl_device.allocate(tensor.size()*sizeof(DataType)));
  DataType* gpu_data2  = static_cast<DataType*>(sycl_device.allocate(tensor2.size()*sizeof(DataType)));
  DataType* gpu_data3  = static_cast<DataType*>(sycl_device.allocate(slice.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu1(gpu_data1, tensorRange);
  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu2(gpu_data2, tensorRange);
  TensorMap<Tensor<DataType, 2,DataLayout,IndexType>> gpu3(gpu_data3, sliceRange);


  tensor.setRandom();
  sycl_device.memcpyHostToDevice(gpu_data1, tensor.data(),(tensor.size())*sizeof(DataType));
  gpu2.device(sycl_device)=gpu1;

  slice.setRandom();
  sycl_device.memcpyHostToDevice(gpu_data3, slice.data(),(slice.size())*sizeof(DataType));


  gpu1.slice(indicesStart,lengths).device(sycl_device)=gpu3;
  gpu2.stridedSlice(indicesStart,indicesStop,strides).device(sycl_device)=gpu3;
  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data1,(tensor.size())*sizeof(DataType));
  sycl_device.memcpyDeviceToHost(tensor2.data(), gpu_data2,(tensor2.size())*sizeof(DataType));

  for(IndexType i=0;i<sizeDim1;i++)
    for(IndexType j=0;j<sizeDim2;j++){
    VERIFY_IS_EQUAL(tensor(i,j), tensor2(i,j));
  }
  sycl_device.deallocate(gpu_data1);
  sycl_device.deallocate(gpu_data2);
  sycl_device.deallocate(gpu_data3);
}

template <typename OutIndex, typename DSizes>
Eigen::array<OutIndex, DSizes::count> To32BitDims(const DSizes& in) {
  Eigen::array<OutIndex, DSizes::count> out;
  for (int i = 0; i < DSizes::count; ++i) {
    out[i] = in[i];
  }
  return out;
}

template <class DataType, int DataLayout, typename IndexType, typename ConvertedIndexType>
int run_eigen(const SyclDevice& sycl_device) {
  using TensorI64 = Tensor<DataType, 5, DataLayout, IndexType>;
  using TensorI32 = Tensor<DataType, 5, DataLayout, ConvertedIndexType>;
  using TensorMI64 = TensorMap<TensorI64>;
  using TensorMI32 = TensorMap<TensorI32>;
  Eigen::array<IndexType, 5> tensor_range{{4, 1, 1, 1, 6}};
  Eigen::array<IndexType, 5> slice_range{{4, 1, 1, 1, 3}};

  TensorI64 out_tensor_gpu(tensor_range);
  TensorI64 out_tensor_cpu(tensor_range);
  out_tensor_cpu.setRandom();

  TensorI64 sub_tensor(slice_range);
  sub_tensor.setRandom();

  DataType* out_gpu_data = static_cast<DataType*>(sycl_device.allocate(out_tensor_cpu.size() * sizeof(DataType)));
  DataType* sub_gpu_data = static_cast<DataType*>(sycl_device.allocate(sub_tensor.size() * sizeof(DataType)));
  TensorMI64 out_gpu(out_gpu_data, tensor_range);
  TensorMI64 sub_gpu(sub_gpu_data, slice_range);

  sycl_device.memcpyHostToDevice(out_gpu_data, out_tensor_cpu.data(), out_tensor_cpu.size() * sizeof(DataType));
  sycl_device.memcpyHostToDevice(sub_gpu_data, sub_tensor.data(), sub_tensor.size() * sizeof(DataType));

  Eigen::array<ConvertedIndexType, 5> slice_offset_32{{0, 0, 0, 0, 3}};
  Eigen::array<ConvertedIndexType, 5> slice_range_32{{4, 1, 1, 1, 3}};
  TensorMI32 out_cpu_32(out_tensor_cpu.data(), To32BitDims<ConvertedIndexType>(out_tensor_cpu.dimensions()));
  TensorMI32 sub_cpu_32(sub_tensor.data(), To32BitDims<ConvertedIndexType>(sub_tensor.dimensions()));
  TensorMI32 out_gpu_32(out_gpu.data(), To32BitDims<ConvertedIndexType>(out_gpu.dimensions()));
  TensorMI32 sub_gpu_32(sub_gpu.data(), To32BitDims<ConvertedIndexType>(sub_gpu.dimensions()));

  out_gpu_32.slice(slice_offset_32, slice_range_32).device(sycl_device) = sub_gpu_32;

  out_cpu_32.slice(slice_offset_32, slice_range_32) = sub_cpu_32;

  sycl_device.memcpyDeviceToHost(out_tensor_gpu.data(), out_gpu_data, out_tensor_cpu.size() * sizeof(DataType));
  int has_err = 0;
  for (IndexType i = 0; i < out_tensor_cpu.size(); ++i) {
    auto exp = out_tensor_cpu(i);
    auto val = out_tensor_gpu(i);
    if (val != exp) {
      std::cout << "#" << i << " got " << val << " but expected " << exp << std::endl;
      has_err = 1;
    }
  }
  sycl_device.deallocate(out_gpu_data);
  sycl_device.deallocate(sub_gpu_data);
  return has_err;
}

template<typename DataType, typename dev_Selector> void sycl_morphing_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_simple_slice<DataType, RowMajor, int64_t>(sycl_device);
  test_simple_slice<DataType, ColMajor, int64_t>(sycl_device);
  test_simple_reshape<DataType, RowMajor, int64_t>(sycl_device);
  test_simple_reshape<DataType, ColMajor, int64_t>(sycl_device);
  test_reshape_as_lvalue<DataType, RowMajor, int64_t>(sycl_device);
  test_reshape_as_lvalue<DataType, ColMajor, int64_t>(sycl_device);
  test_strided_slice_write_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_strided_slice_write_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_strided_slice_as_rhs_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_strided_slice_as_rhs_sycl<DataType, RowMajor, int64_t>(sycl_device);
  run_eigen<float, RowMajor, long, int>(sycl_device); 
}
EIGEN_DECLARE_TEST(cxx11_tensor_morphing_sycl)
{
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_morphing_test_per_device<float>(device));
  }
}
