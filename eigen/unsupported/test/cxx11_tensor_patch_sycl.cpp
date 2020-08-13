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

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <typename DataType, int DataLayout, typename IndexType>
static void test_simple_patch_sycl(const Eigen::SyclDevice& sycl_device){

  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  array<IndexType, 4> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  array<IndexType, 5> patchTensorRange;
  if (DataLayout == ColMajor) {
   patchTensorRange = {{1, 1, 1, 1, sizeDim1*sizeDim2*sizeDim3*sizeDim4}};
  }else{
     patchTensorRange = {{sizeDim1*sizeDim2*sizeDim3*sizeDim4,1, 1, 1, 1}};
  }

  Tensor<DataType, 4, DataLayout,IndexType> tensor(tensorRange);
  Tensor<DataType, 5, DataLayout,IndexType> no_patch(patchTensorRange);

  tensor.setRandom();

  array<ptrdiff_t, 4> patch_dims;
  patch_dims[0] = 1;
  patch_dims[1] = 1;
  patch_dims[2] = 1;
  patch_dims[3] = 1;

  const size_t tensorBuffSize =tensor.size()*sizeof(DataType);
  size_t patchTensorBuffSize =no_patch.size()*sizeof(DataType);
  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  DataType* gpu_data_no_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));

  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_no_patch(gpu_data_no_patch, patchTensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
  gpu_no_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
  sycl_device.memcpyDeviceToHost(no_patch.data(), gpu_data_no_patch, patchTensorBuffSize);

  if (DataLayout == ColMajor) {
    VERIFY_IS_EQUAL(no_patch.dimension(0), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(1), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(2), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(3), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(4), tensor.size());
  } else {
    VERIFY_IS_EQUAL(no_patch.dimension(0), tensor.size());
    VERIFY_IS_EQUAL(no_patch.dimension(1), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(2), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(3), 1);
    VERIFY_IS_EQUAL(no_patch.dimension(4), 1);
  }

  for (int i = 0; i < tensor.size(); ++i) {
    VERIFY_IS_EQUAL(tensor.data()[i], no_patch.data()[i]);
  }

  patch_dims[0] = 2;
  patch_dims[1] = 3;
  patch_dims[2] = 5;
  patch_dims[3] = 7;

  if (DataLayout == ColMajor) {
   patchTensorRange = {{sizeDim1,sizeDim2,sizeDim3,sizeDim4,1}};
  }else{
     patchTensorRange = {{1,sizeDim1,sizeDim2,sizeDim3,sizeDim4}};
  }
  Tensor<DataType, 5, DataLayout,IndexType> single_patch(patchTensorRange);
  patchTensorBuffSize =single_patch.size()*sizeof(DataType);
  DataType* gpu_data_single_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_single_patch(gpu_data_single_patch, patchTensorRange);

  gpu_single_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
  sycl_device.memcpyDeviceToHost(single_patch.data(), gpu_data_single_patch, patchTensorBuffSize);

  if (DataLayout == ColMajor) {
    VERIFY_IS_EQUAL(single_patch.dimension(0), 2);
    VERIFY_IS_EQUAL(single_patch.dimension(1), 3);
    VERIFY_IS_EQUAL(single_patch.dimension(2), 5);
    VERIFY_IS_EQUAL(single_patch.dimension(3), 7);
    VERIFY_IS_EQUAL(single_patch.dimension(4), 1);
  } else {
    VERIFY_IS_EQUAL(single_patch.dimension(0), 1);
    VERIFY_IS_EQUAL(single_patch.dimension(1), 2);
    VERIFY_IS_EQUAL(single_patch.dimension(2), 3);
    VERIFY_IS_EQUAL(single_patch.dimension(3), 5);
    VERIFY_IS_EQUAL(single_patch.dimension(4), 7);
  }

  for (int i = 0; i < tensor.size(); ++i) {
    VERIFY_IS_EQUAL(tensor.data()[i], single_patch.data()[i]);
  }
  patch_dims[0] = 1;
  patch_dims[1] = 2;
  patch_dims[2] = 2;
  patch_dims[3] = 1;
  
  if (DataLayout == ColMajor) {
   patchTensorRange = {{1,2,2,1,2*2*4*7}};
  }else{
     patchTensorRange = {{2*2*4*7, 1, 2,2,1}};
  }
  Tensor<DataType, 5, DataLayout,IndexType> twod_patch(patchTensorRange);
  patchTensorBuffSize =twod_patch.size()*sizeof(DataType);
  DataType* gpu_data_twod_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_twod_patch(gpu_data_twod_patch, patchTensorRange);

  gpu_twod_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
  sycl_device.memcpyDeviceToHost(twod_patch.data(), gpu_data_twod_patch, patchTensorBuffSize);

  if (DataLayout == ColMajor) {
    VERIFY_IS_EQUAL(twod_patch.dimension(0), 1);
    VERIFY_IS_EQUAL(twod_patch.dimension(1), 2);
    VERIFY_IS_EQUAL(twod_patch.dimension(2), 2);
    VERIFY_IS_EQUAL(twod_patch.dimension(3), 1);
    VERIFY_IS_EQUAL(twod_patch.dimension(4), 2*2*4*7);
  } else {
    VERIFY_IS_EQUAL(twod_patch.dimension(0), 2*2*4*7);
    VERIFY_IS_EQUAL(twod_patch.dimension(1), 1);
    VERIFY_IS_EQUAL(twod_patch.dimension(2), 2);
    VERIFY_IS_EQUAL(twod_patch.dimension(3), 2);
    VERIFY_IS_EQUAL(twod_patch.dimension(4), 1);
  }

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 7; ++l) {
          int patch_loc;
          if (DataLayout == ColMajor) {
            patch_loc = i + 2 * (j + 2 * (k + 4 * l));
          } else {
            patch_loc = l + 7 * (k + 4 * (j + 2 * i));
          }
          for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 2; ++y) {
              if (DataLayout == ColMajor) {
                VERIFY_IS_EQUAL(tensor(i,j+x,k+y,l), twod_patch(0,x,y,0,patch_loc));
              } else {
                VERIFY_IS_EQUAL(tensor(i,j+x,k+y,l), twod_patch(patch_loc,0,x,y,0));
              }
            }
          }
        }
      }
    }
  }

  patch_dims[0] = 1;
  patch_dims[1] = 2;
  patch_dims[2] = 3;
  patch_dims[3] = 5;

  if (DataLayout == ColMajor) {
   patchTensorRange = {{1,2,3,5,2*2*3*3}};
  }else{
     patchTensorRange = {{2*2*3*3, 1, 2,3,5}};
  }
  Tensor<DataType, 5, DataLayout,IndexType> threed_patch(patchTensorRange);
  patchTensorBuffSize =threed_patch.size()*sizeof(DataType);
  DataType* gpu_data_threed_patch  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_threed_patch(gpu_data_threed_patch, patchTensorRange);

  gpu_threed_patch.device(sycl_device)=gpu_tensor.extract_patches(patch_dims);
  sycl_device.memcpyDeviceToHost(threed_patch.data(), gpu_data_threed_patch, patchTensorBuffSize);

  if (DataLayout == ColMajor) {
    VERIFY_IS_EQUAL(threed_patch.dimension(0), 1);
    VERIFY_IS_EQUAL(threed_patch.dimension(1), 2);
    VERIFY_IS_EQUAL(threed_patch.dimension(2), 3);
    VERIFY_IS_EQUAL(threed_patch.dimension(3), 5);
    VERIFY_IS_EQUAL(threed_patch.dimension(4), 2*2*3*3);
  } else {
    VERIFY_IS_EQUAL(threed_patch.dimension(0), 2*2*3*3);
    VERIFY_IS_EQUAL(threed_patch.dimension(1), 1);
    VERIFY_IS_EQUAL(threed_patch.dimension(2), 2);
    VERIFY_IS_EQUAL(threed_patch.dimension(3), 3);
    VERIFY_IS_EQUAL(threed_patch.dimension(4), 5);
  }

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          int patch_loc;
          if (DataLayout == ColMajor) {
            patch_loc = i + 2 * (j + 2 * (k + 3 * l));
          } else {
            patch_loc = l + 3 * (k + 3 * (j + 2 * i));
          }
          for (int x = 0; x < 2; ++x) {
            for (int y = 0; y < 3; ++y) {
              for (int z = 0; z < 5; ++z) {
                if (DataLayout == ColMajor) {
                  VERIFY_IS_EQUAL(tensor(i,j+x,k+y,l+z), threed_patch(0,x,y,z,patch_loc));
                } else {
                  VERIFY_IS_EQUAL(tensor(i,j+x,k+y,l+z), threed_patch(patch_loc,0,x,y,z));
                }
              }
            }
          }
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data_tensor);
  sycl_device.deallocate(gpu_data_no_patch);
  sycl_device.deallocate(gpu_data_single_patch);
  sycl_device.deallocate(gpu_data_twod_patch);
  sycl_device.deallocate(gpu_data_threed_patch);
}

template<typename DataType, typename dev_Selector> void sycl_tensor_patch_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  test_simple_patch_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_simple_patch_sycl<DataType, ColMajor, int64_t>(sycl_device);
}
EIGEN_DECLARE_TEST(cxx11_tensor_patch_sycl)
{
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_tensor_patch_test_per_device<float>(device));
  }
}
