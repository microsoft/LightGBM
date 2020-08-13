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
static const int DataLayout = ColMajor;

template <typename DataType, typename IndexType>
static void test_single_voxel_patch_sycl(const Eigen::SyclDevice& sycl_device)
{

IndexType sizeDim0 = 4;
IndexType sizeDim1 = 2;
IndexType sizeDim2 = 3;
IndexType sizeDim3 = 5;
IndexType sizeDim4 = 7;
array<IndexType, 5> tensorColMajorRange = {{sizeDim0, sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
array<IndexType, 5> tensorRowMajorRange = {{sizeDim4, sizeDim3, sizeDim2, sizeDim1, sizeDim0}};
Tensor<DataType, 5, DataLayout,IndexType> tensor_col_major(tensorColMajorRange);
Tensor<DataType, 5, RowMajor,IndexType> tensor_row_major(tensorRowMajorRange);
tensor_col_major.setRandom();


  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 5, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
  TensorMap<Tensor<DataType, 5, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);

  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();


  // single volume patch: ColMajor
  array<IndexType, 6> patchColMajorTensorRange={{sizeDim0,1, 1, 1, sizeDim1*sizeDim2*sizeDim3, sizeDim4}};
  Tensor<DataType, 6, DataLayout,IndexType> single_voxel_patch_col_major(patchColMajorTensorRange);
  size_t patchTensorBuffSize =single_voxel_patch_col_major.size()*sizeof(DataType);
  DataType* gpu_data_single_voxel_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 6, DataLayout,IndexType>> gpu_single_voxel_patch_col_major(gpu_data_single_voxel_patch_col_major, patchColMajorTensorRange);
  gpu_single_voxel_patch_col_major.device(sycl_device)=gpu_col_major.extract_volume_patches(1, 1, 1);
  sycl_device.memcpyDeviceToHost(single_voxel_patch_col_major.data(), gpu_data_single_voxel_patch_col_major, patchTensorBuffSize);


  VERIFY_IS_EQUAL(single_voxel_patch_col_major.dimension(0), 4);
  VERIFY_IS_EQUAL(single_voxel_patch_col_major.dimension(1), 1);
  VERIFY_IS_EQUAL(single_voxel_patch_col_major.dimension(2), 1);
  VERIFY_IS_EQUAL(single_voxel_patch_col_major.dimension(3), 1);
  VERIFY_IS_EQUAL(single_voxel_patch_col_major.dimension(4), 2 * 3 * 5);
  VERIFY_IS_EQUAL(single_voxel_patch_col_major.dimension(5), 7);

  array<IndexType, 6> patchRowMajorTensorRange={{sizeDim4, sizeDim1*sizeDim2*sizeDim3, 1, 1, 1, sizeDim0}};
  Tensor<DataType, 6, RowMajor,IndexType> single_voxel_patch_row_major(patchRowMajorTensorRange);
  patchTensorBuffSize =single_voxel_patch_row_major.size()*sizeof(DataType);
  DataType* gpu_data_single_voxel_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 6, RowMajor,IndexType>> gpu_single_voxel_patch_row_major(gpu_data_single_voxel_patch_row_major, patchRowMajorTensorRange);
  gpu_single_voxel_patch_row_major.device(sycl_device)=gpu_row_major.extract_volume_patches(1, 1, 1);
  sycl_device.memcpyDeviceToHost(single_voxel_patch_row_major.data(), gpu_data_single_voxel_patch_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(0), 7);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(1), 2 * 3 * 5);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(2), 1);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(3), 1);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(4), 1);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(5), 4);

 sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));
 for (IndexType i = 0; i < tensor_col_major.size(); ++i) {
       VERIFY_IS_EQUAL(tensor_col_major.data()[i], single_voxel_patch_col_major.data()[i]);
    VERIFY_IS_EQUAL(tensor_row_major.data()[i], single_voxel_patch_row_major.data()[i]);
    VERIFY_IS_EQUAL(tensor_col_major.data()[i], tensor_row_major.data()[i]);
  }


  sycl_device.deallocate(gpu_data_col_major);
  sycl_device.deallocate(gpu_data_row_major);
  sycl_device.deallocate(gpu_data_single_voxel_patch_col_major);
  sycl_device.deallocate(gpu_data_single_voxel_patch_row_major);
}

template <typename DataType, typename IndexType>
static void test_entire_volume_patch_sycl(const Eigen::SyclDevice& sycl_device)
{
  const int depth = 4;
  const int patch_z = 2;
  const int patch_y = 3;
  const int patch_x = 5;
  const int batch = 7;

  array<IndexType, 5> tensorColMajorRange = {{depth, patch_z, patch_y, patch_x, batch}};
  array<IndexType, 5> tensorRowMajorRange = {{batch, patch_x, patch_y, patch_z, depth}};
  Tensor<DataType, 5, DataLayout,IndexType> tensor_col_major(tensorColMajorRange);
  Tensor<DataType, 5, RowMajor,IndexType> tensor_row_major(tensorRowMajorRange);
  tensor_col_major.setRandom();


    DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
    DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
    TensorMap<Tensor<DataType, 5, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
    TensorMap<Tensor<DataType, 5, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);

    sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
    gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
    sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));


    // single volume patch: ColMajor
    array<IndexType, 6> patchColMajorTensorRange={{depth,patch_z, patch_y, patch_x, patch_z*patch_y*patch_x, batch}};
    Tensor<DataType, 6, DataLayout,IndexType> entire_volume_patch_col_major(patchColMajorTensorRange);
    size_t patchTensorBuffSize =entire_volume_patch_col_major.size()*sizeof(DataType);
    DataType* gpu_data_entire_volume_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
    TensorMap<Tensor<DataType, 6, DataLayout,IndexType>> gpu_entire_volume_patch_col_major(gpu_data_entire_volume_patch_col_major, patchColMajorTensorRange);
    gpu_entire_volume_patch_col_major.device(sycl_device)=gpu_col_major.extract_volume_patches(patch_z, patch_y, patch_x);
    sycl_device.memcpyDeviceToHost(entire_volume_patch_col_major.data(), gpu_data_entire_volume_patch_col_major, patchTensorBuffSize);


//  Tensor<float, 5> tensor(depth, patch_z, patch_y, patch_x, batch);
//  tensor.setRandom();
//  Tensor<float, 5, RowMajor> tensor_row_major = tensor.swap_layout();

  //Tensor<float, 6> entire_volume_patch;
  //entire_volume_patch = tensor.extract_volume_patches(patch_z, patch_y, patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch_col_major.dimension(0), depth);
  VERIFY_IS_EQUAL(entire_volume_patch_col_major.dimension(1), patch_z);
  VERIFY_IS_EQUAL(entire_volume_patch_col_major.dimension(2), patch_y);
  VERIFY_IS_EQUAL(entire_volume_patch_col_major.dimension(3), patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch_col_major.dimension(4), patch_z * patch_y * patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch_col_major.dimension(5), batch);

//  Tensor<float, 6, RowMajor> entire_volume_patch_row_major;
  //entire_volume_patch_row_major = tensor_row_major.extract_volume_patches(patch_z, patch_y, patch_x);

  array<IndexType, 6> patchRowMajorTensorRange={{batch,patch_z*patch_y*patch_x, patch_x, patch_y, patch_z, depth}};
  Tensor<DataType, 6, RowMajor,IndexType> entire_volume_patch_row_major(patchRowMajorTensorRange);
  patchTensorBuffSize =entire_volume_patch_row_major.size()*sizeof(DataType);
  DataType* gpu_data_entire_volume_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 6, RowMajor,IndexType>> gpu_entire_volume_patch_row_major(gpu_data_entire_volume_patch_row_major, patchRowMajorTensorRange);
  gpu_entire_volume_patch_row_major.device(sycl_device)=gpu_row_major.extract_volume_patches(patch_z, patch_y, patch_x);
  sycl_device.memcpyDeviceToHost(entire_volume_patch_row_major.data(), gpu_data_entire_volume_patch_row_major, patchTensorBuffSize);


  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(0), batch);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(1), patch_z * patch_y * patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(2), patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(3), patch_y);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(4), patch_z);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(5), depth);

  const int dz = patch_z - 1;
  const int dy = patch_y - 1;
  const int dx = patch_x - 1;

  const int forward_pad_z = dz - dz / 2;
  const int forward_pad_y = dy - dy / 2;
  const int forward_pad_x = dx - dx / 2;

  for (int pz = 0; pz < patch_z; pz++) {
    for (int py = 0; py < patch_y; py++) {
      for (int px = 0; px < patch_x; px++) {
        const int patchId = pz + patch_z * (py + px * patch_y);
        for (int z = 0; z < patch_z; z++) {
          for (int y = 0; y < patch_y; y++) {
            for (int x = 0; x < patch_x; x++) {
              for (int b = 0; b < batch; b++) {
                for (int d = 0; d < depth; d++) {
                  float expected = 0.0f;
                  float expected_row_major = 0.0f;
                  const int eff_z = z - forward_pad_z + pz;
                  const int eff_y = y - forward_pad_y + py;
                  const int eff_x = x - forward_pad_x + px;
                  if (eff_z >= 0 && eff_y >= 0 && eff_x >= 0 &&
                      eff_z < patch_z && eff_y < patch_y && eff_x < patch_x) {
                    expected = tensor_col_major(d, eff_z, eff_y, eff_x, b);
                    expected_row_major = tensor_row_major(b, eff_x, eff_y, eff_z, d);
                  }
                  VERIFY_IS_EQUAL(entire_volume_patch_col_major(d, z, y, x, patchId, b), expected);
                  VERIFY_IS_EQUAL(entire_volume_patch_row_major(b, patchId, x, y, z, d), expected_row_major);
                }
              }
            }
          }
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data_col_major);
  sycl_device.deallocate(gpu_data_row_major);
  sycl_device.deallocate(gpu_data_entire_volume_patch_col_major);
  sycl_device.deallocate(gpu_data_entire_volume_patch_row_major);
}



template<typename DataType, typename dev_Selector> void sycl_tensor_volume_patch_test_per_device(dev_Selector s){
QueueInterface queueInterface(s);
auto sycl_device = Eigen::SyclDevice(&queueInterface);
std::cout << "Running on " << s.template get_info<cl::sycl::info::device::name>() << std::endl;
test_single_voxel_patch_sycl<DataType, int64_t>(sycl_device);
test_entire_volume_patch_sycl<DataType, int64_t>(sycl_device);
}
EIGEN_DECLARE_TEST(cxx11_tensor_volume_patch_sycl)
{
for (const auto& device :Eigen::get_sycl_supported_devices()) {
  CALL_SUBTEST(sycl_tensor_volume_patch_test_per_device<float>(device));
}
}
