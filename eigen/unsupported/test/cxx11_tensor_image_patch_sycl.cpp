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
static void test_simple_image_patch_sycl(const Eigen::SyclDevice& sycl_device)
{
  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  array<IndexType, 4> tensorColMajorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  array<IndexType, 4> tensorRowMajorRange = {{sizeDim4, sizeDim3, sizeDim2, sizeDim1}};
  Tensor<DataType, 4, DataLayout,IndexType> tensor_col_major(tensorColMajorRange);
  Tensor<DataType, 4, RowMajor,IndexType> tensor_row_major(tensorRowMajorRange);
  tensor_col_major.setRandom();

  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);

  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));

  VERIFY_IS_EQUAL(tensor_col_major.dimension(0), tensor_row_major.dimension(3));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(1), tensor_row_major.dimension(2));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(2), tensor_row_major.dimension(1));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(3), tensor_row_major.dimension(0));

  // Single pixel patch: ColMajor
  array<IndexType, 5> patchColMajorTensorRange={{sizeDim1, 1, 1, sizeDim2*sizeDim3, sizeDim4}};
  Tensor<DataType, 5, DataLayout,IndexType> single_patch_col_major(patchColMajorTensorRange);
  size_t patchTensorBuffSize =single_patch_col_major.size()*sizeof(DataType);
  DataType* gpu_data_single_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_single_patch_col_major(gpu_data_single_patch_col_major, patchColMajorTensorRange);
  gpu_single_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(1, 1);
  sycl_device.memcpyDeviceToHost(single_patch_col_major.data(), gpu_data_single_patch_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(single_patch_col_major.dimension(0), 2);
  VERIFY_IS_EQUAL(single_patch_col_major.dimension(1), 1);
  VERIFY_IS_EQUAL(single_patch_col_major.dimension(2), 1);
  VERIFY_IS_EQUAL(single_patch_col_major.dimension(3), 3*5);
  VERIFY_IS_EQUAL(single_patch_col_major.dimension(4), 7);

  // Single pixel patch: RowMajor
  array<IndexType, 5> patchRowMajorTensorRange={{sizeDim4, sizeDim2*sizeDim3, 1, 1, sizeDim1}};
  Tensor<DataType, 5, RowMajor,IndexType> single_patch_row_major(patchRowMajorTensorRange);
  patchTensorBuffSize =single_patch_row_major.size()*sizeof(DataType);
  DataType* gpu_data_single_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_single_patch_row_major(gpu_data_single_patch_row_major, patchRowMajorTensorRange);
  gpu_single_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(1, 1);
  sycl_device.memcpyDeviceToHost(single_patch_row_major.data(), gpu_data_single_patch_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(single_patch_row_major.dimension(0), 7);
  VERIFY_IS_EQUAL(single_patch_row_major.dimension(1), 3*5);
  VERIFY_IS_EQUAL(single_patch_row_major.dimension(2), 1);
  VERIFY_IS_EQUAL(single_patch_row_major.dimension(3), 1);
  VERIFY_IS_EQUAL(single_patch_row_major.dimension(4), 2);

  for (IndexType i = 0; i < tensor_col_major.size(); ++i) {
    // ColMajor
    if (tensor_col_major.data()[i] != single_patch_col_major.data()[i]) {
      std::cout << "Mismatch detected at index colmajor " << i << " : "
           << tensor_col_major.data()[i] << " vs " << single_patch_col_major.data()[i]
           << std::endl;
    }
    VERIFY_IS_EQUAL(single_patch_col_major.data()[i], tensor_col_major.data()[i]);
    // RowMajor
    if (tensor_row_major.data()[i] != single_patch_row_major.data()[i]) {
      std::cout << "Mismatch detected at index row major" << i << " : "
           << tensor_row_major.data()[i] << " vs "
           << single_patch_row_major.data()[i] << std::endl;
    }
    VERIFY_IS_EQUAL(single_patch_row_major.data()[i],
                    tensor_row_major.data()[i]);
    VERIFY_IS_EQUAL(tensor_col_major.data()[i], tensor_row_major.data()[i]);
    VERIFY_IS_EQUAL(single_patch_col_major.data()[i],
                    single_patch_row_major.data()[i]);
  }


  // Entire image patch: ColMajor
  patchColMajorTensorRange={{sizeDim1, sizeDim2, sizeDim3, sizeDim2*sizeDim3, sizeDim4}};
  Tensor<DataType, 5, DataLayout,IndexType> entire_image_patch_col_major(patchColMajorTensorRange);
  patchTensorBuffSize =entire_image_patch_col_major.size()*sizeof(DataType);
  DataType* gpu_data_entire_image_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_entire_image_patch_col_major(gpu_data_entire_image_patch_col_major, patchColMajorTensorRange);
  gpu_entire_image_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(3, 5);
  sycl_device.memcpyDeviceToHost(entire_image_patch_col_major.data(), gpu_data_entire_image_patch_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(entire_image_patch_col_major.dimension(0), 2);
  VERIFY_IS_EQUAL(entire_image_patch_col_major.dimension(1), 3);
  VERIFY_IS_EQUAL(entire_image_patch_col_major.dimension(2), 5);
  VERIFY_IS_EQUAL(entire_image_patch_col_major.dimension(3), 3*5);
  VERIFY_IS_EQUAL(entire_image_patch_col_major.dimension(4), 7);

  // Entire image patch: RowMajor
  patchRowMajorTensorRange={{sizeDim4, sizeDim2*sizeDim3, sizeDim3, sizeDim2, sizeDim1}};
  Tensor<DataType, 5, RowMajor,IndexType> entire_image_patch_row_major(patchRowMajorTensorRange);
  patchTensorBuffSize =entire_image_patch_row_major.size()*sizeof(DataType);
  DataType* gpu_data_entire_image_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_entire_image_patch_row_major(gpu_data_entire_image_patch_row_major, patchRowMajorTensorRange);
  gpu_entire_image_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(3, 5);
  sycl_device.memcpyDeviceToHost(entire_image_patch_row_major.data(), gpu_data_entire_image_patch_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(entire_image_patch_row_major.dimension(0), 7);
  VERIFY_IS_EQUAL(entire_image_patch_row_major.dimension(1), 3*5);
  VERIFY_IS_EQUAL(entire_image_patch_row_major.dimension(2), 5);
  VERIFY_IS_EQUAL(entire_image_patch_row_major.dimension(3), 3);
  VERIFY_IS_EQUAL(entire_image_patch_row_major.dimension(4), 2);

  for (IndexType i = 0; i < 3; ++i) {
    for (IndexType j = 0; j < 5; ++j) {
      IndexType patchId = i+3*j;
      for (IndexType r = 0; r < 3; ++r) {
        for (IndexType c = 0; c < 5; ++c) {
          for (IndexType d = 0; d < 2; ++d) {
            for (IndexType b = 0; b < 7; ++b) {
              DataType expected_col_major = 0.0f;
              DataType expected_row_major = 0.0f;
              if (r-1+i >= 0 && c-2+j >= 0 && r-1+i < 3 && c-2+j < 5) {
                expected_col_major = tensor_col_major(d, r-1+i, c-2+j, b);
                expected_row_major = tensor_row_major(b, c-2+j, r-1+i, d);
              }
              // ColMajor
              if (entire_image_patch_col_major(d, r, c, patchId, b) != expected_col_major) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(entire_image_patch_col_major(d, r, c, patchId, b), expected_col_major);
              // RowMajor
              if (entire_image_patch_row_major(b, patchId, c, r, d) !=
                  expected_row_major) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j
                     << " r=" << r << " c=" << c << " d=" << d << " b=" << b
                     << std::endl;
              }
              VERIFY_IS_EQUAL(entire_image_patch_row_major(b, patchId, c, r, d),
                              expected_row_major);
              // Check that ColMajor and RowMajor agree.
              VERIFY_IS_EQUAL(expected_col_major, expected_row_major);
            }
          }
        }
      }
    }
  }

  // 2D patch: ColMajor
  patchColMajorTensorRange={{sizeDim1, 2, 2, sizeDim2*sizeDim3, sizeDim4}};
  Tensor<DataType, 5, DataLayout,IndexType> twod_patch_col_major(patchColMajorTensorRange);
  patchTensorBuffSize =twod_patch_col_major.size()*sizeof(DataType);
  DataType* gpu_data_twod_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_twod_patch_col_major(gpu_data_twod_patch_col_major, patchColMajorTensorRange);
  gpu_twod_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(2, 2);
  sycl_device.memcpyDeviceToHost(twod_patch_col_major.data(), gpu_data_twod_patch_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(twod_patch_col_major.dimension(0), 2);
  VERIFY_IS_EQUAL(twod_patch_col_major.dimension(1), 2);
  VERIFY_IS_EQUAL(twod_patch_col_major.dimension(2), 2);
  VERIFY_IS_EQUAL(twod_patch_col_major.dimension(3), 3*5);
  VERIFY_IS_EQUAL(twod_patch_col_major.dimension(4), 7);

  // 2D patch: RowMajor
  patchRowMajorTensorRange={{sizeDim4, sizeDim2*sizeDim3, 2, 2, sizeDim1}};
  Tensor<DataType, 5, RowMajor,IndexType> twod_patch_row_major(patchRowMajorTensorRange);
  patchTensorBuffSize =twod_patch_row_major.size()*sizeof(DataType);
  DataType* gpu_data_twod_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_twod_patch_row_major(gpu_data_twod_patch_row_major, patchRowMajorTensorRange);
  gpu_twod_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(2, 2);
  sycl_device.memcpyDeviceToHost(twod_patch_row_major.data(), gpu_data_twod_patch_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(twod_patch_row_major.dimension(0), 7);
  VERIFY_IS_EQUAL(twod_patch_row_major.dimension(1), 3*5);
  VERIFY_IS_EQUAL(twod_patch_row_major.dimension(2), 2);
  VERIFY_IS_EQUAL(twod_patch_row_major.dimension(3), 2);
  VERIFY_IS_EQUAL(twod_patch_row_major.dimension(4), 2);


  // Based on the calculation described in TensorTraits.h, padding happens to be 0.
  IndexType row_padding = 0;
  IndexType col_padding = 0;
  IndexType stride = 1;

  for (IndexType i = 0; i < 3; ++i) {
    for (IndexType j = 0; j < 5; ++j) {
      IndexType patchId = i+3*j;
      for (IndexType r = 0; r < 2; ++r) {
        for (IndexType c = 0; c < 2; ++c) {
          for (IndexType d = 0; d < 2; ++d) {
            for (IndexType b = 0; b < 7; ++b) {
              DataType expected_col_major = 0.0f;
              DataType expected_row_major = 0.0f;
              IndexType row_offset = r*stride + i - row_padding;
              IndexType col_offset = c*stride + j - col_padding;
              // ColMajor
              if (row_offset >= 0 && col_offset >= 0 && row_offset < tensor_col_major.dimension(1) && col_offset < tensor_col_major.dimension(2)) {
                expected_col_major = tensor_col_major(d, row_offset, col_offset, b);
              }
              if (twod_patch_col_major(d, r, c, patchId, b) != expected_col_major) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(twod_patch_col_major(d, r, c, patchId, b), expected_col_major);

              // RowMajor
              if (row_offset >= 0 && col_offset >= 0 && row_offset < tensor_row_major.dimension(2) && col_offset < tensor_row_major.dimension(1)) {
                expected_row_major = tensor_row_major(b, col_offset, row_offset, d);

              }
              if (twod_patch_row_major(b, patchId, c, r, d) != expected_row_major) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(twod_patch_row_major(b, patchId, c, r, d), expected_row_major);
              // Check that ColMajor and RowMajor agree.
              VERIFY_IS_EQUAL(expected_col_major, expected_row_major);
            }
          }
        }
      }
    }
  }

  sycl_device.deallocate(gpu_data_col_major);
  sycl_device.deallocate(gpu_data_row_major);
  sycl_device.deallocate(gpu_data_single_patch_col_major);
  sycl_device.deallocate(gpu_data_single_patch_row_major);
  sycl_device.deallocate(gpu_data_entire_image_patch_col_major);
  sycl_device.deallocate(gpu_data_entire_image_patch_row_major);
  sycl_device.deallocate(gpu_data_twod_patch_col_major);
  sycl_device.deallocate(gpu_data_twod_patch_row_major);

}


// Verifies VALID padding (no padding) with incrementing values.
template <typename DataType, typename IndexType>
static void test_patch_padding_valid_sycl(const Eigen::SyclDevice& sycl_device){
  IndexType input_depth = 3;
  IndexType input_rows = 3;
  IndexType input_cols = 3;
  IndexType input_batches = 1;
  IndexType ksize = 2;  // Corresponds to the Rows and Cols for tensor.extract_image_patches<>.
  IndexType stride = 2;  // Only same stride is supported.

  array<IndexType, 4> tensorColMajorRange = {{input_depth, input_rows, input_cols, input_batches}};
  array<IndexType, 4> tensorRowMajorRange = {{input_batches, input_cols, input_rows, input_depth}};
  Tensor<DataType, 4, DataLayout,IndexType> tensor_col_major(tensorColMajorRange);
  Tensor<DataType, 4, RowMajor,IndexType> tensor_row_major(tensorRowMajorRange);

  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);

  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));

  VERIFY_IS_EQUAL(tensor_col_major.dimension(0), tensor_row_major.dimension(3));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(1), tensor_row_major.dimension(2));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(2), tensor_row_major.dimension(1));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(3), tensor_row_major.dimension(0));

  // Initializes tensor with incrementing numbers.
  for (IndexType i = 0; i < tensor_col_major.size(); ++i) {
    tensor_col_major.data()[i] = i + 1;
  }
  // ColMajor
  array<IndexType, 5> patchColMajorTensorRange={{input_depth, ksize, ksize, 1, input_batches}};
  Tensor<DataType, 5, DataLayout,IndexType> result_col_major(patchColMajorTensorRange);
  size_t patchTensorBuffSize =result_col_major.size()*sizeof(DataType);
  DataType* gpu_data_result_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_result_col_major(gpu_data_result_col_major, patchColMajorTensorRange);
  gpu_result_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
  sycl_device.memcpyDeviceToHost(result_col_major.data(), gpu_data_result_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(result_col_major.dimension(0), input_depth);  // depth
  VERIFY_IS_EQUAL(result_col_major.dimension(1), ksize);  // kernel rows
  VERIFY_IS_EQUAL(result_col_major.dimension(2), ksize);  // kernel cols
  VERIFY_IS_EQUAL(result_col_major.dimension(3), 1);  // number of patches
  VERIFY_IS_EQUAL(result_col_major.dimension(4), input_batches);  // number of batches

  // RowMajor
  array<IndexType, 5> patchRowMajorTensorRange={{input_batches, 1, ksize, ksize, input_depth }};
  Tensor<DataType, 5, RowMajor,IndexType> result_row_major(patchRowMajorTensorRange);
  patchTensorBuffSize =result_row_major.size()*sizeof(DataType);
  DataType* gpu_data_result_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_result_row_major(gpu_data_result_row_major, patchRowMajorTensorRange);
  gpu_result_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
  sycl_device.memcpyDeviceToHost(result_row_major.data(), gpu_data_result_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(result_col_major.dimension(0), result_row_major.dimension(4));
  VERIFY_IS_EQUAL(result_col_major.dimension(1), result_row_major.dimension(3));
  VERIFY_IS_EQUAL(result_col_major.dimension(2), result_row_major.dimension(2));
  VERIFY_IS_EQUAL(result_col_major.dimension(3), result_row_major.dimension(1));
  VERIFY_IS_EQUAL(result_col_major.dimension(4), result_row_major.dimension(0));

  // No padding is carried out.
  IndexType row_padding = 0;
  IndexType col_padding = 0;

  for (IndexType i = 0; (i+stride+ksize-1) < input_rows; i += stride) {  // input rows
    for (IndexType j = 0; (j+stride+ksize-1) < input_cols; j += stride) {  // input cols
      IndexType patchId = i+input_rows*j;
      for (IndexType r = 0; r < ksize; ++r) {  // patch rows
        for (IndexType c = 0; c < ksize; ++c) {  // patch cols
          for (IndexType d = 0; d < input_depth; ++d) {  // depth
            for (IndexType b = 0; b < input_batches; ++b) {  // batch
              DataType expected_col_major = 0.0f;
              DataType expected_row_major = 0.0f;
              IndexType row_offset = r + i - row_padding;
              IndexType col_offset = c + j - col_padding;
              if (row_offset >= 0 && col_offset >= 0 && row_offset < input_rows && col_offset < input_cols) {
                expected_col_major = tensor_col_major(d, row_offset, col_offset, b);
                expected_row_major = tensor_row_major(b, col_offset, row_offset, d);
              }
              // ColMajor
              if (result_col_major(d, r, c, patchId, b) != expected_col_major) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(result_col_major(d, r, c, patchId, b), expected_col_major);
              // RowMajor
              if (result_row_major(b, patchId, c, r, d) != expected_row_major) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(result_row_major(b, patchId, c, r, d), expected_row_major);
              // Check that ColMajor and RowMajor agree.
              VERIFY_IS_EQUAL(expected_col_major, expected_row_major);
            }
          }
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data_col_major);
  sycl_device.deallocate(gpu_data_row_major);
  sycl_device.deallocate(gpu_data_result_col_major);
  sycl_device.deallocate(gpu_data_result_row_major);
}

// Verifies VALID padding (no padding) with the same value.
template <typename DataType, typename IndexType>
static void test_patch_padding_valid_same_value_sycl(const Eigen::SyclDevice& sycl_device){
  IndexType input_depth = 1;
  IndexType input_rows = 5;
  IndexType input_cols = 5;
  IndexType input_batches = 2;
  IndexType ksize = 3;  // Corresponds to the Rows and Cols for tensor.extract_image_patches<>.
  IndexType stride = 2;  // Only same stride is supported.
  // ColMajor

  array<IndexType, 4> tensorColMajorRange = {{input_depth, input_rows, input_cols, input_batches}};
  array<IndexType, 4> tensorRowMajorRange = {{input_batches, input_cols, input_rows, input_depth}};
  Tensor<DataType, 4, DataLayout,IndexType> tensor_col_major(tensorColMajorRange);
  Tensor<DataType, 4, RowMajor,IndexType> tensor_row_major(tensorRowMajorRange);

  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);
  gpu_col_major.device(sycl_device)=gpu_col_major.constant(11.0f);
  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
  sycl_device.memcpyDeviceToHost(tensor_col_major.data(), gpu_data_col_major, (tensor_col_major.size())*sizeof(DataType));
  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_row_major.size())*sizeof(DataType));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(0), tensor_row_major.dimension(3));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(1), tensor_row_major.dimension(2));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(2), tensor_row_major.dimension(1));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(3), tensor_row_major.dimension(0));

  array<IndexType, 5> patchColMajorTensorRange={{input_depth, ksize, ksize, 4, input_batches}};
  Tensor<DataType, 5, DataLayout,IndexType> result_col_major(patchColMajorTensorRange);
  size_t patchTensorBuffSize =result_col_major.size()*sizeof(DataType);
  DataType* gpu_data_result_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_result_col_major(gpu_data_result_col_major, patchColMajorTensorRange);
  gpu_result_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
  sycl_device.memcpyDeviceToHost(result_col_major.data(), gpu_data_result_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(result_col_major.dimension(0), input_depth);  // depth
  VERIFY_IS_EQUAL(result_col_major.dimension(1), ksize);  // kernel rows
  VERIFY_IS_EQUAL(result_col_major.dimension(2), ksize);  // kernel cols
  VERIFY_IS_EQUAL(result_col_major.dimension(3), 4);  // number of patches
  VERIFY_IS_EQUAL(result_col_major.dimension(4), input_batches);  // number of batches

  // RowMajor
  array<IndexType, 5> patchRowMajorTensorRange={{input_batches, 4, ksize, ksize, input_depth }};
  Tensor<DataType, 5, RowMajor,IndexType> result_row_major(patchRowMajorTensorRange);
  patchTensorBuffSize =result_row_major.size()*sizeof(DataType);
  DataType* gpu_data_result_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_result_row_major(gpu_data_result_row_major, patchRowMajorTensorRange);
  gpu_result_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(ksize, ksize, stride, stride, 1, 1, PADDING_VALID);
  sycl_device.memcpyDeviceToHost(result_row_major.data(), gpu_data_result_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(result_col_major.dimension(0), result_row_major.dimension(4));
  VERIFY_IS_EQUAL(result_col_major.dimension(1), result_row_major.dimension(3));
  VERIFY_IS_EQUAL(result_col_major.dimension(2), result_row_major.dimension(2));
  VERIFY_IS_EQUAL(result_col_major.dimension(3), result_row_major.dimension(1));
  VERIFY_IS_EQUAL(result_col_major.dimension(4), result_row_major.dimension(0));

  // No padding is carried out.
  IndexType row_padding = 0;
  IndexType col_padding = 0;

  for (IndexType i = 0; (i+stride+ksize-1) <= input_rows; i += stride) {  // input rows
    for (IndexType j = 0; (j+stride+ksize-1) <= input_cols; j += stride) {  // input cols
      IndexType patchId = i+input_rows*j;
      for (IndexType r = 0; r < ksize; ++r) {  // patch rows
        for (IndexType c = 0; c < ksize; ++c) {  // patch cols
          for (IndexType d = 0; d < input_depth; ++d) {  // depth
            for (IndexType b = 0; b < input_batches; ++b) {  // batch
              DataType expected_col_major = 0.0f;
              DataType expected_row_major = 0.0f;
              IndexType row_offset = r + i - row_padding;
              IndexType col_offset = c + j - col_padding;
              if (row_offset >= 0 && col_offset >= 0 && row_offset < input_rows && col_offset < input_cols) {
                expected_col_major = tensor_col_major(d, row_offset, col_offset, b);
                expected_row_major = tensor_row_major(b, col_offset, row_offset, d);
              }
              // ColMajor
              if (result_col_major(d, r, c, patchId, b) != expected_col_major) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(result_col_major(d, r, c, patchId, b), expected_col_major);
              // RowMajor
              if (result_row_major(b, patchId, c, r, d) != expected_row_major) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(result_row_major(b, patchId, c, r, d), expected_row_major);
              // Check that ColMajor and RowMajor agree.
              VERIFY_IS_EQUAL(expected_col_major, expected_row_major);
            }
          }
        }
      }
    }
  }
}

// Verifies SAME padding.
template <typename DataType, typename IndexType>
static void test_patch_padding_same_sycl(const Eigen::SyclDevice& sycl_device){
  IndexType input_depth = 3;
  IndexType input_rows = 4;
  IndexType input_cols = 2;
  IndexType input_batches = 1;
  IndexType ksize = 2;  // Corresponds to the Rows and Cols for tensor.extract_image_patches<>.
  IndexType stride = 2;  // Only same stride is supported.

  // ColMajor
  array<IndexType, 4> tensorColMajorRange = {{input_depth, input_rows, input_cols, input_batches}};
  array<IndexType, 4> tensorRowMajorRange = {{input_batches, input_cols, input_rows, input_depth}};
  Tensor<DataType, 4, DataLayout,IndexType> tensor_col_major(tensorColMajorRange);
  Tensor<DataType, 4, RowMajor,IndexType> tensor_row_major(tensorRowMajorRange);

  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
  TensorMap<Tensor<DataType, 4, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);

  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_col_major.size())*sizeof(DataType));

  VERIFY_IS_EQUAL(tensor_col_major.dimension(0), tensor_row_major.dimension(3));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(1), tensor_row_major.dimension(2));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(2), tensor_row_major.dimension(1));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(3), tensor_row_major.dimension(0));

  // Initializes tensor with incrementing numbers.
  for (IndexType i = 0; i < tensor_col_major.size(); ++i) {
    tensor_col_major.data()[i] = i + 1;
  }

array<IndexType, 5> patchColMajorTensorRange={{input_depth, ksize, ksize, 2, input_batches}};
Tensor<DataType, 5, DataLayout,IndexType> result_col_major(patchColMajorTensorRange);
size_t patchTensorBuffSize =result_col_major.size()*sizeof(DataType);
DataType* gpu_data_result_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_result_col_major(gpu_data_result_col_major, patchColMajorTensorRange);
gpu_result_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(ksize, ksize, stride, stride, PADDING_SAME);
sycl_device.memcpyDeviceToHost(result_col_major.data(), gpu_data_result_col_major, patchTensorBuffSize);


  VERIFY_IS_EQUAL(result_col_major.dimension(0), input_depth);  // depth
  VERIFY_IS_EQUAL(result_col_major.dimension(1), ksize);  // kernel rows
  VERIFY_IS_EQUAL(result_col_major.dimension(2), ksize);  // kernel cols
  VERIFY_IS_EQUAL(result_col_major.dimension(3), 2);  // number of patches
  VERIFY_IS_EQUAL(result_col_major.dimension(4), input_batches);  // number of batches

  // RowMajor

  array<IndexType, 5> patchRowMajorTensorRange={{input_batches, 2, ksize, ksize, input_depth }};
  Tensor<DataType, 5, RowMajor,IndexType> result_row_major(patchRowMajorTensorRange);
  patchTensorBuffSize =result_row_major.size()*sizeof(DataType);
  DataType* gpu_data_result_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_result_row_major(gpu_data_result_row_major, patchRowMajorTensorRange);
  gpu_result_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(ksize, ksize, stride, stride, PADDING_SAME);
  sycl_device.memcpyDeviceToHost(result_row_major.data(), gpu_data_result_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(result_col_major.dimension(0), result_row_major.dimension(4));
  VERIFY_IS_EQUAL(result_col_major.dimension(1), result_row_major.dimension(3));
  VERIFY_IS_EQUAL(result_col_major.dimension(2), result_row_major.dimension(2));
  VERIFY_IS_EQUAL(result_col_major.dimension(3), result_row_major.dimension(1));
  VERIFY_IS_EQUAL(result_col_major.dimension(4), result_row_major.dimension(0));

  // Based on the calculation described in TensorTraits.h, padding happens to be 0.
  IndexType row_padding = 0;
  IndexType col_padding = 0;

  for (IndexType i = 0; (i+stride+ksize-1) <= input_rows; i += stride) {  // input rows
    for (IndexType j = 0; (j+stride+ksize-1) <= input_cols; j += stride) {  // input cols
      IndexType patchId = i+input_rows*j;
      for (IndexType r = 0; r < ksize; ++r) {  // patch rows
        for (IndexType c = 0; c < ksize; ++c) {  // patch cols
          for (IndexType d = 0; d < input_depth; ++d) {  // depth
            for (IndexType b = 0; b < input_batches; ++b) {  // batch
              DataType expected_col_major = 0.0f;
              DataType expected_row_major = 0.0f;
              IndexType row_offset = r*stride + i - row_padding;
              IndexType col_offset = c*stride + j - col_padding;
              if (row_offset >= 0 && col_offset >= 0 && row_offset < input_rows && col_offset < input_cols) {
                expected_col_major = tensor_col_major(d, row_offset, col_offset, b);
                expected_row_major = tensor_row_major(b, col_offset, row_offset, d);
              }
              // ColMajor
              if (result_col_major(d, r, c, patchId, b) != expected_col_major) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(result_col_major(d, r, c, patchId, b), expected_col_major);
              // RowMajor
              if (result_row_major(b, patchId, c, r, d) != expected_row_major) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(result_row_major(b, patchId, c, r, d), expected_row_major);
              // Check that ColMajor and RowMajor agree.
              VERIFY_IS_EQUAL(expected_col_major, expected_row_major);
            }
          }
        }
      }
    }
  }
}


template <typename DataType, typename IndexType>
static void test_patch_no_extra_dim_sycl(const Eigen::SyclDevice& sycl_device){

  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;

  // ColMajor
  array<IndexType, 3> tensorColMajorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  array<IndexType, 3> tensorRowMajorRange = {{sizeDim3, sizeDim2, sizeDim1}};
  Tensor<DataType, 3, DataLayout,IndexType> tensor_col_major(tensorColMajorRange);
  tensor_col_major.setRandom();
  Tensor<DataType, 3, RowMajor,IndexType> tensor_row_major(tensorRowMajorRange);

  DataType* gpu_data_col_major  = static_cast<DataType*>(sycl_device.allocate(tensor_col_major.size()*sizeof(DataType)));
  DataType* gpu_data_row_major  = static_cast<DataType*>(sycl_device.allocate(tensor_row_major.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 3, ColMajor, IndexType>> gpu_col_major(gpu_data_col_major, tensorColMajorRange);
  TensorMap<Tensor<DataType, 3, RowMajor, IndexType>> gpu_row_major(gpu_data_row_major, tensorRowMajorRange);

  sycl_device.memcpyHostToDevice(gpu_data_col_major, tensor_col_major.data(),(tensor_col_major.size())*sizeof(DataType));
  gpu_row_major.device(sycl_device)=gpu_col_major.swap_layout();
  sycl_device.memcpyDeviceToHost(tensor_row_major.data(), gpu_data_row_major, (tensor_row_major.size())*sizeof(DataType));

  VERIFY_IS_EQUAL(tensor_col_major.dimension(0), tensor_row_major.dimension(2));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(1), tensor_row_major.dimension(1));
  VERIFY_IS_EQUAL(tensor_col_major.dimension(2), tensor_row_major.dimension(0));


  // Single pixel patch: ColMajor
  array<IndexType, 4> patchColMajorTensorRange={{sizeDim1, 1, 1, sizeDim2*sizeDim3}};
  Tensor<DataType, 4, DataLayout,IndexType> single_patch_col_major(patchColMajorTensorRange);
  size_t patchTensorBuffSize =single_patch_col_major.size()*sizeof(DataType);
  DataType* gpu_data_single_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_single_patch_col_major(gpu_data_single_patch_col_major, patchColMajorTensorRange);
  gpu_single_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(1, 1);
  sycl_device.memcpyDeviceToHost(single_patch_col_major.data(), gpu_data_single_patch_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(single_patch_col_major.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(single_patch_col_major.dimension(1), 1);
  VERIFY_IS_EQUAL(single_patch_col_major.dimension(2), 1);
  VERIFY_IS_EQUAL(single_patch_col_major.dimension(3), sizeDim2*sizeDim3);

  // Single pixel patch: RowMajor
  array<IndexType, 4> patchRowMajorTensorRange={{sizeDim2*sizeDim3, 1, 1, sizeDim1}};
  Tensor<DataType, 4, RowMajor,IndexType> single_patch_row_major(patchRowMajorTensorRange);
  patchTensorBuffSize =single_patch_row_major.size()*sizeof(DataType);
  DataType* gpu_data_single_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 4, RowMajor,IndexType>> gpu_single_patch_row_major(gpu_data_single_patch_row_major, patchRowMajorTensorRange);
  gpu_single_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(1, 1);
  sycl_device.memcpyDeviceToHost(single_patch_row_major.data(), gpu_data_single_patch_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(single_patch_row_major.dimension(0), sizeDim2*sizeDim3);
  VERIFY_IS_EQUAL(single_patch_row_major.dimension(1), 1);
  VERIFY_IS_EQUAL(single_patch_row_major.dimension(2), 1);
  VERIFY_IS_EQUAL(single_patch_row_major.dimension(3), sizeDim1);

  for (IndexType i = 0; i < tensor_col_major.size(); ++i) {
    // ColMajor
    if (tensor_col_major.data()[i] != single_patch_col_major.data()[i]) {
      std::cout << "Mismatch detected at index " << i << " : " << tensor_col_major.data()[i] << " vs " << single_patch_col_major.data()[i] << std::endl;
    }
    VERIFY_IS_EQUAL(single_patch_col_major.data()[i], tensor_col_major.data()[i]);
    // RowMajor
    if (tensor_row_major.data()[i] != single_patch_row_major.data()[i]) {
      std::cout << "Mismatch detected at index " << i << " : "
           << tensor_col_major.data()[i] << " vs "
           << single_patch_row_major.data()[i] << std::endl;
    }
    VERIFY_IS_EQUAL(single_patch_row_major.data()[i],
                    tensor_row_major.data()[i]);
    VERIFY_IS_EQUAL(tensor_col_major.data()[i], tensor_row_major.data()[i]);
    VERIFY_IS_EQUAL(single_patch_col_major.data()[i],
                    single_patch_row_major.data()[i]);
  }

  // Entire image patch: ColMajor
  patchColMajorTensorRange={{sizeDim1, sizeDim2, sizeDim3, sizeDim2*sizeDim3}};
  Tensor<DataType, 4, DataLayout,IndexType> entire_image_patch_col_major(patchColMajorTensorRange);
  patchTensorBuffSize =entire_image_patch_col_major.size()*sizeof(DataType);
  DataType* gpu_data_entire_image_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_entire_image_patch_col_major(gpu_data_entire_image_patch_col_major, patchColMajorTensorRange);
  gpu_entire_image_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(3, 5);
  sycl_device.memcpyDeviceToHost(entire_image_patch_col_major.data(), gpu_data_entire_image_patch_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(entire_image_patch_col_major.dimension(0), 2);
  VERIFY_IS_EQUAL(entire_image_patch_col_major.dimension(1), 3);
  VERIFY_IS_EQUAL(entire_image_patch_col_major.dimension(2), 5);
  VERIFY_IS_EQUAL(entire_image_patch_col_major.dimension(3), 3*5);

  // Entire image patch: RowMajor
patchRowMajorTensorRange={{sizeDim2*sizeDim3, sizeDim3, sizeDim2, sizeDim1}};
Tensor<DataType, 4, RowMajor,IndexType> entire_image_patch_row_major(patchRowMajorTensorRange);
patchTensorBuffSize =entire_image_patch_row_major.size()*sizeof(DataType);
DataType* gpu_data_entire_image_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
TensorMap<Tensor<DataType, 4, RowMajor,IndexType>> gpu_entire_image_patch_row_major(gpu_data_entire_image_patch_row_major, patchRowMajorTensorRange);
gpu_entire_image_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(3, 5);
sycl_device.memcpyDeviceToHost(entire_image_patch_row_major.data(), gpu_data_entire_image_patch_row_major, patchTensorBuffSize);
  VERIFY_IS_EQUAL(entire_image_patch_row_major.dimension(0), 3*5);
  VERIFY_IS_EQUAL(entire_image_patch_row_major.dimension(1), 5);
  VERIFY_IS_EQUAL(entire_image_patch_row_major.dimension(2), 3);
  VERIFY_IS_EQUAL(entire_image_patch_row_major.dimension(3), 2);

  for (IndexType i = 0; i < 3; ++i) {
    for (IndexType j = 0; j < 5; ++j) {
      IndexType patchId = i+3*j;
      for (IndexType r = 0; r < 3; ++r) {
        for (IndexType c = 0; c < 5; ++c) {
          for (IndexType d = 0; d < 2; ++d) {
            DataType expected_col_major = 0.0f;
            DataType expected_row_major = 0.0f;
            if (r-1+i >= 0 && c-2+j >= 0 && r-1+i < 3 && c-2+j < 5) {
              expected_col_major = tensor_col_major(d, r-1+i, c-2+j);
              expected_row_major = tensor_row_major(c-2+j, r-1+i, d);
            }
            // ColMajor
            if (entire_image_patch_col_major(d, r, c, patchId) != expected_col_major) {
              std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << std::endl;
            }
            VERIFY_IS_EQUAL(entire_image_patch_col_major(d, r, c, patchId), expected_col_major);
            // RowMajor
            if (entire_image_patch_row_major(patchId, c, r, d) !=
                expected_row_major) {
              std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << std::endl;
            }
            VERIFY_IS_EQUAL(entire_image_patch_row_major(patchId, c, r, d),
                            expected_row_major);
            // Check that ColMajor and RowMajor agree.
            VERIFY_IS_EQUAL(expected_col_major, expected_row_major);
          }
        }
      }
    }
  }

  // 2D patch: ColMajor
  patchColMajorTensorRange={{sizeDim1, 2, 2, sizeDim2*sizeDim3}};
  Tensor<DataType, 4, DataLayout,IndexType> twod_patch_col_major(patchColMajorTensorRange);
  patchTensorBuffSize =twod_patch_col_major.size()*sizeof(DataType);
  DataType* gpu_data_twod_patch_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_twod_patch_col_major(gpu_data_twod_patch_col_major, patchColMajorTensorRange);
  gpu_twod_patch_col_major.device(sycl_device)=gpu_col_major.extract_image_patches(2, 2);
  sycl_device.memcpyDeviceToHost(twod_patch_col_major.data(), gpu_data_twod_patch_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(twod_patch_col_major.dimension(0), 2);
  VERIFY_IS_EQUAL(twod_patch_col_major.dimension(1), 2);
  VERIFY_IS_EQUAL(twod_patch_col_major.dimension(2), 2);
  VERIFY_IS_EQUAL(twod_patch_col_major.dimension(3), 3*5);

  // 2D patch: RowMajor
  patchRowMajorTensorRange={{sizeDim2*sizeDim3, 2, 2, sizeDim1}};
  Tensor<DataType, 4, RowMajor,IndexType> twod_patch_row_major(patchRowMajorTensorRange);
  patchTensorBuffSize =twod_patch_row_major.size()*sizeof(DataType);
  DataType* gpu_data_twod_patch_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 4, RowMajor,IndexType>> gpu_twod_patch_row_major(gpu_data_twod_patch_row_major, patchRowMajorTensorRange);
  gpu_twod_patch_row_major.device(sycl_device)=gpu_row_major.extract_image_patches(2, 2);
  sycl_device.memcpyDeviceToHost(twod_patch_row_major.data(), gpu_data_twod_patch_row_major, patchTensorBuffSize);
  VERIFY_IS_EQUAL(twod_patch_row_major.dimension(0), 3*5);
  VERIFY_IS_EQUAL(twod_patch_row_major.dimension(1), 2);
  VERIFY_IS_EQUAL(twod_patch_row_major.dimension(2), 2);
  VERIFY_IS_EQUAL(twod_patch_row_major.dimension(3), 2);

  // Based on the calculation described in TensorTraits.h, padding happens to be 0.
  IndexType row_padding = 0;
  IndexType col_padding = 0;
  IndexType stride = 1;

  for (IndexType i = 0; i < 3; ++i) {
    for (IndexType j = 0; j < 5; ++j) {
      IndexType patchId = i+3*j;
      for (IndexType r = 0; r < 2; ++r) {
        for (IndexType c = 0; c < 2; ++c) {
          for (IndexType d = 0; d < 2; ++d) {
            DataType expected_col_major = 0.0f;
            DataType expected_row_major = 0.0f;
            IndexType row_offset = r*stride + i - row_padding;
            IndexType col_offset = c*stride + j - col_padding;
            // ColMajor
            if (row_offset >= 0 && col_offset >= 0 && row_offset < tensor_col_major.dimension(1) && col_offset < tensor_col_major.dimension(2)) {
              expected_col_major = tensor_col_major(d, row_offset, col_offset);
            }
            if (twod_patch_col_major(d, r, c, patchId) != expected_col_major) {
              std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << std::endl;
            }
            VERIFY_IS_EQUAL(twod_patch_col_major(d, r, c, patchId), expected_col_major);
            // RowMajor
            if (row_offset >= 0 && col_offset >= 0 && row_offset < tensor_row_major.dimension(1) && col_offset < tensor_row_major.dimension(0)) {
              expected_row_major = tensor_row_major(col_offset, row_offset, d);
            }
            if (twod_patch_row_major(patchId, c, r, d) != expected_row_major) {
              std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << std::endl;
            }
            VERIFY_IS_EQUAL(twod_patch_row_major(patchId, c, r, d), expected_row_major);
            // Check that ColMajor and RowMajor agree.
            VERIFY_IS_EQUAL(expected_col_major, expected_row_major);
          }
        }
      }
    }
  }

  sycl_device.deallocate(gpu_data_col_major);
  sycl_device.deallocate(gpu_data_row_major);
  sycl_device.deallocate(gpu_data_single_patch_col_major);
  sycl_device.deallocate(gpu_data_single_patch_row_major);
  sycl_device.deallocate(gpu_data_entire_image_patch_col_major);
  sycl_device.deallocate(gpu_data_entire_image_patch_row_major);
  sycl_device.deallocate(gpu_data_twod_patch_col_major);
  sycl_device.deallocate(gpu_data_twod_patch_row_major);
}

template <typename DataType, typename IndexType>
static void test_imagenet_patches_sycl(const Eigen::SyclDevice& sycl_device)
{
  // Test the code on typical configurations used by the 'imagenet' benchmarks at
  // https://github.com/soumith/convnet-benchmarks
  // ColMajor
  IndexType sizeDim1 = 3;
  IndexType sizeDim2 = 128;
  IndexType sizeDim3 = 128;
  IndexType sizeDim4 = 16;
  array<IndexType, 4> tensorColMajorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  Tensor<DataType, 4, DataLayout,IndexType> l_in_col_major(tensorColMajorRange);
  l_in_col_major.setRandom();

  DataType* gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>> gpu_l_in_col_major(gpu_data_l_in_col_major, tensorColMajorRange);

  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));

  array<IndexType, 5> patchTensorRange={{sizeDim1, 11, 11, sizeDim2*sizeDim3, sizeDim4}};
  Tensor<DataType, 5, DataLayout,IndexType> l_out_col_major(patchTensorRange);
  size_t patchTensorBuffSize =l_out_col_major.size()*sizeof(DataType);
  DataType* gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_l_out_col_major(gpu_data_l_out_col_major, patchTensorRange);
  gpu_l_out_col_major.device(sycl_device)=gpu_l_in_col_major.extract_image_patches(11, 11);
  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(l_out_col_major.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(1), 11);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(2), 11);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(3), sizeDim2*sizeDim3);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(4), sizeDim4);

  // RowMajor
  patchTensorRange={{sizeDim4, sizeDim2*sizeDim3, 11, 11, sizeDim1}};
  Tensor<DataType, 5, RowMajor,IndexType> l_out_row_major(patchTensorRange);
  patchTensorBuffSize =l_out_row_major.size()*sizeof(DataType);
  DataType* gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>> gpu_l_out_row_major(gpu_data_l_out_row_major, patchTensorRange);
  gpu_l_out_row_major.device(sycl_device)=gpu_l_in_col_major.swap_layout().extract_image_patches(11, 11);
  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(l_out_row_major.dimension(0), sizeDim4);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(1), sizeDim2*sizeDim3);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(2), 11);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(3), 11);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(4), sizeDim1);

  for (IndexType b = 0; b < 16; ++b) {
    for (IndexType i = 0; i < 128; ++i) {
      for (IndexType j = 0; j < 128; ++j) {
        IndexType patchId = i+128*j;
        for (IndexType c = 0; c < 11; ++c) {
          for (IndexType r = 0; r < 11; ++r) {
            for (IndexType d = 0; d < 3; ++d) {
              DataType expected = 0.0f;
              if (r-5+i >= 0 && c-5+j >= 0 && r-5+i < 128 && c-5+j < 128) {
                expected = l_in_col_major(d, r-5+i, c-5+j, b);
              }
              // ColMajor
              if (l_out_col_major(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(l_out_col_major(d, r, c, patchId, b), expected);
              // RowMajor
              if (l_out_row_major(b, patchId, c, r, d) !=
                  expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j
                     << " r=" << r << " c=" << c << " d=" << d << " b=" << b
                     << std::endl;
              }
              VERIFY_IS_EQUAL(l_out_row_major(b, patchId, c, r, d),
                              expected);
            }
          }
        }
      }
    }
  }

  // ColMajor
  sycl_device.deallocate(gpu_data_l_in_col_major);
  sycl_device.deallocate(gpu_data_l_out_col_major);
  sizeDim1 = 16;
  sizeDim2 = 64;
  sizeDim3 = 64;
  sizeDim4 = 32;
  tensorColMajorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  l_in_col_major.resize(tensorColMajorRange);
  l_in_col_major.setRandom();
  gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>>gpu_l_in_col_major_resize1(gpu_data_l_in_col_major, tensorColMajorRange);

  patchTensorRange={{sizeDim1, 9, 9, sizeDim2*sizeDim3, sizeDim4}};
  l_out_col_major.resize(patchTensorRange);
  patchTensorBuffSize =l_out_col_major.size()*sizeof(DataType);
  gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>>gpu_l_out_col_major_resize1(gpu_data_l_out_col_major, patchTensorRange);
  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
  gpu_l_out_col_major_resize1.device(sycl_device)=gpu_l_in_col_major_resize1.extract_image_patches(9, 9);
  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(0), 16);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(1), 9);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(2), 9);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(3), 64*64);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(4), 32);

// RowMajor
  sycl_device.deallocate(gpu_data_l_out_row_major);
  patchTensorRange={{sizeDim4, sizeDim2*sizeDim3, 9, 9 ,sizeDim1}};
  l_out_row_major.resize(patchTensorRange);
  patchTensorBuffSize =l_out_row_major.size()*sizeof(DataType);
  gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>>gpu_l_out_row_major_resize1(gpu_data_l_out_row_major, patchTensorRange);
  gpu_l_out_row_major_resize1.device(sycl_device)=gpu_l_in_col_major_resize1.swap_layout().extract_image_patches(9, 9);
  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(l_out_row_major.dimension(0), 32);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(1), 64*64);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(2), 9);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(3), 9);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(4), 16);

  for (IndexType b = 0; b < 32; ++b) {
    for (IndexType i = 0; i < 64; ++i) {
      for (IndexType j = 0; j < 64; ++j) {
        IndexType patchId = i+64*j;
        for (IndexType c = 0; c < 9; ++c) {
          for (IndexType r = 0; r < 9; ++r) {
            for (IndexType d = 0; d < 16; ++d) {
              DataType expected = 0.0f;
              if (r-4+i >= 0 && c-4+j >= 0 && r-4+i < 64 && c-4+j < 64) {
                expected = l_in_col_major(d, r-4+i, c-4+j, b);
              }
              // ColMajor
              if (l_out_col_major(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(l_out_col_major(d, r, c, patchId, b), expected);
              // RowMajor
              if (l_out_row_major(b, patchId, c, r, d) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(l_out_row_major(b, patchId, c, r, d), expected);
            }
          }
        }
      }
    }
  }

  // ColMajor

  sycl_device.deallocate(gpu_data_l_in_col_major);
  sycl_device.deallocate(gpu_data_l_out_col_major);
  sizeDim1 = 32;
  sizeDim2 = 16;
  sizeDim3 = 16;
  sizeDim4 = 32;
  tensorColMajorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  l_in_col_major.resize(tensorColMajorRange);
  l_in_col_major.setRandom();
  gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>>gpu_l_in_col_major_resize2(gpu_data_l_in_col_major, tensorColMajorRange);

  patchTensorRange={{sizeDim1, 7, 7, sizeDim2*sizeDim3, sizeDim4}};
  l_out_col_major.resize(patchTensorRange);
  patchTensorBuffSize =l_out_col_major.size()*sizeof(DataType);
  gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>>gpu_l_out_col_major_resize2(gpu_data_l_out_col_major, patchTensorRange);
  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
  gpu_l_out_col_major_resize2.device(sycl_device)=gpu_l_in_col_major_resize2.extract_image_patches(7, 7);
  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(l_out_col_major.dimension(0), 32);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(1), 7);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(2), 7);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(3), 16*16);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(4), 32);

  // RowMajor
  sycl_device.deallocate(gpu_data_l_out_row_major);
  patchTensorRange={{sizeDim4, sizeDim2*sizeDim3, 7, 7 ,sizeDim1}};
  l_out_row_major.resize(patchTensorRange);
  patchTensorBuffSize =l_out_row_major.size()*sizeof(DataType);
  gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>>gpu_l_out_row_major_resize2(gpu_data_l_out_row_major, patchTensorRange);
  gpu_l_out_row_major_resize2.device(sycl_device)=gpu_l_in_col_major_resize2.swap_layout().extract_image_patches(7, 7);
  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(l_out_row_major.dimension(0), 32);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(1), 16*16);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(2), 7);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(3), 7);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(4), 32);

  for (IndexType b = 0; b < 32; ++b) {
    for (IndexType i = 0; i < 16; ++i) {
      for (IndexType j = 0; j < 16; ++j) {
        IndexType patchId = i+16*j;
        for (IndexType c = 0; c < 7; ++c) {
          for (IndexType r = 0; r < 7; ++r) {
            for (IndexType d = 0; d < 32; ++d) {
              DataType expected = 0.0f;
              if (r-3+i >= 0 && c-3+j >= 0 && r-3+i < 16 && c-3+j < 16) {
                expected = l_in_col_major(d, r-3+i, c-3+j, b);
              }
              // ColMajor
              if (l_out_col_major(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(l_out_col_major(d, r, c, patchId, b), expected);
              // RowMajor
              if (l_out_row_major(b, patchId, c, r, d) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(l_out_row_major(b, patchId, c, r, d), expected);
            }
          }
        }
      }
    }
  }

  // ColMajor
  sycl_device.deallocate(gpu_data_l_in_col_major);
  sycl_device.deallocate(gpu_data_l_out_col_major);
  sizeDim1 = 64;
  sizeDim2 = 13;
  sizeDim3 = 13;
  sizeDim4 = 32;
  tensorColMajorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  l_in_col_major.resize(tensorColMajorRange);
  l_in_col_major.setRandom();
  gpu_data_l_in_col_major  = static_cast<DataType*>(sycl_device.allocate(l_in_col_major.size()*sizeof(DataType)));
  TensorMap<Tensor<DataType, 4, ColMajor, IndexType>>gpu_l_in_col_major_resize3(gpu_data_l_in_col_major, tensorColMajorRange);

  patchTensorRange={{sizeDim1, 3, 3, sizeDim2*sizeDim3, sizeDim4}};
  l_out_col_major.resize(patchTensorRange);
  patchTensorBuffSize =l_out_col_major.size()*sizeof(DataType);
  gpu_data_l_out_col_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>>gpu_l_out_col_major_resize3(gpu_data_l_out_col_major, patchTensorRange);
  sycl_device.memcpyHostToDevice(gpu_data_l_in_col_major, l_in_col_major.data(),(l_in_col_major.size())*sizeof(DataType));
  gpu_l_out_col_major_resize3.device(sycl_device)=gpu_l_in_col_major_resize3.extract_image_patches(3, 3);
  sycl_device.memcpyDeviceToHost(l_out_col_major.data(), gpu_data_l_out_col_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(l_out_col_major.dimension(0), 64);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(1), 3);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(2), 3);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(3), 13*13);
  VERIFY_IS_EQUAL(l_out_col_major.dimension(4), 32);

  // RowMajor
  sycl_device.deallocate(gpu_data_l_out_row_major);
  patchTensorRange={{sizeDim4, sizeDim2*sizeDim3, 3, 3 ,sizeDim1}};
  l_out_row_major.resize(patchTensorRange);
  patchTensorBuffSize =l_out_row_major.size()*sizeof(DataType);
  gpu_data_l_out_row_major  = static_cast<DataType*>(sycl_device.allocate(patchTensorBuffSize));
  TensorMap<Tensor<DataType, 5, RowMajor,IndexType>>gpu_l_out_row_major_resize3(gpu_data_l_out_row_major, patchTensorRange);
  gpu_l_out_row_major_resize3.device(sycl_device)=gpu_l_in_col_major_resize3.swap_layout().extract_image_patches(3, 3);
  sycl_device.memcpyDeviceToHost(l_out_row_major.data(), gpu_data_l_out_row_major, patchTensorBuffSize);

  VERIFY_IS_EQUAL(l_out_row_major.dimension(0), 32);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(1), 13*13);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(2), 3);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(3), 3);
  VERIFY_IS_EQUAL(l_out_row_major.dimension(4), 64);

  for (IndexType b = 0; b < 32; ++b) {
    for (IndexType i = 0; i < 13; ++i) {
      for (IndexType j = 0; j < 13; ++j) {
        IndexType patchId = i+13*j;
        for (IndexType c = 0; c < 3; ++c) {
          for (IndexType r = 0; r < 3; ++r) {
            for (IndexType d = 0; d < 64; ++d) {
              DataType expected = 0.0f;
              if (r-1+i >= 0 && c-1+j >= 0 && r-1+i < 13 && c-1+j < 13) {
                expected = l_in_col_major(d, r-1+i, c-1+j, b);
              }
              // ColMajor
              if (l_out_col_major(d, r, c, patchId, b) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(l_out_col_major(d, r, c, patchId, b), expected);
              // RowMajor
              if (l_out_row_major(b, patchId, c, r, d) != expected) {
                std::cout << "Mismatch detected at index i=" << i << " j=" << j << " r=" << r << " c=" << c << " d=" << d << " b=" << b << std::endl;
              }
              VERIFY_IS_EQUAL(l_out_row_major(b, patchId, c, r, d), expected);
            }
          }
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data_l_in_col_major);
  sycl_device.deallocate(gpu_data_l_out_col_major);
  sycl_device.deallocate(gpu_data_l_out_row_major);
}


template<typename DataType, typename dev_Selector> void sycl_tensor_image_patch_test_per_device(dev_Selector s){
QueueInterface queueInterface(s);
auto sycl_device = Eigen::SyclDevice(&queueInterface);
test_simple_image_patch_sycl<DataType, int64_t>(sycl_device);
test_patch_padding_valid_sycl<DataType, int64_t>(sycl_device);
test_patch_padding_valid_same_value_sycl<DataType, int64_t>(sycl_device);
test_patch_padding_same_sycl<DataType, int64_t>(sycl_device);
test_patch_no_extra_dim_sycl<DataType, int64_t>(sycl_device);
test_imagenet_patches_sycl<DataType, int64_t>(sycl_device);
}
EIGEN_DECLARE_TEST(cxx11_tensor_image_patch_sycl)
{
for (const auto& device :Eigen::get_sycl_supported_devices()) {
  CALL_SUBTEST(sycl_tensor_image_patch_test_per_device<float>(device));
}
}
