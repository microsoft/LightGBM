/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_row_data.hpp>

#define COPY_SUBROW_BLOCK_SIZE_ROW_DATA (1024)

namespace LightGBM {

template <typename BIN_TYPE>
__global__ void CopySubrowDenseKernel(const BIN_TYPE* full_set_bin_data, const int num_column, const data_size_t num_used_indices,
  const data_size_t* used_indices, BIN_TYPE* bin_data) {
  const data_size_t local_data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (local_data_index < num_used_indices) {
    const data_size_t global_data_index = used_indices[local_data_index];
    const BIN_TYPE* src = full_set_bin_data + global_data_index * num_column;
    BIN_TYPE* dst = bin_data + local_data_index * num_column;
    for (int column_index = 0; column_index < num_column; ++column_index) {
      dst[column_index] = src[column_index];
    }
  }
}

void CUDARowData::LaunchCopySubrowKernel(const CUDARowData* full_set) {
  const int num_column = feature_partition_column_index_offsets_.back();
  if (!is_sparse_) {
    const int num_blocks = (num_used_indices_ + COPY_SUBROW_BLOCK_SIZE_ROW_DATA - 1) / COPY_SUBROW_BLOCK_SIZE_ROW_DATA;
    if (bit_type_ == 8) {
      const uint8_t* full_set_bin_data = full_set->cuda_data_uint8_t_;
      CopySubrowDenseKernel<uint8_t><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
        full_set_bin_data, num_column, num_used_indices_, cuda_used_indices_, cuda_data_uint8_t_);
    } else if (bit_type_ == 16) {
      const uint16_t* full_set_bin_data = full_set->cuda_data_uint16_t_;
      CopySubrowDenseKernel<uint16_t><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
        full_set_bin_data, num_column, num_used_indices_, cuda_used_indices_, cuda_data_uint16_t_);
    } else if (bit_type_ == 32) {
      const uint32_t* full_set_bin_data = full_set->cuda_data_uint32_t_;
      CopySubrowDenseKernel<uint32_t><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
        full_set_bin_data, num_column, num_used_indices_, cuda_used_indices_, cuda_data_uint32_t_);
    }
  } else {
    // TODO(shiyu1994): copy subrow for sparse data
  }
}

}  // namespace LightGBM
