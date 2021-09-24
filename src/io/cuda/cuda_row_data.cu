/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_row_data.hpp>
#include <LightGBM/cuda/cuda_algorithms.hpp>

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

void CUDARowData::LaunchCopyDenseSubrowKernel(const CUDARowData* full_set) {
  const int num_column = feature_partition_column_index_offsets_.back();
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
}

template <typename ROW_PTR_TYPE>
__global__ void CalcTotalNumberOfElementsKernel(
  const data_size_t num_used_indices,
  const data_size_t* cuda_used_indices,
  const ROW_PTR_TYPE* cuda_row_ptr,
  const int num_feature_partitions,
  const data_size_t num_data,
  uint64_t* block_sum_buffer) {
  __shared__ uint64_t shared_mem_buffer[32];
  const data_size_t local_data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const int partition_index = static_cast<int>(blockIdx.y);
  const ROW_PTR_TYPE* partition_row_ptr = cuda_row_ptr + partition_index * (num_data + 1);
  uint64_t num_elements_in_row = 0;
  if (local_data_index < num_used_indices) {
    const data_size_t global_data_index = cuda_used_indices[local_data_index];
    const data_size_t row_start = partition_row_ptr[global_data_index];
    const data_size_t row_end = partition_row_ptr[global_data_index + 1];
    num_elements_in_row = static_cast<uint64_t>(row_end - row_start);
  }
  const uint64_t num_elements_in_block = ShuffleReduceSum<uint64_t>(num_elements_in_row, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    printf("blockIdx.x = %d, partition_index = %d, num_elements_in_block = %d\n", blockIdx.x, partition_index, num_elements_in_block);
    block_sum_buffer[partition_index * gridDim.x + blockIdx.x] = num_elements_in_block;
  }
}

__global__ void ReduceBlockSumKernel(
  const uint64_t* block_sum_buffer,
  const int num_blocks,
  const int num_feature_partitions,
  uint64_t* cuda_partition_ptr_buffer) {
  __shared__ uint64_t shared_mem_buffer[32];
  uint64_t thread_sum = 0;
  const int partition_index = static_cast<int>(blockIdx.x);
  const uint64_t* block_sum_buffer_ptr = block_sum_buffer + partition_index * num_blocks;
  for (data_size_t block_index = static_cast<data_size_t>(threadIdx.x); block_index < num_blocks; block_index += static_cast<data_size_t>(blockDim.x)) {
    thread_sum += block_sum_buffer_ptr[block_index];
  }
  if (threadIdx.x == 0) {
    printf("thread_sum = %d\n", thread_sum);
  }
  const uint64_t num_total_elements = ShuffleReduceSum<uint64_t>(thread_sum, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    printf("partition_index = %d, num_total_elements = %d\n", partition_index, num_total_elements);
    cuda_partition_ptr_buffer[partition_index + 1] = num_total_elements;
    if (blockIdx.x == 0) {
      cuda_partition_ptr_buffer[0] = 0;
    }
  }
}

__global__ void ComputePartitionPtr(
  uint64_t* cuda_partition_ptr_buffer,
  const int num_feature_partitions) {
  __shared__ uint64_t shared_mem_buffer[32];
  const int num_partitions_per_thread = (num_feature_partitions + blockDim.x - 1) / (blockDim.x - 1);
  int start_partition = threadIdx.x == 0 ? 0 : num_partitions_per_thread * static_cast<int>(threadIdx.x - 1);
  int end_partition = threadIdx.x == 0 ? 0 : min(start_partition + num_partitions_per_thread, num_feature_partitions + 1);
  uint64_t thread_sum = 0;
  for (int partition_index = start_partition; partition_index < end_partition; ++partition_index) {
    thread_sum += cuda_partition_ptr_buffer[partition_index];
  }
  const uint64_t thread_base = ShufflePrefixSum<uint64_t>(thread_sum, shared_mem_buffer);
  start_partition = threadIdx.x == blockDim.x - 1 ? 0 : num_partitions_per_thread * static_cast<int>(threadIdx.x);
  end_partition = threadIdx.x == blockDim.x - 1 ? 0 : min(start_partition + num_partitions_per_thread, num_feature_partitions + 1);
  for (int partition_index = start_partition + 1; partition_index < end_partition; ++partition_index) {
    cuda_partition_ptr_buffer[partition_index] += cuda_partition_ptr_buffer[partition_index - 1];
  }
  for (int partition_index = start_partition; partition_index < end_partition; ++partition_index) {
    cuda_partition_ptr_buffer[partition_index] += thread_base;
  }
  if (threadIdx.x == blockDim.x - 1) {
    cuda_partition_ptr_buffer[num_feature_partitions] = thread_sum;
  }
}

uint64_t CUDARowData::LaunchCalcTotalNumberOfElementsKernel(const CUDARowData* full_set) {
  const int num_blocks = (num_used_indices_ + COPY_SUBROW_BLOCK_SIZE_ROW_DATA - 1) / COPY_SUBROW_BLOCK_SIZE_ROW_DATA;
  SetCUDAMemoryOuter<uint64_t>(cuda_block_sum_buffer_, 0, static_cast<size_t>(num_blocks * num_feature_partitions_) + 1, __FILE__, __LINE__);
  if (full_set->row_ptr_bit_type_ == 16) {
    CalcTotalNumberOfElementsKernel<uint16_t><<<dim3(num_blocks, num_feature_partitions_), COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
      num_used_indices_,
      cuda_used_indices_,
      full_set->cuda_row_ptr_uint16_t_,
      num_feature_partitions_,
      num_data_,
      cuda_block_sum_buffer_);
  } else if (full_set->row_ptr_bit_type_ == 32) {
    CalcTotalNumberOfElementsKernel<uint32_t><<<dim3(num_blocks, num_feature_partitions_), COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
      num_used_indices_,
      cuda_used_indices_,
      full_set->cuda_row_ptr_uint32_t_,
      num_feature_partitions_,
      num_data_,
      cuda_block_sum_buffer_);
  } else if (full_set->row_ptr_bit_type_ == 64) {
    CalcTotalNumberOfElementsKernel<uint64_t><<<dim3(num_blocks, num_feature_partitions_), COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
      num_used_indices_,
      cuda_used_indices_,
      full_set->cuda_row_ptr_uint64_t_,
      num_feature_partitions_,
      num_data_,
      cuda_block_sum_buffer_);
  }
  Log::Warning("num_feature_partitions_ = %d", num_feature_partitions_);
  ReduceBlockSumKernel<<<num_feature_partitions_, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
    cuda_block_sum_buffer_, num_blocks, num_feature_partitions_, cuda_partition_ptr_buffer_);
  ComputePartitionPtr<<<1, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(cuda_partition_ptr_buffer_, num_feature_partitions_);
  uint64_t num_total_elements = 0;
  CopyFromCUDADeviceToHostOuter<uint64_t>(&num_total_elements, cuda_partition_ptr_buffer_ + num_feature_partitions_, 1, __FILE__, __LINE__);
  Log::Warning("num_used_indices = %d, num_blocks = %d", num_used_indices_, num_blocks);
  return num_total_elements;
}

template <typename ROW_PTR_TYPE>
__global__ void CopyPartitionPtrKernel(
  const uint64_t* cuda_partition_ptr_buffer,
  const int num_feature_partitions,
  ROW_PTR_TYPE* cuda_partition_ptr) {
  for (int partition_index = static_cast<int>(threadIdx.x); partition_index < num_feature_partitions + 1; partition_index += static_cast<int>(blockDim.x)) {
    cuda_partition_ptr[partition_index] = static_cast<ROW_PTR_TYPE>(cuda_partition_ptr_buffer[partition_index]);
  }
}

template <typename IN_ROW_PTR_TYPE, typename OUT_ROW_PTR_TYPE>
__global__ void CopySparseSubrowRowPtrKernel(
  const IN_ROW_PTR_TYPE* cuda_row_ptr,
  const data_size_t num_used_indices,
  const data_size_t* cuda_used_indices,
  OUT_ROW_PTR_TYPE* out_cuda_row_ptr) {
  const data_size_t local_data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (local_data_index < num_used_indices) {
    const data_size_t global_data_index = cuda_used_indices[local_data_index];
    const IN_ROW_PTR_TYPE row_start = cuda_row_ptr[global_data_index];
    const IN_ROW_PTR_TYPE row_end = cuda_row_ptr[global_data_index + 1];
    const OUT_ROW_PTR_TYPE num_elements_in_row = static_cast<OUT_ROW_PTR_TYPE>(row_end - row_start);
    out_cuda_row_ptr[local_data_index + 1] = num_elements_in_row;
  }
}

template <typename BIN_TYPE, typename IN_ROW_PTR_TYPE, typename OUT_ROW_PTR_TYPE>
__global__ void CopySparseSubrowDataKernel(
  const BIN_TYPE* in_cuda_data,
  const IN_ROW_PTR_TYPE* in_cuda_row_ptr, 
  const OUT_ROW_PTR_TYPE* out_cuda_row_ptr,
  const data_size_t num_used_indices,
  const data_size_t* cuda_used_indices,
  BIN_TYPE* out_cuda_data) {
  const data_size_t local_data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (local_data_index < num_used_indices) {
    const data_size_t global_data_index = cuda_used_indices[local_data_index];
    const IN_ROW_PTR_TYPE in_row_start = in_cuda_row_ptr[global_data_index];
    const IN_ROW_PTR_TYPE in_row_end = in_cuda_row_ptr[global_data_index + 1];
    const IN_ROW_PTR_TYPE in_num_elements_in_row = in_row_end - in_row_start;
    const OUT_ROW_PTR_TYPE out_row_start = out_cuda_row_ptr[local_data_index];
    const OUT_ROW_PTR_TYPE out_row_end = out_cuda_row_ptr[local_data_index + 1];
    const OUT_ROW_PTR_TYPE out_num_elements_in_row = out_row_end - out_row_start;
    if (in_num_elements_in_row != out_num_elements_in_row) {
      printf("error !!!!!, in_num_elements_in_row = %d, out_num_elements_in_row = %d\n", in_num_elements_in_row, out_num_elements_in_row);
    }
    if (out_row_end > in_row_end || out_row_start > in_row_start) {
      printf("error !!!!!, out_row_end = %d, in_row_end = %d, out_row_start = %d, in_row_start = %d\n", out_row_end, in_row_end, out_row_start, in_row_start);
    }
    const BIN_TYPE* in_cuda_data_ptr = in_cuda_data + in_row_start; 
    BIN_TYPE* out_cuda_data_ptr = out_cuda_data + out_row_start;
    for (IN_ROW_PTR_TYPE element_index = 0; element_index < in_num_elements_in_row; ++element_index) {
      out_cuda_data_ptr[element_index] = in_cuda_data_ptr[element_index];
    }
  }
}

template <typename OUT_ROW_PTR_TYPE, typename BIN_TYPE>
void CUDARowData::LaunchCopySparseSubrowKernelInner1(
  const CUDARowData* full_set,
  const BIN_TYPE* in_cuda_data,
  const OUT_ROW_PTR_TYPE* out_cuda_row_ptr,
  BIN_TYPE* out_cuda_data) {
  CHECK_GE(full_set->row_ptr_bit_type_, row_ptr_bit_type_);
  const int num_blocks = (num_used_indices_ + COPY_SUBROW_BLOCK_SIZE_ROW_DATA - 1) / COPY_SUBROW_BLOCK_SIZE_ROW_DATA;
  if (full_set->row_ptr_bit_type_ == 16) {
    CopySparseSubrowDataKernel<BIN_TYPE, uint16_t, OUT_ROW_PTR_TYPE><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
      in_cuda_data, full_set->cuda_row_ptr_uint16_t_, out_cuda_row_ptr, num_used_indices_, cuda_used_indices_, out_cuda_data);
  } else if (full_set->row_ptr_bit_type_ == 32) {
    CopySparseSubrowDataKernel<BIN_TYPE, uint32_t, OUT_ROW_PTR_TYPE><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
      in_cuda_data, full_set->cuda_row_ptr_uint32_t_, out_cuda_row_ptr, num_used_indices_, cuda_used_indices_, out_cuda_data);
  } else if (full_set->row_ptr_bit_type_ == 64) {
    CopySparseSubrowDataKernel<BIN_TYPE, uint64_t, OUT_ROW_PTR_TYPE><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
      in_cuda_data, full_set->cuda_row_ptr_uint64_t_, out_cuda_row_ptr, num_used_indices_, cuda_used_indices_, out_cuda_data);
  }
}

template <typename OUT_ROW_PTR_TYPE>
void CUDARowData::LaunchCopySparseSubrowKernelInner0(
  const CUDARowData* full_set,
  OUT_ROW_PTR_TYPE* out_cuda_row_ptr) {
  ShufflePrefixSumGlobal<OUT_ROW_PTR_TYPE>(
    out_cuda_row_ptr,
    static_cast<size_t>(num_used_indices_) + 1,
    reinterpret_cast<OUT_ROW_PTR_TYPE*>(cuda_block_sum_buffer_));
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  if (bit_type_ == 8) {
    LaunchCopySparseSubrowKernelInner1<OUT_ROW_PTR_TYPE, uint8_t>(full_set, full_set->cuda_data_uint8_t_, out_cuda_row_ptr, cuda_data_uint8_t_);
  } else if (bit_type_ == 16) {
    LaunchCopySparseSubrowKernelInner1<OUT_ROW_PTR_TYPE, uint16_t>(full_set, full_set->cuda_data_uint16_t_, out_cuda_row_ptr, cuda_data_uint16_t_);
  } else if (bit_type_ == 32) {
    LaunchCopySparseSubrowKernelInner1<OUT_ROW_PTR_TYPE, uint32_t>(full_set, full_set->cuda_data_uint32_t_, out_cuda_row_ptr, cuda_data_uint32_t_);
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

void CUDARowData::LaunchCopySparseSubrowKernel(const CUDARowData* full_set) {
  if (row_ptr_bit_type_ == 16) {
    CopyPartitionPtrKernel<<<1, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(cuda_partition_ptr_buffer_, num_feature_partitions_, cuda_partition_ptr_uint16_t_);
  } else if (row_ptr_bit_type_ == 32) {
    CopyPartitionPtrKernel<<<1, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(cuda_partition_ptr_buffer_, num_feature_partitions_, cuda_partition_ptr_uint32_t_);
  } else if (row_ptr_bit_type_ == 64) {
    CopyPartitionPtrKernel<<<1, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(cuda_partition_ptr_buffer_, num_feature_partitions_, cuda_partition_ptr_uint64_t_);
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  const int num_blocks = (num_used_indices_ + COPY_SUBROW_BLOCK_SIZE_ROW_DATA - 1) / COPY_SUBROW_BLOCK_SIZE_ROW_DATA;
  if (full_set->row_ptr_bit_type_ == 16) {
    CHECK_EQ(row_ptr_bit_type_, 16);
    CopySparseSubrowRowPtrKernel<uint16_t, uint16_t><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
      full_set->cuda_row_ptr_uint16_t_, num_used_indices_, cuda_used_indices_, cuda_row_ptr_uint16_t_);
  } else if (full_set->row_ptr_bit_type_ == 32) {
    CHECK(row_ptr_bit_type_ == 16 || row_ptr_bit_type_ == 32);
    if (row_ptr_bit_type_ == 16) {
      CopySparseSubrowRowPtrKernel<uint32_t, uint16_t><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
        full_set->cuda_row_ptr_uint32_t_, num_used_indices_, cuda_used_indices_, cuda_row_ptr_uint16_t_);
    } else if (row_ptr_bit_type_ == 32) {
      CopySparseSubrowRowPtrKernel<uint32_t, uint32_t><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
        full_set->cuda_row_ptr_uint32_t_, num_used_indices_, cuda_used_indices_, cuda_row_ptr_uint32_t_);
    }
  } else if (full_set->row_ptr_bit_type_ == 64) {
    if (row_ptr_bit_type_ == 16) {
      CopySparseSubrowRowPtrKernel<uint64_t, uint16_t><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
        full_set->cuda_row_ptr_uint64_t_, num_used_indices_, cuda_used_indices_, cuda_row_ptr_uint16_t_);
    } else if (row_ptr_bit_type_ == 32) {
      CopySparseSubrowRowPtrKernel<uint64_t, uint32_t><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
        full_set->cuda_row_ptr_uint64_t_, num_used_indices_, cuda_used_indices_, cuda_row_ptr_uint32_t_);
    } else if (row_ptr_bit_type_ == 64) {
      CopySparseSubrowRowPtrKernel<uint64_t, uint64_t><<<num_blocks, COPY_SUBROW_BLOCK_SIZE_ROW_DATA>>>(
        full_set->cuda_row_ptr_uint64_t_, num_used_indices_, cuda_used_indices_, cuda_row_ptr_uint64_t_);
    }
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  if (row_ptr_bit_type_ == 16) {
    LaunchCopySparseSubrowKernelInner0<uint16_t>(full_set, cuda_row_ptr_uint16_t_);
  } else if (row_ptr_bit_type_ == 32) {
    LaunchCopySparseSubrowKernelInner0<uint32_t>(full_set, cuda_row_ptr_uint32_t_);
  } else if (row_ptr_bit_type_ == 64) {
    LaunchCopySparseSubrowKernelInner0<uint64_t>(full_set, cuda_row_ptr_uint64_t_);
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

}  // namespace LightGBM
