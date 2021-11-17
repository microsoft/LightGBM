/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_column_data.hpp>

namespace LightGBM {

CUDAColumnData::CUDAColumnData(const data_size_t num_data, const int gpu_device_id) {
  num_threads_ = OMP_NUM_THREADS();
  num_data_ = num_data;
  if (gpu_device_id >= 0) {
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  } else {
    SetCUDADevice(0, __FILE__, __LINE__);
  }
  cuda_used_indices_ = nullptr;
  cuda_data_by_column_ = nullptr;
  cuda_column_bit_type_ = nullptr;
  cuda_feature_min_bin_ = nullptr;
  cuda_feature_max_bin_ = nullptr;
  cuda_feature_offset_ = nullptr;
  cuda_feature_most_freq_bin_ = nullptr;
  cuda_feature_default_bin_ = nullptr;
  cuda_feature_missing_is_zero_ = nullptr;
  cuda_feature_missing_is_na_ = nullptr;
  cuda_feature_mfb_is_zero_ = nullptr;
  cuda_feature_mfb_is_na_ = nullptr;
  cuda_feature_to_column_ = nullptr;
  data_by_column_.clear();
}

CUDAColumnData::~CUDAColumnData() {
  DeallocateCUDAMemory<data_size_t>(&cuda_used_indices_, __FILE__, __LINE__);
  DeallocateCUDAMemory<void*>(&cuda_data_by_column_, __FILE__, __LINE__);
  for (size_t i = 0; i < data_by_column_.size(); ++i) {
    DeallocateCUDAMemory<void>(&data_by_column_[i], __FILE__, __LINE__);
  }
  DeallocateCUDAMemory<uint8_t>(&cuda_column_bit_type_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_min_bin_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_max_bin_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_offset_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_most_freq_bin_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_feature_default_bin_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_feature_missing_is_zero_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_feature_missing_is_na_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_feature_mfb_is_zero_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&cuda_feature_mfb_is_na_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&cuda_feature_to_column_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_used_indices_, __FILE__, __LINE__);
}

template <bool IS_SPARSE, bool IS_4BIT, typename BIN_TYPE>
void CUDAColumnData::InitOneColumnData(const void* in_column_data, BinIterator* bin_iterator, void** out_column_data_pointer) {
  BIN_TYPE* cuda_column_data = nullptr;
  if (!IS_SPARSE) {
    if (IS_4BIT) {
      std::vector<BIN_TYPE> expanded_column_data(num_data_, 0);
      const BIN_TYPE* in_column_data_reintrepreted = reinterpret_cast<const BIN_TYPE*>(in_column_data);
      for (data_size_t i = 0; i < num_data_; ++i) {
        expanded_column_data[i] = static_cast<BIN_TYPE>((in_column_data_reintrepreted[i >> 1] >> ((i & 1) << 2)) & 0xf);
      }
      InitCUDAMemoryFromHostMemory<BIN_TYPE>(&cuda_column_data,
                                                  expanded_column_data.data(),
                                                  static_cast<size_t>(num_data_),
                                                  __FILE__,
                                                  __LINE__);
    } else {
      InitCUDAMemoryFromHostMemory<BIN_TYPE>(&cuda_column_data,
                                                  reinterpret_cast<const BIN_TYPE*>(in_column_data),
                                                  static_cast<size_t>(num_data_),
                                                  __FILE__,
                                                  __LINE__);
    }
  } else {
    // need to iterate bin iterator
    std::vector<BIN_TYPE> expanded_column_data(num_data_, 0);
    for (data_size_t i = 0; i < num_data_; ++i) {
      expanded_column_data[i] = static_cast<BIN_TYPE>(bin_iterator->RawGet(i));
    }
    InitCUDAMemoryFromHostMemory<BIN_TYPE>(&cuda_column_data,
                                                expanded_column_data.data(),
                                                static_cast<size_t>(num_data_),
                                                __FILE__,
                                                __LINE__);
  }
  *out_column_data_pointer = reinterpret_cast<void*>(cuda_column_data);
}

void CUDAColumnData::Init(const int num_columns,
                          const std::vector<const void*>& column_data,
                          const std::vector<BinIterator*>& column_bin_iterator,
                          const std::vector<uint8_t>& column_bit_type,
                          const std::vector<uint32_t>& feature_max_bin,
                          const std::vector<uint32_t>& feature_min_bin,
                          const std::vector<uint32_t>& feature_offset,
                          const std::vector<uint32_t>& feature_most_freq_bin,
                          const std::vector<uint32_t>& feature_default_bin,
                          const std::vector<uint8_t>& feature_missing_is_zero,
                          const std::vector<uint8_t>& feature_missing_is_na,
                          const std::vector<uint8_t>& feature_mfb_is_zero,
                          const std::vector<uint8_t>& feature_mfb_is_na,
                          const std::vector<int>& feature_to_column) {
  num_columns_ = num_columns;
  column_bit_type_ = column_bit_type;
  feature_max_bin_ = feature_max_bin;
  feature_min_bin_ = feature_min_bin;
  feature_offset_ = feature_offset;
  feature_most_freq_bin_ = feature_most_freq_bin;
  feature_default_bin_ = feature_default_bin;
  feature_missing_is_zero_ = feature_missing_is_zero;
  feature_missing_is_na_ = feature_missing_is_na;
  feature_mfb_is_zero_ = feature_mfb_is_zero;
  feature_mfb_is_na_ = feature_mfb_is_na;
  data_by_column_.resize(num_columns_, nullptr);
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int column_index = 0; column_index < num_columns_; ++column_index) {
    OMP_LOOP_EX_BEGIN();
    const int8_t bit_type = column_bit_type[column_index];
    if (column_data[column_index] != nullptr) {
      // is dense column
      if (bit_type == 4) {
        column_bit_type_[column_index] = 8;
        InitOneColumnData<false, true, uint8_t>(column_data[column_index], nullptr, &data_by_column_[column_index]);
      } else if (bit_type == 8) {
        InitOneColumnData<false, false, uint8_t>(column_data[column_index], nullptr, &data_by_column_[column_index]);
      } else if (bit_type == 16) {
        InitOneColumnData<false, false, uint16_t>(column_data[column_index], nullptr, &data_by_column_[column_index]);
      } else if (bit_type == 32) {
        InitOneColumnData<false, false, uint32_t>(column_data[column_index], nullptr, &data_by_column_[column_index]);
      } else {
        Log::Fatal("Unknow column bit type %d", bit_type);
      }
    } else {
      // is sparse column
      if (bit_type == 8) {
        InitOneColumnData<true, false, uint8_t>(nullptr, column_bin_iterator[column_index], &data_by_column_[column_index]);
      } else if (bit_type == 16) {
        InitOneColumnData<true, false, uint16_t>(nullptr, column_bin_iterator[column_index], &data_by_column_[column_index]);
      } else if (bit_type == 32) {
        InitOneColumnData<true, false, uint32_t>(nullptr, column_bin_iterator[column_index], &data_by_column_[column_index]);
      } else {
        Log::Fatal("Unknow column bit type %d", bit_type);
      }
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  feature_to_column_ = feature_to_column;
  InitCUDAMemoryFromHostMemory<void*>(&cuda_data_by_column_,
                                           data_by_column_.data(),
                                           data_by_column_.size(),
                                           __FILE__,
                                           __LINE__);
  InitColumnMetaInfo();
}

void CUDAColumnData::CopySubrow(
  const CUDAColumnData* full_set,
  const data_size_t* used_indices,
  const data_size_t num_used_indices) {
  num_threads_ = full_set->num_threads_;
  num_columns_ = full_set->num_columns_;
  column_bit_type_ = full_set->column_bit_type_;
  feature_min_bin_ = full_set->feature_min_bin_;
  feature_max_bin_ = full_set->feature_max_bin_;
  feature_offset_ = full_set->feature_offset_;
  feature_most_freq_bin_ = full_set->feature_most_freq_bin_;
  feature_default_bin_ = full_set->feature_default_bin_;
  feature_missing_is_zero_ = full_set->feature_missing_is_zero_;
  feature_missing_is_na_ = full_set->feature_missing_is_na_;
  feature_mfb_is_zero_ = full_set->feature_mfb_is_zero_;
  feature_mfb_is_na_ = full_set->feature_mfb_is_na_;
  feature_to_column_ = full_set->feature_to_column_;
  if (cuda_used_indices_ == nullptr) {
    // initialize the subset cuda column data
    const size_t num_used_indices_size = static_cast<size_t>(num_used_indices);
    AllocateCUDAMemory<data_size_t>(&cuda_used_indices_, num_used_indices_size, __FILE__, __LINE__);
    data_by_column_.resize(num_columns_, nullptr);
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int column_index = 0; column_index < num_columns_; ++column_index) {
      OMP_LOOP_EX_BEGIN();
      const uint8_t bit_type = column_bit_type_[column_index];
      if (bit_type == 8) {
        uint8_t* column_data = nullptr;
        AllocateCUDAMemory<uint8_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
        data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
      } else if (bit_type == 16) {
        uint16_t* column_data = nullptr;
        AllocateCUDAMemory<uint16_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
        data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
      } else if (bit_type == 32) {
        uint32_t* column_data = nullptr;
        AllocateCUDAMemory<uint32_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
        data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    InitCUDAMemoryFromHostMemory<void*>(&cuda_data_by_column_, data_by_column_.data(), data_by_column_.size(), __FILE__, __LINE__);
    InitColumnMetaInfo();
    cur_subset_buffer_size_ = num_used_indices;
  } else {
    if (num_used_indices > cur_subset_buffer_size_) {
      ResizeWhenCopySubrow(num_used_indices);
      cur_subset_buffer_size_ = num_used_indices;
    }
  }
  CopyFromHostToCUDADevice<data_size_t>(cuda_used_indices_, used_indices, static_cast<size_t>(num_used_indices), __FILE__, __LINE__);
  num_used_indices_ = num_used_indices;
  LaunchCopySubrowKernel(full_set->cuda_data_by_column());
}

void CUDAColumnData::ResizeWhenCopySubrow(const data_size_t num_used_indices) {
  const size_t num_used_indices_size = static_cast<size_t>(num_used_indices);
  DeallocateCUDAMemory<data_size_t>(&cuda_used_indices_, __FILE__, __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_used_indices_, num_used_indices_size, __FILE__, __LINE__);
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int column_index = 0; column_index < num_columns_; ++column_index) {
    OMP_LOOP_EX_BEGIN();
    const uint8_t bit_type = column_bit_type_[column_index];
    if (bit_type == 8) {
      uint8_t* column_data = reinterpret_cast<uint8_t*>(data_by_column_[column_index]);
      DeallocateCUDAMemory<uint8_t>(&column_data, __FILE__, __LINE__);
      AllocateCUDAMemory<uint8_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
      data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
    } else if (bit_type == 16) {
      uint16_t* column_data = reinterpret_cast<uint16_t*>(data_by_column_[column_index]);
      DeallocateCUDAMemory<uint16_t>(&column_data, __FILE__, __LINE__);
      AllocateCUDAMemory<uint16_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
      data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
    } else if (bit_type == 32) {
      uint32_t* column_data = reinterpret_cast<uint32_t*>(data_by_column_[column_index]);
      DeallocateCUDAMemory<uint32_t>(&column_data, __FILE__, __LINE__);
      AllocateCUDAMemory<uint32_t>(&column_data, num_used_indices_size, __FILE__, __LINE__);
      data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  DeallocateCUDAMemory<void*>(&cuda_data_by_column_, __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemory<void*>(&cuda_data_by_column_, data_by_column_.data(), data_by_column_.size(), __FILE__, __LINE__);
}

void CUDAColumnData::InitColumnMetaInfo() {
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_column_bit_type_,
                                       column_bit_type_.data(),
                                       column_bit_type_.size(),
                                       __FILE__,
                                       __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_max_bin_,
                                         feature_max_bin_.data(),
                                         feature_max_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_min_bin_,
                                         feature_min_bin_.data(),
                                         feature_min_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_offset_,
                                         feature_offset_.data(),
                                         feature_offset_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_most_freq_bin_,
                                         feature_most_freq_bin_.data(),
                                         feature_most_freq_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_default_bin_,
                                         feature_default_bin_.data(),
                                         feature_default_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_missing_is_zero_,
                                         feature_missing_is_zero_.data(),
                                         feature_missing_is_zero_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_missing_is_na_,
                                        feature_missing_is_na_.data(),
                                        feature_missing_is_na_.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_mfb_is_zero_,
                                        feature_mfb_is_zero_.data(),
                                        feature_mfb_is_zero_.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_mfb_is_na_,
                                        feature_mfb_is_na_.data(),
                                        feature_mfb_is_na_.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemory<int>(&cuda_feature_to_column_,
                                    feature_to_column_.data(),
                                    feature_to_column_.size(),
                                    __FILE__,
                                    __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
