/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_column_data.hpp>

namespace LightGBM {

CUDAColumnData::CUDAColumnData(const data_size_t num_data, const int gpu_device_id) {
  num_threads_ = OMP_NUM_THREADS();
  num_data_ = num_data;
  if (gpu_device_id >= 0) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_device_id));
  } else {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  }
  cuda_used_indices_ = nullptr;
  cuda_data_by_column_ = nullptr;
}

CUDAColumnData::~CUDAColumnData() {}

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
      InitCUDAMemoryFromHostMemoryOuter<BIN_TYPE>(&cuda_column_data,
                                                  expanded_column_data.data(),
                                                  static_cast<size_t>(num_data_),
                                                  __FILE__,
                                                  __LINE__);
    } else {
      InitCUDAMemoryFromHostMemoryOuter<BIN_TYPE>(&cuda_column_data,
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
    InitCUDAMemoryFromHostMemoryOuter<BIN_TYPE>(&cuda_column_data,
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
  InitCUDAMemoryFromHostMemoryOuter<void*>(&cuda_data_by_column_,
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
  if (cuda_used_indices_ == nullptr) {
    // initialize the subset cuda column data
    const size_t full_set_num_data = static_cast<size_t>(full_set->num_data_);
    AllocateCUDAMemoryOuter<data_size_t>(&cuda_used_indices_, full_set_num_data, __FILE__, __LINE__);
    CopyFromHostToCUDADeviceOuter<data_size_t>(cuda_used_indices_, used_indices, static_cast<size_t>(num_used_indices), __FILE__, __LINE__);
    data_by_column_.resize(num_columns_, nullptr);
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int column_index = 0; column_index < num_columns_; ++column_index) {
      const uint8_t bit_type = column_bit_type_[column_index];
      if (bit_type == 8) {
        uint8_t* column_data = nullptr;
        AllocateCUDAMemoryOuter<uint8_t>(&column_data, full_set_num_data, __FILE__, __LINE__);
        data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
      } else if (bit_type == 16) {
        uint16_t* column_data = nullptr;
        AllocateCUDAMemoryOuter<uint16_t>(&column_data, full_set_num_data, __FILE__, __LINE__);
        data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
      } else if (bit_type == 32) {
        uint32_t* column_data = nullptr;
        AllocateCUDAMemoryOuter<uint32_t>(&column_data, full_set_num_data, __FILE__, __LINE__);
        data_by_column_[column_index] = reinterpret_cast<void*>(column_data);
      }
    }
    InitCUDAMemoryFromHostMemoryOuter<void*>(&cuda_data_by_column_, data_by_column_.data(), data_by_column_.size(), __FILE__, __LINE__);
    InitColumnMetaInfo();
  }
  LaunchCopySubrowKernel(full_set->cuda_data_by_column(), num_used_indices);
}

void CUDAColumnData::InitColumnMetaInfo() {
  InitCUDAMemoryFromHostMemoryOuter<uint8_t>(&cuda_column_bit_type_,
                                       column_bit_type_.data(),
                                       column_bit_type_.size(),
                                       __FILE__,
                                       __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_max_bin_,
                                         feature_max_bin_.data(),
                                         feature_max_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_min_bin_,
                                         feature_min_bin_.data(),
                                         feature_min_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_offset_,
                                         feature_offset_.data(),
                                         feature_offset_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_most_freq_bin_,
                                         feature_most_freq_bin_.data(),
                                         feature_most_freq_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_default_bin_,
                                         feature_default_bin_.data(),
                                         feature_default_bin_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint8_t>(&cuda_feature_missing_is_zero_,
                                         feature_missing_is_zero_.data(),
                                         feature_missing_is_zero_.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint8_t>(&cuda_feature_missing_is_na_,
                                        feature_missing_is_na_.data(),
                                        feature_missing_is_na_.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint8_t>(&cuda_feature_mfb_is_zero_,
                                        feature_mfb_is_zero_.data(),
                                        feature_mfb_is_zero_.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint8_t>(&cuda_feature_mfb_is_na_,
                                        feature_mfb_is_na_.data(),
                                        feature_mfb_is_na_.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_feature_to_column_,
                                    feature_to_column_.data(),
                                    feature_to_column_.size(),
                                    __FILE__,
                                    __LINE__);
}

}  // namespace LightGBM
