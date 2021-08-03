/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_column_data.hpp>

namespace LightGBM {

CUDAColumnData::CUDAColumnData(const data_size_t num_data) {
  num_threads_ = OMP_NUM_THREADS();
  num_data_ = num_data;
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
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int column_index = 0; column_index < num_columns_; ++column_index) {
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
  }
  feature_to_column_ = feature_to_column;
  InitCUDAMemoryFromHostMemoryOuter<void*>(&cuda_data_by_column_,
                                           data_by_column_.data(),
                                           data_by_column_.size(),
                                           __FILE__,
                                           __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint8_t>(&cuda_column_bit_type_,
                                       column_bit_type_.data(),
                                       column_bit_type_.size(),
                                       __FILE__,
                                       __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_max_bin_,
                                         feature_max_bin.data(),
                                         feature_max_bin.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_min_bin_,
                                         feature_min_bin.data(),
                                         feature_min_bin.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_offset_,
                                         feature_offset.data(),
                                         feature_offset.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_most_freq_bin_,
                                         feature_most_freq_bin.data(),
                                         feature_most_freq_bin.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_feature_default_bin_,
                                         feature_default_bin.data(),
                                         feature_default_bin.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint8_t>(&cuda_feature_missing_is_zero_,
                                         feature_missing_is_zero.data(),
                                         feature_missing_is_zero.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint8_t>(&cuda_feature_missing_is_na_,
                                        feature_missing_is_na.data(),
                                        feature_missing_is_na.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint8_t>(&cuda_feature_mfb_is_zero_,
                                        feature_mfb_is_zero.data(),
                                        feature_mfb_is_zero.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint8_t>(&cuda_feature_mfb_is_na_,
                                        feature_mfb_is_na.data(),
                                        feature_mfb_is_na.size(),
                                        __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_feature_to_column_,
                                    feature_to_column_.data(),
                                    feature_to_column_.size(),
                                    __FILE__,
                                    __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

}  // namespace LightGBM
