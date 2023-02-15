/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#ifndef LIGHTGBM_CUDA_CUDA_COLUMN_DATA_HPP_
#define LIGHTGBM_CUDA_CUDA_COLUMN_DATA_HPP_

#include <LightGBM/config.h>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/bin.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <vector>

namespace LightGBM {

class CUDAColumnData {
 public:
  CUDAColumnData(const data_size_t num_data, const int gpu_device_id);

  ~CUDAColumnData();

  void Init(const int num_columns,
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
            const std::vector<int>& feature_to_column);

  const void* GetColumnData(const int column_index) const { return data_by_column_[column_index]; }

  void CopySubrow(const CUDAColumnData* full_set, const data_size_t* used_indices, const data_size_t num_used_indices);

  void* const* cuda_data_by_column() const { return cuda_data_by_column_; }

  uint32_t feature_min_bin(const int feature_index) const { return feature_min_bin_[feature_index]; }

  uint32_t feature_max_bin(const int feature_index) const { return feature_max_bin_[feature_index]; }

  uint32_t feature_offset(const int feature_index) const { return feature_offset_[feature_index]; }

  uint32_t feature_most_freq_bin(const int feature_index) const { return feature_most_freq_bin_[feature_index]; }

  uint32_t feature_default_bin(const int feature_index) const { return feature_default_bin_[feature_index]; }

  uint8_t feature_missing_is_zero(const int feature_index) const { return feature_missing_is_zero_[feature_index]; }

  uint8_t feature_missing_is_na(const int feature_index) const { return feature_missing_is_na_[feature_index]; }

  uint8_t feature_mfb_is_zero(const int feature_index) const { return feature_mfb_is_zero_[feature_index]; }

  uint8_t feature_mfb_is_na(const int feature_index) const { return feature_mfb_is_na_[feature_index]; }

  const uint32_t* cuda_feature_min_bin() const { return cuda_feature_min_bin_; }

  const uint32_t* cuda_feature_max_bin() const { return cuda_feature_max_bin_; }

  const uint32_t* cuda_feature_offset() const { return cuda_feature_offset_; }

  const uint32_t* cuda_feature_most_freq_bin() const { return cuda_feature_most_freq_bin_; }

  const uint32_t* cuda_feature_default_bin() const { return cuda_feature_default_bin_; }

  const uint8_t* cuda_feature_missing_is_zero() const { return cuda_feature_missing_is_zero_; }

  const uint8_t* cuda_feature_missing_is_na() const { return cuda_feature_missing_is_na_; }

  const uint8_t* cuda_feature_mfb_is_zero() const { return cuda_feature_mfb_is_zero_; }

  const uint8_t* cuda_feature_mfb_is_na() const { return cuda_feature_mfb_is_na_; }

  const int* cuda_feature_to_column() const { return cuda_feature_to_column_; }

  const uint8_t* cuda_column_bit_type() const { return cuda_column_bit_type_; }

  int feature_to_column(const int feature_index) const { return feature_to_column_[feature_index]; }

  uint8_t column_bit_type(const int column_index) const { return column_bit_type_[column_index]; }

 private:
  template <bool IS_SPARSE, bool IS_4BIT, typename BIN_TYPE>
  void InitOneColumnData(const void* in_column_data, BinIterator* bin_iterator, void** out_column_data_pointer);

  void LaunchCopySubrowKernel(void* const* in_cuda_data_by_column);

  void InitColumnMetaInfo();

  void ResizeWhenCopySubrow(const data_size_t num_used_indices);

  int num_threads_;
  data_size_t num_data_;
  int num_columns_;
  std::vector<uint8_t> column_bit_type_;
  std::vector<uint32_t> feature_min_bin_;
  std::vector<uint32_t> feature_max_bin_;
  std::vector<uint32_t> feature_offset_;
  std::vector<uint32_t> feature_most_freq_bin_;
  std::vector<uint32_t> feature_default_bin_;
  std::vector<uint8_t> feature_missing_is_zero_;
  std::vector<uint8_t> feature_missing_is_na_;
  std::vector<uint8_t> feature_mfb_is_zero_;
  std::vector<uint8_t> feature_mfb_is_na_;
  void** cuda_data_by_column_;
  std::vector<int> feature_to_column_;
  std::vector<void*> data_by_column_;

  uint8_t* cuda_column_bit_type_;
  uint32_t* cuda_feature_min_bin_;
  uint32_t* cuda_feature_max_bin_;
  uint32_t* cuda_feature_offset_;
  uint32_t* cuda_feature_most_freq_bin_;
  uint32_t* cuda_feature_default_bin_;
  uint8_t* cuda_feature_missing_is_zero_;
  uint8_t* cuda_feature_missing_is_na_;
  uint8_t* cuda_feature_mfb_is_zero_;
  uint8_t* cuda_feature_mfb_is_na_;
  int* cuda_feature_to_column_;

  // used when bagging with subset
  data_size_t* cuda_used_indices_;
  data_size_t num_used_indices_;
  data_size_t cur_subset_buffer_size_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_CUDA_CUDA_COLUMN_DATA_HPP_

#endif  // USE_CUDA
