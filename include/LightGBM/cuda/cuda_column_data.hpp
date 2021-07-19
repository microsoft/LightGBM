/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_CUDA_COLUMN_DATA_HPP_
#define LIGHTGBM_CUDA_COLUMN_DATA_HPP_

#include <LightGBM/config.h>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/bin.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <vector>

namespace LightGBM {

class CUDAColumnData {
 public:
  CUDAColumnData(const data_size_t num_data);

  ~CUDAColumnData();

  void Init(const int num_columns,
            const std::vector<const void*>& column_data,
            const std::vector<BinIterator*>& column_bin_iterator,
            const std::vector<int8_t>& column_bit_type,
            const std::vector<uint32_t>& feature_max_bin,
            const std::vector<uint32_t>& feature_min_bin,
            const std::vector<uint32_t>& feature_offset,
            const std::vector<uint32_t>& feature_most_freq_bin,
            const std::vector<uint32_t>& feature_default_bin,
            const std::vector<int>& feature_to_column);

  void* const* cuda_data_by_column() const { return cuda_data_by_column_; }

  const uint32_t* cuda_feature_min_bin() const { return cuda_feature_min_bin_; }

  const uint32_t* cuda_feature_max_bin() const { return cuda_feature_max_bin_; }

  const uint32_t* cuda_feature_offset() const { return cuda_feature_offset_; }

  const uint32_t* cuda_feature_most_freq_bin() const { return cuda_feature_most_freq_bin_; }

  const uint32_t* cuda_feature_default_bin() const { return cuda_feature_default_bin_; }

  const int* cuda_feature_to_column() const { return cuda_feature_to_column_; }

  const int8_t* cuda_column_bit_type() const { return cuda_column_bit_type_; }

 private:
  template <bool IS_SPARSE, bool IS_4BIT, typename BIN_TYPE>
  void InitOneColumnData(const void* in_column_data, BinIterator* bin_iterator, void** out_column_data_pointer);

  int num_threads_;
  data_size_t num_data_;
  int num_columns_;
  std::vector<int8_t> column_bit_type_;
  void** cuda_data_by_column_;
  std::vector<int> feature_to_column_;
  std::vector<void*> data_by_column_;

  int8_t* cuda_column_bit_type_;
  uint32_t* cuda_feature_min_bin_;
  uint32_t* cuda_feature_max_bin_;
  uint32_t* cuda_feature_offset_;
  uint32_t* cuda_feature_most_freq_bin_;
  uint32_t* cuda_feature_default_bin_;
  int* cuda_feature_to_column_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_CUDA_COLUMN_DATA_HPP_
