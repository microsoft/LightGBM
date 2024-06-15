/*!
 * Copyright (c) 2024 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
#define LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_

#include "multi_val_dense_bin.hpp"

template <typename MULTIVAL_BIN_TYPE>
class MultiValPairwiseLambdarankBin : public MULTIVAL_BIN_TYPE {
 protected:
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
};


template <typename VAL_T>
class MultiValDensePairwiseLambdarankBin: public MultiValPairwiseLambdarankBin<MultiValDenseBin<VAL_T>> {
 public:
  MultiValDensePairwiseLambdarankBin(data_size_t num_data, int num_bin, int num_feature,
    const std::vector<uint32_t>& offsets, const data_size_t* paired_ranking_item_global_index_map): MultiValDenseBin<VAL_T>(num_data, num_bin, num_feature, offsets) {
    paired_ranking_item_global_index_map_ = paired_ranking_item_global_index_map;
  }

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED>
  void ConstructHistogramInner(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* gradients, const score_t* hessians, hist_t* out) const {
    data_size_t i = start;
    hist_t* grad = out;
    hist_t* hess = out + 1;

    for (; i < end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const data_size_t first_idx = paired_ranking_item_index_map_[idx].first;
      const data_size_t second_idx = paired_ranking_item_index_map_[idx].second;
      const auto first_j_start = RowPtr(first_idx);
      const VAL_T* first_data_ptr = data_.data() + first_j_start;
      const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
      const score_t hessian = ORDERED ? hessians[i] : hessians[idx];
      for (int j = 0; j < num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(first_data_ptr[j]);
        const auto ti = (bin + offsets_[j]) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }

      const auto second_j_start = RowPtr(second_idx);
      const VAL_T* second_data_ptr = data_.data() + second_j_start;
      const auto base_offset = offsets_.back();
      for (int j = 0; j < num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(second_data_ptr[j]);
        const auto ti = (bin + offsets_[j] + base_offset) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
  }
};


#endif  // LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
