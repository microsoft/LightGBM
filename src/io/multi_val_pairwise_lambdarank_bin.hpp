/*!
 * Copyright (c) 2024 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
#define LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_

#include "multi_val_dense_bin.hpp"

namespace LightGBM {

template <typename BIN_TYPE, template<typename> class MULTI_VAL_BIN_TYPE>
class MultiValPairwiseLambdarankBin : public MULTI_VAL_BIN_TYPE<BIN_TYPE> {
 public:
  MultiValPairwiseLambdarankBin(data_size_t num_data, int num_bin, int num_feature, const std::vector<uint32_t>& offsets): MULTI_VAL_BIN_TYPE<BIN_TYPE>(num_data, num_bin, num_feature, offsets) {
    this->num_bin_ = num_bin;
    Log::Warning("num_bin = %d", num_bin);
  }
 protected:
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_global_index_map_;
};


template <typename BIN_TYPE>
class MultiValDensePairwiseLambdarankBin: public MultiValPairwiseLambdarankBin<BIN_TYPE, MultiValDenseBin> {
 public:
  MultiValDensePairwiseLambdarankBin(data_size_t num_data, int num_bin, int num_feature,
    const std::vector<uint32_t>& offsets, const std::pair<data_size_t, data_size_t>* paired_ranking_item_global_index_map): MultiValPairwiseLambdarankBin<BIN_TYPE, LightGBM::MultiValDenseBin>(num_data, num_bin, num_feature, offsets) {
    this->paired_ranking_item_global_index_map_ = paired_ranking_item_global_index_map;
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* gradients,
                          const score_t* hessians, hist_t* out) const override {
    ConstructHistogramInner<true, true, false>(data_indices, start, end,
                                               gradients, hessians, out);
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
                          const score_t* gradients, const score_t* hessians,
                          hist_t* out) const override {
    ConstructHistogramInner<false, false, false>(
        nullptr, start, end, gradients, hessians, out);
  }

  void ConstructHistogramOrdered(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const score_t* gradients,
                                 const score_t* hessians,
                                 hist_t* out) const override {
    ConstructHistogramInner<true, true, true>(data_indices, start, end,
                                              gradients, hessians, out);
  }

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED>
  void ConstructHistogramInner(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* gradients, const score_t* hessians, hist_t* out) const {
    data_size_t i = start;
    hist_t* grad = out;
    hist_t* hess = out + 1;
    for (; i < end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const data_size_t first_idx = this->paired_ranking_item_global_index_map_[idx].first;
      const data_size_t second_idx = this->paired_ranking_item_global_index_map_[idx].second;
      const auto first_j_start = this->RowPtr(first_idx);
      const BIN_TYPE* first_data_ptr = this->data_.data() + first_j_start;
      const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
      const score_t hessian = ORDERED ? hessians[i] : hessians[idx];
      for (int j = 0; j < this->num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(first_data_ptr[j]);
        // if (bin != 0) {
        //   Log::Warning("first bin = %d, num_feature_ = %d", bin, this->num_feature_);
        // }
        // if (j == 0) {
        //   Log::Warning("group index = %d bin = %d gradient = %f hessian = %f", j, bin, gradient, hessian);
        // }

        const auto ti = (bin + this->offsets_[j]) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }

      const auto second_j_start = this->RowPtr(second_idx);
      const BIN_TYPE* second_data_ptr = this->data_.data() + second_j_start;
      const auto base_offset = this->offsets_.back();
      for (int j = 0; j < this->num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(second_data_ptr[j]);
        // if (bin != 0) {
        //   Log::Warning("second bin = %d, num_feature_ = %d", bin, this->num_feature_);
        // }
        const auto ti = (bin + this->offsets_[j] + base_offset) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
  }
};

}  // namespace LightGBM

#endif  // LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
