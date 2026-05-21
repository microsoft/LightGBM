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
  MultiValPairwiseLambdarankBin(data_size_t num_data, int num_bin, int num_feature,
                                const std::vector<uint32_t>& offsets)
      : MULTI_VAL_BIN_TYPE<BIN_TYPE>(num_data, num_bin, num_feature, offsets) {
    this->num_bin_ = num_bin;
    Log::Warning("num_bin = %d", num_bin);
  }

 protected:
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_global_index_map_;
};

template <typename BIN_TYPE>
class MultiValDensePairwiseLambdarankBin
    : public MultiValPairwiseLambdarankBin<BIN_TYPE, MultiValDenseBin> {
 public:
  MultiValDensePairwiseLambdarankBin(
      data_size_t num_data, int num_bin, int num_feature,
      const std::vector<uint32_t>& offsets,
      const std::pair<data_size_t, data_size_t>* paired_ranking_item_global_index_map,
      const std::vector<const BinMapper*> diff_feature_bin_mappers,
      const std::vector<const BinMapper*> original_feature_bin_mappers,
      const bool use_pairwise_bin_lookup,
      const std::vector<std::vector<float>>* raw_data,
      const std::vector<uint32_t>& all_offsets,
      const std::vector<int>& diff_feature_to_original_feature_slot,
      const std::vector<int>& diff_feature_to_raw_feature_index)
      : MultiValPairwiseLambdarankBin<BIN_TYPE, LightGBM::MultiValDenseBin>(
            num_data, num_bin, num_feature, offsets),
        diff_feature_to_original_feature_slot_(diff_feature_to_original_feature_slot),
        pointwise_histogram_buffers_(OMP_NUM_THREADS()),
        use_fast_pairwise_bin_lookup_(use_pairwise_bin_lookup) {
    this->paired_ranking_item_global_index_map_ =
        paired_ranking_item_global_index_map;
    CHECK_EQ(diff_feature_bin_mappers.size(), diff_feature_to_original_feature_slot.size());
    CHECK_EQ(diff_feature_bin_mappers.size(), diff_feature_to_raw_feature_index.size());
    num_diff_features_ = static_cast<int>(diff_feature_bin_mappers.size());
    diff_feature_bin_mappers_.reserve(num_diff_features_);
    diff_feature_hist_offsets_.reserve(num_diff_features_);
    diff_feature_bin_value_offsets_.reserve(num_diff_features_);
    original_feature_bin_value_offsets_.reserve(num_diff_features_);
    diff_feature_raw_data_ptrs_.reserve(num_diff_features_);
    original_feature_most_freq_bins_.reserve(num_diff_features_);
    active_diff_feature_slots_.reserve(num_diff_features_);
    active_diff_feature_hist_offsets_.reserve(num_diff_features_);
    active_diff_feature_bin_value_offsets_.reserve(num_diff_features_);
    active_diff_feature_bin_mappers_.reserve(num_diff_features_);
    for (int j = 0; j < num_diff_features_; ++j) {
      diff_feature_bin_mappers_.emplace_back(diff_feature_bin_mappers[j]);
      diff_feature_hist_offsets_.push_back(all_offsets[2 * this->num_feature_ + j]);
      diff_feature_bin_value_offsets_.push_back(
          1u - static_cast<uint32_t>(diff_feature_bin_mappers_[j]->GetMostFreqBin() == 0));
      original_feature_bin_value_offsets_.push_back(
          1u - static_cast<uint32_t>(original_feature_bin_mappers[j]->GetMostFreqBin() == 0));
      original_feature_most_freq_bins_.push_back(original_feature_bin_mappers[j]->GetMostFreqBin());
      if (raw_data != nullptr && !raw_data->empty()) {
        const int raw_feature_index = diff_feature_to_raw_feature_index[j];
        CHECK_GE(raw_feature_index, 0);
        CHECK_LT(raw_feature_index, static_cast<int>(raw_data->size()));
        const float* raw_data_ptr = (*raw_data)[raw_feature_index].data();
        diff_feature_raw_data_ptrs_.push_back(raw_data_ptr);
        if (raw_data_ptr != nullptr) {
          active_diff_feature_slots_.push_back(j);
          active_diff_feature_hist_offsets_.push_back(diff_feature_hist_offsets_[j]);
          active_diff_feature_bin_value_offsets_.push_back(diff_feature_bin_value_offsets_[j]);
          active_diff_feature_bin_mappers_.push_back(diff_feature_bin_mappers_[j].get());
          use_fast_pairwise_bin_lookup_ &= diff_feature_bin_mappers_[j]->HasPairwiseBinRanges();
        }
      } else {
        diff_feature_raw_data_ptrs_.push_back(nullptr);
      }
    }
    InitDiffFeatureRawValueBuffer(num_data);
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
    ConstructHistogramInner<false, false, false>(nullptr, start, end, gradients,
                                                 hessians, out);
  }

  void ConstructHistogramOrdered(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const score_t* gradients,
                                 const score_t* hessians,
                                 hist_t* out) const override {
    ConstructHistogramInner<true, true, true>(data_indices, start, end,
                                              gradients, hessians, out);
  }

  template <bool USE_INDICES, bool USE_PREFETCH, bool ORDERED>
  void ConstructHistogramInner(const data_size_t* data_indices,
                               data_size_t start, data_size_t end,
                               const score_t* gradients,
                               const score_t* hessians, hist_t* out) const {
    data_size_t i = start;
    hist_t* grad = out;
    hist_t* hess = out + 1;
    const auto base_offset = this->offsets_.back();
    const data_size_t num_original_data = this->num_data_;
    const int tid = omp_get_thread_num();
    CHECK_LT(tid, static_cast<int>(pointwise_histogram_buffers_.size()));
    PointwiseHistogramBuffer& pointwise_buffer = pointwise_histogram_buffers_[tid];
    pointwise_buffer.Reset(num_original_data, static_cast<size_t>(end - start));
    std::vector<hist_t>& first_gradients = pointwise_buffer.first_gradients;
    std::vector<hist_t>& first_hessians = pointwise_buffer.first_hessians;
    std::vector<hist_t>& second_gradients = pointwise_buffer.second_gradients;
    std::vector<hist_t>& second_hessians = pointwise_buffer.second_hessians;
    std::vector<data_size_t>& first_touched = pointwise_buffer.first_touched;
    std::vector<data_size_t>& second_touched = pointwise_buffer.second_touched;
    const uint32_t stamp = pointwise_buffer.stamp;

    if (!active_diff_feature_slots_.empty() && use_fast_pairwise_bin_lookup_) {
      const float* diff_feature_raw_values = diff_feature_raw_values_.data();
      for (; i < end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        if (USE_PREFETCH) {
          const data_size_t pf_offset = 32 / sizeof(BIN_TYPE);
          const data_size_t pf_i = i + pf_offset;
          if (pf_i < end) {
            const auto pf_idx = USE_INDICES ? data_indices[pf_i] : pf_i;
            const data_size_t pf_first_idx =
                this->paired_ranking_item_global_index_map_[pf_idx].first;
            const data_size_t pf_second_idx =
                this->paired_ranking_item_global_index_map_[pf_idx].second;
            PREFETCH_T0(this->data_.data() + this->RowPtr(pf_first_idx));
            PREFETCH_T0(this->data_.data() + this->RowPtr(pf_second_idx));
            PREFETCH_T0(diff_feature_raw_values +
                        static_cast<size_t>(pf_first_idx) * num_active_diff_features_);
            PREFETCH_T0(diff_feature_raw_values +
                        static_cast<size_t>(pf_second_idx) * num_active_diff_features_);
            if (!ORDERED) {
              PREFETCH_T0(gradients + pf_idx);
              PREFETCH_T0(hessians + pf_idx);
            }
          }
        }
        const data_size_t first_idx =
            this->paired_ranking_item_global_index_map_[idx].first;
        const data_size_t second_idx =
            this->paired_ranking_item_global_index_map_[idx].second;
        const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
        const score_t hessian = ORDERED ? hessians[i] : hessians[idx];

        pointwise_buffer.AddFirst(first_idx, gradient, hessian, stamp);
        pointwise_buffer.AddSecond(second_idx, gradient, hessian, stamp);

        const float* first_row_diff_feature_raw_values =
            diff_feature_raw_values +
            static_cast<size_t>(first_idx) * num_active_diff_features_;
        const float* second_row_diff_feature_raw_values =
            diff_feature_raw_values +
            static_cast<size_t>(second_idx) * num_active_diff_features_;
        for (int active_j = 0; active_j < num_active_diff_features_; ++active_j) {
          // const int original_feature_slot = diff_feature_to_original_feature_slot_[j];
          // uint32_t first_bin = static_cast<uint32_t>(first_data_ptr[original_feature_slot]);
          // uint32_t second_bin = static_cast<uint32_t>(second_data_ptr[original_feature_slot]);
          const float diff_value =
              first_row_diff_feature_raw_values[active_j] -
              second_row_diff_feature_raw_values[active_j];

          const uint32_t diff_bin =
              static_cast<uint32_t>(diff_value > static_cast<float>(-kEpsilon)) +
              static_cast<uint32_t>(diff_value > static_cast<float>(kEpsilon));
          // if (first_bin >= original_feature_bin_value_offsets_[j]) {
          //   first_bin -= original_feature_bin_value_offsets_[j];
          // } else {
          //   CHECK_EQ(first_bin, 0);
          //   first_bin = original_feature_most_freq_bins_[j];
          // }
          // if (second_bin >= original_feature_bin_value_offsets_[j]) {
          //   second_bin -= original_feature_bin_value_offsets_[j];
          // } else {
          //   CHECK_EQ(second_bin, 0);
          //   second_bin = original_feature_most_freq_bins_[j];
          // }

          // const uint32_t diff_bin =
          //     diff_feature_bin_mappers_[j]->ValueToBinWithPairwiseRangeUnchecked(
          //         diff_value, first_bin, second_bin, original_feature_slot);
          // // The original row-wise bins already exclude feature-group offsets.
          // // Differential features still need the single-feature-group packing fix below.
          const uint32_t bin = diff_bin + active_diff_feature_bin_value_offsets_[active_j];
          const auto ti = (bin + active_diff_feature_hist_offsets_[active_j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    } else {
      const float* diff_feature_raw_values = diff_feature_raw_values_.data();
      for (; i < end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        if (USE_PREFETCH) {
          const data_size_t pf_offset = 32 / sizeof(BIN_TYPE);
          const data_size_t pf_i = i + pf_offset;
          if (pf_i < end) {
            const auto pf_idx = USE_INDICES ? data_indices[pf_i] : pf_i;
            const data_size_t pf_first_idx =
                this->paired_ranking_item_global_index_map_[pf_idx].first;
            const data_size_t pf_second_idx =
                this->paired_ranking_item_global_index_map_[pf_idx].second;
            PREFETCH_T0(this->data_.data() + this->RowPtr(pf_first_idx));
            PREFETCH_T0(this->data_.data() + this->RowPtr(pf_second_idx));
            if (!active_diff_feature_slots_.empty()) {
              PREFETCH_T0(diff_feature_raw_values +
                          static_cast<size_t>(pf_first_idx) * num_active_diff_features_);
              PREFETCH_T0(diff_feature_raw_values +
                          static_cast<size_t>(pf_second_idx) * num_active_diff_features_);
            }
            if (!ORDERED) {
              PREFETCH_T0(gradients + pf_idx);
              PREFETCH_T0(hessians + pf_idx);
            }
          }
        }
        const data_size_t first_idx =
            this->paired_ranking_item_global_index_map_[idx].first;
        const data_size_t second_idx =
            this->paired_ranking_item_global_index_map_[idx].second;
        const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
        const score_t hessian = ORDERED ? hessians[i] : hessians[idx];

        pointwise_buffer.AddFirst(first_idx, gradient, hessian, stamp);
        pointwise_buffer.AddSecond(second_idx, gradient, hessian, stamp);

        if (active_diff_feature_slots_.empty()) {
          continue;
        }
        const float* first_row_diff_feature_raw_values =
            diff_feature_raw_values +
            static_cast<size_t>(first_idx) * num_active_diff_features_;
        const float* second_row_diff_feature_raw_values =
            diff_feature_raw_values +
            static_cast<size_t>(second_idx) * num_active_diff_features_;
        for (int active_j = 0; active_j < num_active_diff_features_; ++active_j) {
          const float diff_value =
              first_row_diff_feature_raw_values[active_j] -
              second_row_diff_feature_raw_values[active_j];
          const uint32_t diff_bin = active_diff_feature_bin_mappers_[active_j]->ValueToBin(diff_value);
          // imitates push the bin into a single feature group (+ min_bin_ (which is always 1 for single feature groups) - offset)
          // and then extracted in a FeatureGroupIteartor (+ min_bin_ (which is 1 for FeatureGroupIterator) - offset (which is 1 for FeatureGroupIterator))
          // thus effectively should do only (+ 1 - offset)
          const uint32_t bin = diff_bin + active_diff_feature_bin_value_offsets_[active_j];
          const auto ti = (bin + active_diff_feature_hist_offsets_[active_j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    }
    for (const data_size_t row_idx : first_touched) {
      const BIN_TYPE* data_ptr = this->data_.data() + this->RowPtr(row_idx);
      const hist_t gradient = first_gradients[row_idx];
      const hist_t hessian = first_hessians[row_idx];
      for (int j = 0; j < this->num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + this->offsets_[j]) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
    for (const data_size_t row_idx : second_touched) {
      const BIN_TYPE* data_ptr = this->data_.data() + this->RowPtr(row_idx);
      const hist_t gradient = second_gradients[row_idx];
      const hist_t hessian = second_hessians[row_idx];
      for (int j = 0; j < this->num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + this->offsets_[j] + base_offset) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
  }

 private:
  struct PointwiseHistogramBuffer {
    std::vector<hist_t> first_gradients;
    std::vector<hist_t> first_hessians;
    std::vector<hist_t> second_gradients;
    std::vector<hist_t> second_hessians;
    std::vector<uint32_t> first_stamps;
    std::vector<uint32_t> second_stamps;
    std::vector<data_size_t> first_touched;
    std::vector<data_size_t> second_touched;
    uint32_t stamp = 0;

    void Reset(data_size_t num_data, size_t reserve_size) {
      const size_t size = static_cast<size_t>(num_data);
      if (first_gradients.size() != size) {
        first_gradients.assign(size, 0.0);
        first_hessians.assign(size, 0.0);
        second_gradients.assign(size, 0.0);
        second_hessians.assign(size, 0.0);
        first_stamps.assign(size, 0);
        second_stamps.assign(size, 0);
        stamp = 0;
      } else if (stamp == std::numeric_limits<uint32_t>::max()) {
        std::fill(first_stamps.begin(), first_stamps.end(), 0);
        std::fill(second_stamps.begin(), second_stamps.end(), 0);
        stamp = 0;
      }
      ++stamp;
      first_touched.clear();
      second_touched.clear();
      if (first_touched.capacity() < reserve_size) {
        first_touched.reserve(reserve_size);
      }
      if (second_touched.capacity() < reserve_size) {
        second_touched.reserve(reserve_size);
      }
    }

    void AddFirst(data_size_t idx, score_t gradient, score_t hessian, uint32_t current_stamp) {
      if (first_stamps[idx] != current_stamp) {
        first_stamps[idx] = current_stamp;
        first_gradients[idx] = 0.0;
        first_hessians[idx] = 0.0;
        first_touched.push_back(idx);
      }
      first_gradients[idx] += gradient;
      first_hessians[idx] += hessian;
    }

    void AddSecond(data_size_t idx, score_t gradient, score_t hessian, uint32_t current_stamp) {
      if (second_stamps[idx] != current_stamp) {
        second_stamps[idx] = current_stamp;
        second_gradients[idx] = 0.0;
        second_hessians[idx] = 0.0;
        second_touched.push_back(idx);
      }
      second_gradients[idx] += gradient;
      second_hessians[idx] += hessian;
    }
  };

  void InitDiffFeatureRawValueBuffer(data_size_t num_data) {
    num_active_diff_features_ = static_cast<int>(active_diff_feature_slots_.size());
    if (num_active_diff_features_ == 0) {
      return;
    }
    diff_feature_raw_values_.resize(
        static_cast<size_t>(num_data) * num_active_diff_features_);
    const int num_threads = OMP_NUM_THREADS();
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (data_size_t i = 0; i < num_data; ++i) {
      float* row_diff_feature_raw_values =
          diff_feature_raw_values_.data() +
          static_cast<size_t>(i) * num_active_diff_features_;
      for (int active_j = 0; active_j < num_active_diff_features_; ++active_j) {
        const int j = active_diff_feature_slots_[active_j];
        const float* feature_values = diff_feature_raw_data_ptrs_[j];
        row_diff_feature_raw_values[active_j] = feature_values[i];
      }
    }
  }

  std::vector<std::unique_ptr<const BinMapper>> diff_feature_bin_mappers_;
  std::vector<int> diff_feature_to_original_feature_slot_;
  std::vector<uint32_t> diff_feature_hist_offsets_;
  std::vector<uint32_t> diff_feature_bin_value_offsets_;
  std::vector<uint32_t> original_feature_bin_value_offsets_;
  std::vector<uint32_t> original_feature_most_freq_bins_;
  std::vector<const float*> diff_feature_raw_data_ptrs_;
  std::vector<float> diff_feature_raw_values_;
  std::vector<int> active_diff_feature_slots_;
  std::vector<uint32_t> active_diff_feature_hist_offsets_;
  std::vector<uint32_t> active_diff_feature_bin_value_offsets_;
  std::vector<const BinMapper*> active_diff_feature_bin_mappers_;
  mutable std::vector<PointwiseHistogramBuffer> pointwise_histogram_buffers_;
  int num_diff_features_;
  int num_active_diff_features_ = 0;
  bool use_fast_pairwise_bin_lookup_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
