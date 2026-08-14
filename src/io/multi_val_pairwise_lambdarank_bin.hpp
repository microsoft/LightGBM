/*!
 * Copyright (c) 2024 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
#define LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_

#include "multi_val_dense_bin.hpp"

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#include <immintrin.h>
#define LIGHTGBM_TARGET_AVX512 1
#else
#define LIGHTGBM_TARGET_AVX512 0
#endif

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
      data_size_t num_data, data_size_t num_pairwise_data, int num_bin, int num_feature,
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
        num_pairwise_data_(num_pairwise_data),
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
    active_diff_feature_hist_ti_.reserve(num_diff_features_);
    active_diff_feature_bin_mappers_.reserve(num_diff_features_);
    routing_original_most_freq_bins_.assign(
        this->num_feature_, std::numeric_limits<uint32_t>::max());
    for (int j = 0; j < num_diff_features_; ++j) {
      diff_feature_bin_mappers_.emplace_back(diff_feature_bin_mappers[j]);
      diff_feature_hist_offsets_.push_back(all_offsets[2 * this->num_feature_ + j]);
      diff_feature_bin_value_offsets_.push_back(
          1u - static_cast<uint32_t>(diff_feature_bin_mappers_[j]->GetMostFreqBin() == 0));
      original_feature_bin_value_offsets_.push_back(
          1u - static_cast<uint32_t>(original_feature_bin_mappers[j]->GetMostFreqBin() == 0));
      original_feature_most_freq_bins_.push_back(original_feature_bin_mappers[j]->GetMostFreqBin());
      const int original_feature_slot =
          diff_feature_to_original_feature_slot_[j];
      CHECK_GE(original_feature_slot, 0);
      CHECK_LT(original_feature_slot, this->num_feature_);
      routing_original_most_freq_bins_[original_feature_slot] =
          original_feature_bin_mappers[j]->GetMostFreqBin();
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
          active_diff_feature_hist_ti_.push_back(
              (diff_feature_hist_offsets_[j] + diff_feature_bin_value_offsets_[j]) << 1);
          active_diff_feature_bin_mappers_.push_back(diff_feature_bin_mappers_[j].get());
          use_fast_pairwise_bin_lookup_ &= diff_feature_bin_mappers_[j]->HasPairwiseBinRanges();
        }
      } else {
        diff_feature_raw_data_ptrs_.push_back(nullptr);
      }
    }
    InitDiffFeatureRawValueBuffer(num_data);
    InitTernaryDifferentialBins(num_pairwise_data);
  }

  void FinishLoad() override {
    const uint32_t max_ti =
        this->offsets_.empty() ? 0u : ((this->offsets_.back() - 1u) << 1);
    if (max_ti > std::numeric_limits<uint16_t>::max()) {
      return;
    }
    nonzero_row_offsets_.resize(static_cast<size_t>(this->num_data_) + 1u);
    #pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS())
    for (data_size_t row = 0; row < this->num_data_; ++row) {
      const BIN_TYPE* bins = this->data_.data() + this->RowPtr(row);
      uint32_t count = 0;
      for (int j = 0; j < this->num_feature_; ++j) {
        count += static_cast<uint32_t>(bins[j] != 0);
      }
      nonzero_row_offsets_[static_cast<size_t>(row) + 1u] = count;
    }
    uint64_t total_nonzero = 0;
    for (data_size_t row = 0; row < this->num_data_; ++row) {
      const uint32_t count =
          nonzero_row_offsets_[static_cast<size_t>(row) + 1u];
      CHECK_LE(total_nonzero + count,
               static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
      nonzero_row_offsets_[static_cast<size_t>(row)] =
          static_cast<uint32_t>(total_nonzero);
      total_nonzero += count;
    }
    nonzero_row_offsets_[static_cast<size_t>(this->num_data_)] =
        static_cast<uint32_t>(total_nonzero);
    nonzero_hist_ti_.resize(static_cast<size_t>(total_nonzero));
    #pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS())
    for (data_size_t row = 0; row < this->num_data_; ++row) {
      const BIN_TYPE* bins = this->data_.data() + this->RowPtr(row);
      uint32_t dst = nonzero_row_offsets_[row];
      for (int j = 0; j < this->num_feature_; ++j) {
        if (bins[j] != 0) {
          nonzero_hist_ti_[dst++] = static_cast<uint16_t>(
              (static_cast<uint32_t>(bins[j]) + this->offsets_[j]) << 1);
        }
      }
      CHECK_EQ(dst, nonzero_row_offsets_[static_cast<size_t>(row) + 1u]);
    }
  }

  bool GetPairwiseRowData(PairwiseRowData* out) const override {
    if (out == nullptr || this->paired_ranking_item_global_index_map_ == nullptr ||
        !use_fast_pairwise_bin_lookup_ ||
        num_active_diff_features_ != num_diff_features_ ||
        std::find(routing_original_most_freq_bins_.begin(),
                  routing_original_most_freq_bins_.end(),
                  std::numeric_limits<uint32_t>::max()) !=
            routing_original_most_freq_bins_.end() ||
        (num_diff_features_ > 0 && ternary_diff_bins_.empty())) {
      return false;
    }
    out->original_bins = this->data_.data();
    out->pair_indices = this->paired_ranking_item_global_index_map_;
    out->ternary_bins = ternary_diff_bins_.data();
    out->differential_raw_values = diff_feature_raw_values_.data();
    out->original_most_freq_bins =
        routing_original_most_freq_bins_.data();
    out->num_pairwise_rows = num_pairwise_data_;
    out->num_original_rows = this->num_data_;
    out->num_original_features = this->num_feature_;
    out->num_differential_features = num_diff_features_;
    out->ternary_bytes_per_row = ternary_diff_bin_bytes_;
    out->differential_raw_stride = num_active_diff_features_;
    out->original_bin_bytes = sizeof(BIN_TYPE);
    return true;
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
    if (start >= end) {
      return;
    }
    data_size_t i = start;
    hist_t* grad = out;
    hist_t* hess = out + 1;
    const auto base_offset = this->offsets_.back();
    const data_size_t num_original_data = this->num_data_;
    const int tid = omp_get_thread_num();
    CHECK_LT(tid, static_cast<int>(pointwise_histogram_buffers_.size()));
    PointwiseHistogramBuffer& pointwise_buffer = pointwise_histogram_buffers_[tid];
    pointwise_buffer.Reset(num_original_data, static_cast<size_t>(end - start));
    std::vector<data_size_t>& first_touched = pointwise_buffer.first_touched;
    std::vector<data_size_t>& second_touched = pointwise_buffer.second_touched;
    const uint32_t stamp = pointwise_buffer.stamp;
    hist_t total_gradient = 0.0;
    hist_t total_hessian = 0.0;
    bool reconstruct_pointwise_zero_bin = false;
    if (!active_diff_feature_slots_.empty() && use_fast_pairwise_bin_lookup_) {
      const uint8_t* ternary_diff_bins = ternary_diff_bins_.data();
#if LIGHTGBM_TARGET_AVX512
      if (HasAVX512()) {
        ConstructTernaryHistogramAVX512<USE_INDICES, ORDERED>(
            data_indices, start, end, gradients, hessians,
            &pointwise_buffer, stamp, &total_gradient, &total_hessian, out);
        reconstruct_pointwise_zero_bin = true;
      } else
#endif
      {
        for (; i < end; ++i) {
          const auto idx = USE_INDICES ? data_indices[i] : i;
          if (USE_PREFETCH) {
            const data_size_t pf_i = i + 32;
            if (pf_i < end) {
              const auto pf_idx = USE_INDICES ? data_indices[pf_i] : pf_i;
              PREFETCH_T0(ternary_diff_bins +
                          static_cast<size_t>(pf_idx) * ternary_diff_bin_bytes_);
              if (!ORDERED) {
                PREFETCH_T0(gradients + pf_idx);
                PREFETCH_T0(hessians + pf_idx);
              }
            }
          }
          const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
          const score_t hessian = ORDERED ? hessians[i] : hessians[idx];
          if (gradient == 0.0 && hessian == 0.0) {
            continue;
          }
          const data_size_t first_idx =
              this->paired_ranking_item_global_index_map_[idx].first;
          const data_size_t second_idx =
              this->paired_ranking_item_global_index_map_[idx].second;

          pointwise_buffer.AddFirst(first_idx, gradient, hessian, stamp);
          pointwise_buffer.AddSecond(second_idx, gradient, hessian, stamp);
          const uint8_t* bins = ternary_diff_bins +
              static_cast<size_t>(idx) * ternary_diff_bin_bytes_;
          for (int j = 0; j < num_active_diff_features_; ++j) {
            const uint8_t bit = static_cast<uint8_t>(1u << (j & 7));
            const uint8_t* masks = bins + ((j >> 3) << 1);
            const uint32_t bin = (masks[0] & bit) ? 0u :
                ((masks[1] & bit) ? 2u : 1u);
            const auto ti = active_diff_feature_hist_ti_[j] + (bin << 1);
            grad[ti] += gradient;
            hess[ti] += hessian;
          }
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
        const score_t gradient = ORDERED ? gradients[i] : gradients[idx];
        const score_t hessian = ORDERED ? hessians[i] : hessians[idx];
        if (gradient == 0.0 && hessian == 0.0) {
          continue;
        }
        const data_size_t first_idx =
            this->paired_ranking_item_global_index_map_[idx].first;
        const data_size_t second_idx =
            this->paired_ranking_item_global_index_map_[idx].second;

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
#if LIGHTGBM_TARGET_AVX512
    if (HasAVX512()) {
      if (reconstruct_pointwise_zero_bin) {
        ConstructPointwiseHistogramAVX512Fused(
            first_touched, second_touched, pointwise_buffer, stamp, out);
        ReconstructPointwiseZeroBins(total_gradient, total_hessian, out);
      } else {
        ConstructPointwiseHistogramAVX512<false, false>(
            first_touched, pointwise_buffer, out);
        ConstructPointwiseHistogramAVX512<true, false>(
            second_touched, pointwise_buffer, out);
      }
    } else
#endif
    {
      for (const data_size_t row_idx : first_touched) {
        const BIN_TYPE* data_ptr = this->data_.data() + this->RowPtr(row_idx);
        const auto& entry = pointwise_buffer.entries[row_idx];
        const hist_t gradient = entry.first_gradient;
        const hist_t hessian = entry.first_hessian;
        for (int j = 0; j < this->num_feature_; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + this->offsets_[j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
      for (const data_size_t row_idx : second_touched) {
        const BIN_TYPE* data_ptr = this->data_.data() + this->RowPtr(row_idx);
        const auto& entry = pointwise_buffer.entries[row_idx];
        const hist_t gradient = entry.second_gradient;
        const hist_t hessian = entry.second_hessian;
        for (int j = 0; j < this->num_feature_; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + this->offsets_[j] + base_offset) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    }
  }

 private:
  struct PointwiseHistogramBuffer {
    struct Entry {
      hist_t first_gradient = 0.0;
      hist_t first_hessian = 0.0;
      hist_t second_gradient = 0.0;
      hist_t second_hessian = 0.0;
      uint32_t first_stamp = 0;
      uint32_t second_stamp = 0;
    };

    std::vector<Entry> entries;
    std::vector<data_size_t> first_touched;
    std::vector<data_size_t> second_touched;
    uint32_t stamp = 0;

    void Reset(data_size_t num_data, size_t reserve_size) {
      const size_t size = static_cast<size_t>(num_data);
      if (entries.size() != size) {
        entries.assign(size, Entry{});
        stamp = 0;
      } else if (stamp == std::numeric_limits<uint32_t>::max()) {
        for (auto& entry : entries) {
          entry.first_stamp = 0;
          entry.second_stamp = 0;
        }
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

    void AddFirst(data_size_t idx, hist_t gradient, hist_t hessian, uint32_t current_stamp) {
      Entry& entry = entries[idx];
      if (entry.first_stamp != current_stamp) {
        entry.first_stamp = current_stamp;
        entry.first_gradient = 0.0;
        entry.first_hessian = 0.0;
        first_touched.push_back(idx);
      }
      entry.first_gradient += gradient;
      entry.first_hessian += hessian;
    }

    void AddSecond(data_size_t idx, hist_t gradient, hist_t hessian, uint32_t current_stamp) {
      Entry& entry = entries[idx];
      if (entry.second_stamp != current_stamp) {
        entry.second_stamp = current_stamp;
        entry.second_gradient = 0.0;
        entry.second_hessian = 0.0;
        second_touched.push_back(idx);
      }
      entry.second_gradient += gradient;
      entry.second_hessian += hessian;
    }
  };

#if LIGHTGBM_TARGET_AVX512
  static bool HasAVX512() {
    static const bool supported = __builtin_cpu_supports("avx2") &&
                                  __builtin_cpu_supports("avx512f") &&
                                  __builtin_cpu_supports("avx512bw") &&
                                  __builtin_cpu_supports("avx512vl");
    return supported;
  }

  template <bool USE_INDICES, bool ORDERED>
  __attribute__((target("avx2,avx512f,avx512bw,avx512vl")))
  void ConstructTernaryHistogramAVX512(
      const data_size_t* data_indices, data_size_t start, data_size_t end,
      const score_t* gradients, const score_t* hessians,
      PointwiseHistogramBuffer* pointwise_buffer, uint32_t stamp,
      hist_t* total_gradient_out, hist_t* total_hessian_out,
      hist_t* out) const {
    const uint8_t* ternary_bins = ternary_diff_bins_.data();
    const uint32_t* histogram_ti = active_diff_feature_hist_ti_.data();
    constexpr data_size_t kTileSize = 256;
    data_size_t pair_indices[kTileSize];
    hist_t tile_gradients[kTileSize];
    hist_t tile_hessians[kTileSize];
    hist_t total_gradient = 0.0;
    hist_t total_hessian = 0.0;
    data_size_t pending_first = -1;
    hist_t first_gradient = 0.0;
    hist_t first_hessian = 0.0;
    for (data_size_t tile_start = start; tile_start < end; tile_start += kTileSize) {
      const data_size_t input_size = std::min(kTileSize, end - tile_start);
      data_size_t tile_size = 0;
      for (data_size_t k = 0; k < input_size; ++k) {
        const data_size_t row = tile_start + k;
        const data_size_t idx = USE_INDICES ? data_indices[row] : row;
        const hist_t gradient = ORDERED ? gradients[row] : gradients[idx];
        const hist_t hessian = ORDERED ? hessians[row] : hessians[idx];
        if (gradient == 0.0 && hessian == 0.0) {
          continue;
        }
        pair_indices[tile_size] = idx;
        tile_gradients[tile_size] = gradient;
        tile_hessians[tile_size] = hessian;
        ++tile_size;
        total_gradient += gradient;
        total_hessian += hessian;
        const auto pair = this->paired_ranking_item_global_index_map_[idx];
        if (pair.first != pending_first) {
          if (pending_first >= 0) {
            pointwise_buffer->AddFirst(
                pending_first, first_gradient, first_hessian, stamp);
          }
          pending_first = pair.first;
          first_gradient = gradient;
          first_hessian = hessian;
        } else {
          first_gradient += gradient;
          first_hessian += hessian;
        }
        pointwise_buffer->AddSecond(pair.second, gradient, hessian, stamp);
      }
      int j = 0;
      for (; j + 8 <= num_active_diff_features_; j += 8) {
        const __m256i bin0_indices = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(histogram_ti + j));
        const __m256i bin2_indices = _mm256_add_epi32(
            bin0_indices, _mm256_set1_epi32(4));
        __m512d gradient_bins0 = _mm512_i32gather_pd(bin0_indices, out, 8);
        __m512d gradient_bins2 = _mm512_i32gather_pd(bin2_indices, out, 8);
        __m512d hessian_bins0 = _mm512_i32gather_pd(bin0_indices, out + 1, 8);
        __m512d hessian_bins2 = _mm512_i32gather_pd(bin2_indices, out + 1, 8);
        for (data_size_t k = 0; k < tile_size; ++k) {
          const uint8_t* bins = ternary_bins +
              static_cast<size_t>(pair_indices[k]) * ternary_diff_bin_bytes_ +
              (j >> 2);
          const __mmask8 mask0 = static_cast<__mmask8>(bins[0]);
          const __mmask8 mask2 = static_cast<__mmask8>(bins[1]);
          const __m512d gradient = _mm512_set1_pd(tile_gradients[k]);
          const __m512d hessian = _mm512_set1_pd(tile_hessians[k]);
          gradient_bins0 = _mm512_mask_add_pd(gradient_bins0, mask0, gradient_bins0, gradient);
          gradient_bins2 = _mm512_mask_add_pd(gradient_bins2, mask2, gradient_bins2, gradient);
          hessian_bins0 = _mm512_mask_add_pd(hessian_bins0, mask0, hessian_bins0, hessian);
          hessian_bins2 = _mm512_mask_add_pd(hessian_bins2, mask2, hessian_bins2, hessian);
        }
        _mm512_i32scatter_pd(out, bin0_indices, gradient_bins0, 8);
        _mm512_i32scatter_pd(out, bin2_indices, gradient_bins2, 8);
        _mm512_i32scatter_pd(out + 1, bin0_indices, hessian_bins0, 8);
        _mm512_i32scatter_pd(out + 1, bin2_indices, hessian_bins2, 8);
      }
      for (; j < num_active_diff_features_; ++j) {
        for (data_size_t k = 0; k < tile_size; ++k) {
          const uint8_t* bins = ternary_bins +
              static_cast<size_t>(pair_indices[k]) * ternary_diff_bin_bytes_;
          const uint8_t bit = static_cast<uint8_t>(1u << (j & 7));
          const uint8_t* masks = bins + ((j >> 3) << 1);
          const uint8_t bin = (masks[0] & bit) ? 0u :
              ((masks[1] & bit) ? 2u : 1u);
          if (bin == 1) {
            continue;
          }
          const auto ti = histogram_ti[j] + 2 * bin;
          out[ti] += tile_gradients[k];
          out[ti + 1] += tile_hessians[k];
        }
      }
    }
    if (pending_first >= 0) {
      pointwise_buffer->AddFirst(
          pending_first, first_gradient, first_hessian, stamp);
    }
    for (int j = 0; j < num_active_diff_features_; ++j) {
      const auto ti = histogram_ti[j];
      out[ti + 2] = total_gradient - out[ti] - out[ti + 4];
      out[ti + 3] = total_hessian - out[ti + 1] - out[ti + 5];
    }
    *total_gradient_out = total_gradient;
    *total_hessian_out = total_hessian;
  }

  __attribute__((target("avx2,avx512f,avx512bw,avx512vl")))
  static __m256i LoadBins8(const uint8_t* bins) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(bins)));
  }

  __attribute__((target("avx2,avx512f,avx512bw,avx512vl")))
  static __m256i LoadBins8(const uint16_t* bins) {
    return _mm256_cvtepu16_epi32(_mm_loadu_si128(
        reinterpret_cast<const __m128i*>(bins)));
  }

  __attribute__((target("avx2,avx512f,avx512bw,avx512vl")))
  static __m256i LoadBins8(const uint32_t* bins) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bins));
  }

  template <bool SECOND, bool SKIP_ZERO>
  __attribute__((target("avx2,avx512f,avx512bw,avx512vl")))
  void ConstructPointwiseHistogramAVX512(
      const std::vector<data_size_t>& touched,
      const PointwiseHistogramBuffer& pointwise_buffer, hist_t* out) const {
    const uint32_t side_offset = SECOND ? this->offsets_.back() : 0;
    const __m256i side_offsets = _mm256_set1_epi32(side_offset);
    for (const data_size_t row_idx : touched) {
      const auto& entry = pointwise_buffer.entries[row_idx];
      const hist_t gradient = SECOND ? entry.second_gradient : entry.first_gradient;
      const hist_t hessian = SECOND ? entry.second_hessian : entry.first_hessian;
      const __m512d gradients8 = _mm512_set1_pd(gradient);
      const __m512d hessians8 = _mm512_set1_pd(hessian);
      const BIN_TYPE* bins = this->data_.data() + this->RowPtr(row_idx);
      int j = 0;
      for (; j + 8 <= this->num_feature_; j += 8) {
        const __m256i bin_values = LoadBins8(bins + j);
        __m256i indices = _mm256_add_epi32(
            bin_values,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(this->offsets_.data() + j)));
        indices = _mm256_slli_epi32(_mm256_add_epi32(indices, side_offsets), 1);
        if (SKIP_ZERO) {
          const __mmask8 nonzero_mask = _mm256_cmpneq_epi32_mask(
              bin_values, _mm256_setzero_si256());
          const __m512d old_gradients = _mm512_mask_i32gather_pd(
              _mm512_setzero_pd(), nonzero_mask, indices, out, 8);
          const __m512d old_hessians = _mm512_mask_i32gather_pd(
              _mm512_setzero_pd(), nonzero_mask, indices, out + 1, 8);
          _mm512_mask_i32scatter_pd(
              out, nonzero_mask, indices,
              _mm512_add_pd(old_gradients, gradients8), 8);
          _mm512_mask_i32scatter_pd(
              out + 1, nonzero_mask, indices,
              _mm512_add_pd(old_hessians, hessians8), 8);
        } else {
          const __m512d old_gradients = _mm512_i32gather_pd(indices, out, 8);
          const __m512d old_hessians = _mm512_i32gather_pd(indices, out + 1, 8);
          _mm512_i32scatter_pd(out, indices,
                              _mm512_add_pd(old_gradients, gradients8), 8);
          _mm512_i32scatter_pd(out + 1, indices,
                              _mm512_add_pd(old_hessians, hessians8), 8);
        }
      }
      for (; j < this->num_feature_; ++j) {
        if (SKIP_ZERO && bins[j] == 0) {
          continue;
        }
        const auto ti = (static_cast<uint32_t>(bins[j]) + this->offsets_[j] +
                         side_offset) << 1;
        out[ti] += gradient;
        out[ti + 1] += hessian;
      }
    }
  }

  template <bool UPDATE_FIRST, bool UPDATE_SECOND,
            bool OVERLAP_GATHERS = false>
  __attribute__((target("avx2,avx512f,avx512bw,avx512vl"), always_inline))
  inline void AddPointwiseHistogramRowAVX512(
      data_size_t row_idx, const PointwiseHistogramBuffer& pointwise_buffer,
      hist_t* out) const {
    const auto& entry = pointwise_buffer.entries[row_idx];
    const __m512d first_gradients = _mm512_set1_pd(entry.first_gradient);
    const __m512d first_hessians = _mm512_set1_pd(entry.first_hessian);
    const __m512d second_gradients = _mm512_set1_pd(entry.second_gradient);
    const __m512d second_hessians = _mm512_set1_pd(entry.second_hessian);
    const BIN_TYPE* bins = this->data_.data() + this->RowPtr(row_idx);
    const __m256i zeros = _mm256_setzero_si256();
    const __m256i second_ti_offsets =
        _mm256_set1_epi32(this->offsets_.back() << 1);
    int j = 0;
    for (; j + 8 <= this->num_feature_; j += 8) {
      const __m256i bin_values = LoadBins8(bins + j);
      const __mmask8 nonzero_mask =
          _mm256_cmpneq_epi32_mask(bin_values, zeros);
      __m256i indices = _mm256_add_epi32(
          bin_values,
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
              this->offsets_.data() + j)));
      indices = _mm256_slli_epi32(indices, 1);
      if (OVERLAP_GATHERS) {
        const __m512d old_first_gradients = _mm512_mask_i32gather_pd(
            _mm512_setzero_pd(), nonzero_mask, indices, out, 8);
        const __m512d old_first_hessians = _mm512_mask_i32gather_pd(
            _mm512_setzero_pd(), nonzero_mask, indices, out + 1, 8);
        const __m256i second_indices =
            _mm256_add_epi32(indices, second_ti_offsets);
        const __m512d old_second_gradients = _mm512_mask_i32gather_pd(
            _mm512_setzero_pd(), nonzero_mask, second_indices, out, 8);
        const __m512d old_second_hessians = _mm512_mask_i32gather_pd(
            _mm512_setzero_pd(), nonzero_mask, second_indices, out + 1, 8);
        _mm512_mask_i32scatter_pd(
            out, nonzero_mask, indices,
            _mm512_add_pd(old_first_gradients, first_gradients), 8);
        _mm512_mask_i32scatter_pd(
            out + 1, nonzero_mask, indices,
            _mm512_add_pd(old_first_hessians, first_hessians), 8);
        _mm512_mask_i32scatter_pd(
            out, nonzero_mask, second_indices,
            _mm512_add_pd(old_second_gradients, second_gradients), 8);
        _mm512_mask_i32scatter_pd(
            out + 1, nonzero_mask, second_indices,
            _mm512_add_pd(old_second_hessians, second_hessians), 8);
      } else {
        if (UPDATE_FIRST) {
          const __m512d old_gradients = _mm512_mask_i32gather_pd(
              _mm512_setzero_pd(), nonzero_mask, indices, out, 8);
          const __m512d old_hessians = _mm512_mask_i32gather_pd(
              _mm512_setzero_pd(), nonzero_mask, indices, out + 1, 8);
          _mm512_mask_i32scatter_pd(
              out, nonzero_mask, indices,
              _mm512_add_pd(old_gradients, first_gradients), 8);
          _mm512_mask_i32scatter_pd(
              out + 1, nonzero_mask, indices,
              _mm512_add_pd(old_hessians, first_hessians), 8);
        }
        if (UPDATE_SECOND) {
          const __m256i second_indices =
              _mm256_add_epi32(indices, second_ti_offsets);
          const __m512d old_gradients = _mm512_mask_i32gather_pd(
              _mm512_setzero_pd(), nonzero_mask, second_indices, out, 8);
          const __m512d old_hessians = _mm512_mask_i32gather_pd(
              _mm512_setzero_pd(), nonzero_mask, second_indices, out + 1, 8);
          _mm512_mask_i32scatter_pd(
              out, nonzero_mask, second_indices,
              _mm512_add_pd(old_gradients, second_gradients), 8);
          _mm512_mask_i32scatter_pd(
              out + 1, nonzero_mask, second_indices,
              _mm512_add_pd(old_hessians, second_hessians), 8);
        }
      }
    }
    for (; j < this->num_feature_; ++j) {
      if (bins[j] == 0) {
        continue;
      }
      const uint32_t first_ti =
          (static_cast<uint32_t>(bins[j]) + this->offsets_[j]) << 1;
      if (UPDATE_FIRST) {
        out[first_ti] += entry.first_gradient;
        out[first_ti + 1] += entry.first_hessian;
      }
      if (UPDATE_SECOND) {
        const uint32_t second_ti =
            first_ti + (this->offsets_.back() << 1);
        out[second_ti] += entry.second_gradient;
        out[second_ti + 1] += entry.second_hessian;
      }
    }
  }

  __attribute__((target("avx2,avx512f,avx512bw,avx512vl"), noinline))
  void AddPointwiseHistogramRowAVX512Overlap(
      data_size_t row_idx,
      const PointwiseHistogramBuffer& pointwise_buffer,
      hist_t* out) const {
    if (nonzero_hist_ti_.empty()) {
      AddPointwiseHistogramRowAVX512<true, true, true>(
          row_idx, pointwise_buffer, out);
    } else {
      AddPointwiseSparseHistogramRowAVX512<true, true, true>(
          row_idx, pointwise_buffer, out);
    }
  }

  template <bool UPDATE_FIRST, bool UPDATE_SECOND,
            bool OVERLAP_GATHERS = false>
  __attribute__((target("avx2,avx512f,avx512bw,avx512vl"), always_inline))
  inline void AddPointwiseSparseHistogramRowAVX512(
      data_size_t row_idx, const PointwiseHistogramBuffer& pointwise_buffer,
      hist_t* out) const {
    const auto& entry = pointwise_buffer.entries[row_idx];
    const uint32_t begin = nonzero_row_offsets_[row_idx];
    const uint32_t end = nonzero_row_offsets_[static_cast<size_t>(row_idx) + 1u];
    uint32_t k = begin;
    for (; k < end; ++k) {
      const uint32_t first_ti = nonzero_hist_ti_[k];
      if (UPDATE_FIRST) {
        out[first_ti] += entry.first_gradient;
        out[first_ti + 1] += entry.first_hessian;
      }
      if (UPDATE_SECOND) {
        const uint32_t second_ti =
            first_ti + (this->offsets_.back() << 1);
        out[second_ti] += entry.second_gradient;
        out[second_ti + 1] += entry.second_hessian;
      }
    }
  }

  template <bool UPDATE_FIRST, bool UPDATE_SECOND>
  __attribute__((target("avx2,avx512f,avx512bw,avx512vl"), always_inline))
  inline void AddPointwiseHistogramRowOptimized(
      data_size_t row_idx, const PointwiseHistogramBuffer& pointwise_buffer,
      hist_t* out) const {
    if (nonzero_hist_ti_.empty()) {
      AddPointwiseHistogramRowAVX512<UPDATE_FIRST, UPDATE_SECOND>(
          row_idx, pointwise_buffer, out);
    } else {
      AddPointwiseSparseHistogramRowAVX512<UPDATE_FIRST, UPDATE_SECOND>(
          row_idx, pointwise_buffer, out);
    }
  }

  __attribute__((target("avx2,avx512f,avx512bw,avx512vl"), always_inline))
  inline void PrefetchPointwiseHistogramRow(
      data_size_t row_idx,
      const PointwiseHistogramBuffer& pointwise_buffer) const {
    PREFETCH_T0(pointwise_buffer.entries.data() + row_idx);
    if (!nonzero_hist_ti_.empty()) {
      const uint32_t begin = nonzero_row_offsets_[row_idx];
      const uint32_t end =
          nonzero_row_offsets_[static_cast<size_t>(row_idx) + 1u];
      const char* row = reinterpret_cast<const char*>(
          nonzero_hist_ti_.data() + begin);
      const size_t row_bytes = static_cast<size_t>(end - begin) *
                               sizeof(nonzero_hist_ti_[0]);
      for (size_t offset = 0; offset < row_bytes; offset += 64) {
        PREFETCH_T0(row + offset);
      }
      return;
    }
    const char* row = reinterpret_cast<const char*>(
        this->data_.data() + this->RowPtr(row_idx));
    const size_t row_bytes =
        static_cast<size_t>(this->num_feature_) * sizeof(BIN_TYPE);
    for (size_t offset = 0; offset < row_bytes; offset += 64) {
      PREFETCH_T0(row + offset);
    }
  }

  __attribute__((target("avx2,avx512f,avx512bw,avx512vl")))
  void ConstructPointwiseHistogramAVX512Fused(
      const std::vector<data_size_t>& first_touched,
      const std::vector<data_size_t>& second_touched,
      const PointwiseHistogramBuffer& pointwise_buffer, uint32_t stamp,
      hist_t* out) const {
    constexpr size_t kPrefetchDistance = 8;
    const bool overlap_gathers = first_touched.size() > 5000;
    for (size_t p = 0; p < first_touched.size(); ++p) {
      if (p + kPrefetchDistance < first_touched.size()) {
        PrefetchPointwiseHistogramRow(
            first_touched[p + kPrefetchDistance], pointwise_buffer);
      }
      const data_size_t row_idx = first_touched[p];
      if (pointwise_buffer.entries[row_idx].second_stamp == stamp) {
        if (overlap_gathers) {
          AddPointwiseHistogramRowAVX512Overlap(
              row_idx, pointwise_buffer, out);
        } else {
          AddPointwiseHistogramRowOptimized<true, true>(
              row_idx, pointwise_buffer, out);
        }
      } else {
        AddPointwiseHistogramRowOptimized<true, false>(
            row_idx, pointwise_buffer, out);
      }
    }
    for (size_t p = 0; p < second_touched.size(); ++p) {
      if (p + kPrefetchDistance < second_touched.size()) {
        PrefetchPointwiseHistogramRow(
            second_touched[p + kPrefetchDistance], pointwise_buffer);
      }
      const data_size_t row_idx = second_touched[p];
      if (pointwise_buffer.entries[row_idx].first_stamp != stamp) {
        AddPointwiseHistogramRowOptimized<false, true>(
            row_idx, pointwise_buffer, out);
      }
    }
  }

  void ReconstructPointwiseZeroBins(
      hist_t total_gradient, hist_t total_hessian, hist_t* out) const {
    const uint32_t base_offset = this->offsets_.back();
    for (uint32_t side_offset : {0u, base_offset}) {
      for (int j = 0; j < this->num_feature_; ++j) {
        hist_t zero_gradient = total_gradient;
        hist_t zero_hessian = total_hessian;
        for (uint32_t bin = this->offsets_[j] + 1;
             bin < this->offsets_[j + 1]; ++bin) {
          const uint32_t ti = (bin + side_offset) << 1;
          zero_gradient -= out[ti];
          zero_hessian -= out[ti + 1];
        }
        const uint32_t zero_ti =
            (this->offsets_[j] + side_offset) << 1;
        out[zero_ti] = zero_gradient;
        out[zero_ti + 1] = zero_hessian;
      }
    }
  }

#endif

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

  void InitTernaryDifferentialBins(data_size_t num_pairwise_data) {
    if (!use_fast_pairwise_bin_lookup_ || num_active_diff_features_ == 0) {
      return;
    }
    ternary_diff_bin_bytes_ = 2 * ((num_active_diff_features_ + 7) / 8);
    ternary_diff_bins_.assign(
        static_cast<size_t>(num_pairwise_data) * ternary_diff_bin_bytes_, 0);
    const float* raw_values = diff_feature_raw_values_.data();
    #pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS())
    for (data_size_t i = 0; i < num_pairwise_data; ++i) {
      const auto pair = this->paired_ranking_item_global_index_map_[i];
      const float* first = raw_values +
          static_cast<size_t>(pair.first) * num_active_diff_features_;
      const float* second = raw_values +
          static_cast<size_t>(pair.second) * num_active_diff_features_;
      uint8_t* bins = ternary_diff_bins_.data() +
          static_cast<size_t>(i) * ternary_diff_bin_bytes_;
      for (int j = 0; j < num_active_diff_features_; ++j) {
        const float diff = first[j] - second[j];
        const uint8_t bit = static_cast<uint8_t>(1u << (j & 7));
        uint8_t* masks = bins + ((j >> 3) << 1);
        if (diff <= static_cast<float>(-kEpsilon)) {
          masks[0] |= bit;
        } else if (diff > static_cast<float>(kEpsilon)) {
          masks[1] |= bit;
        }
      }
    }
  }

  std::vector<std::unique_ptr<const BinMapper>> diff_feature_bin_mappers_;
  std::vector<int> diff_feature_to_original_feature_slot_;
  std::vector<uint32_t> diff_feature_hist_offsets_;
  std::vector<uint32_t> diff_feature_bin_value_offsets_;
  std::vector<uint32_t> original_feature_bin_value_offsets_;
  std::vector<uint32_t> original_feature_most_freq_bins_;
  std::vector<uint32_t> routing_original_most_freq_bins_;
  std::vector<const float*> diff_feature_raw_data_ptrs_;
  std::vector<float> diff_feature_raw_values_;
  std::vector<uint8_t> ternary_diff_bins_;
  std::vector<uint32_t> nonzero_row_offsets_;
  std::vector<uint16_t> nonzero_hist_ti_;
  std::vector<int> active_diff_feature_slots_;
  std::vector<uint32_t> active_diff_feature_hist_offsets_;
  std::vector<uint32_t> active_diff_feature_bin_value_offsets_;
  std::vector<uint32_t> active_diff_feature_hist_ti_;
  std::vector<const BinMapper*> active_diff_feature_bin_mappers_;
  mutable std::vector<PointwiseHistogramBuffer> pointwise_histogram_buffers_;
  int num_diff_features_;
  int num_active_diff_features_ = 0;
  int ternary_diff_bin_bytes_ = 0;
  data_size_t num_pairwise_data_ = 0;
  bool use_fast_pairwise_bin_lookup_;
};

}  // namespace LightGBM

#undef LIGHTGBM_TARGET_AVX512

#endif  // LIGHTGBM_IO_MULTI_VAL_PAIRWISE_LAMBDARANK_BIN_HPP_
