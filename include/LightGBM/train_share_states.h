/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TRAIN_SHARE_STATES_H_
#define LIGHTGBM_TRAIN_SHARE_STATES_H_

#include <memory>
#include <vector>
#include <LightGBM/bin.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/feature_group.h>

namespace LightGBM {

class MultiValBinWrapper {
    private:
      bool is_use_subcol_ = false;
      bool is_use_subrow_ = false;
      bool is_subrow_copied_ = false;
      std::unique_ptr<MultiValBin> multi_val_bin_;
      std::unique_ptr<MultiValBin> multi_val_bin_subset_;
      std::vector<uint32_t> hist_move_src_;
      std::vector<uint32_t> hist_move_dest_;
      std::vector<uint32_t> hist_move_size_;
      const std::vector<int> feature_groups_contained_;

      int num_threads_;
      int max_block_size_;
      int num_bin_;
      int num_bin_aligned_;
      int n_data_block_;
      int data_block_size_;
      int min_block_size_;
      int num_data_;

      hist_t* origin_hist_data_;

      const size_t kHistBufferEntrySize = 2 * sizeof(hist_t);

    public:
      MultiValBinWrapper(MultiValBin* bin, data_size_t num_data,
        const std::vector<int>& feature_groups_contained): 
          feature_groups_contained_(feature_groups_contained) {
        num_threads_ = OMP_NUM_THREADS();
        max_block_size_ = num_data;
        num_data_ = num_data;
        if (bin == nullptr) {
          return;
        }
        multi_val_bin_.reset(bin);
        num_bin_ = bin->num_bin();
        num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
      }

      bool IsSparse() {
        if (multi_val_bin_ != nullptr) {
          return multi_val_bin_->IsSparse();
        }
        return false;
      }

      void InitTrain(const std::vector<int>& group_feature_start,
        const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
        const std::vector<int8_t>& is_feature_used,
        const data_size_t* bagging_use_indices,
        data_size_t bagging_indices_cnt) {
        is_use_subcol_ = false;
        if (multi_val_bin_ == nullptr) {
          return;
        }
        CopyMultiValBinSubset(group_feature_start, feature_groups,
          is_feature_used, bagging_use_indices, bagging_indices_cnt);
        const auto cur_multi_val_bin = (is_use_subcol_ || is_use_subrow_)
              ? multi_val_bin_subset_.get()
              : multi_val_bin_.get();
        if (cur_multi_val_bin != nullptr) {
          num_bin_ = cur_multi_val_bin->num_bin();
          num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
          min_block_size_ = std::min<int>(static_cast<int>(0.3f * num_bin_ /
            cur_multi_val_bin->num_element_per_row()) + 1, 1024);
        }
      }

      void HistMove(const std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>& hist_buf) {
        if (!is_use_subcol_) {
          return;
        }
        const hist_t* src = hist_buf.data() + hist_buf.size() - 2 * static_cast<size_t>(num_bin_aligned_);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(hist_move_src_.size()); ++i) {
          std::copy_n(src + hist_move_src_[i], hist_move_size_[i],
                      origin_hist_data_ + hist_move_dest_[i]);
        }
      }

      void HistMerge(std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf) {
        int n_bin_block = 1;
        int bin_block_size = num_bin_;
        Threading::BlockInfo<data_size_t>(num_threads_, num_bin_, 512, &n_bin_block,
                                        &bin_block_size);
        hist_t* dst = origin_hist_data_;
        if (is_use_subcol_) {
          dst = hist_buf->data() + hist_buf->size() - 2 * static_cast<size_t>(num_bin_aligned_);
        }
        #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
        for (int t = 0; t < n_bin_block; ++t) {
          const int start = t * bin_block_size;
          const int end = std::min(start + bin_block_size, num_bin_);
          for (int tid = 1; tid < n_data_block_; ++tid) {
            auto src_ptr = hist_buf->data() + static_cast<size_t>(num_bin_aligned_) * 2 * (tid - 1);
            for (int i = start * 2; i < end * 2; ++i) {
              dst[i] += src_ptr[i];
            }
          }
        }
      }

      void ResizeHistBuf(std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf,
        MultiValBin* sub_multi_val_bin,
        hist_t* origin_hist_data) {
        num_bin_ = sub_multi_val_bin->num_bin();
        num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
        origin_hist_data_ = origin_hist_data;
        size_t new_buf_size = static_cast<size_t>(n_data_block_) * static_cast<size_t>(num_bin_aligned_) * 2;
        if (hist_buf->size() < new_buf_size) {
          hist_buf->resize(new_buf_size);
        }
      }

      template <bool USE_INDICES, bool ORDERED>
      void ConstructHistograms(const data_size_t* data_indices,
          data_size_t num_data,
          const score_t* gradients,
          const score_t* hessians,
          std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf,
          hist_t* origin_hist_data) {
        global_timer.Start("Dataset::sparse_bin_histogram");
        const auto cur_multi_val_bin = (is_use_subcol_ || is_use_subrow_)
              ? multi_val_bin_subset_.get()
              : multi_val_bin_.get();
        if (cur_multi_val_bin != nullptr) {
          n_data_block_ = 1;
          data_block_size_ = num_data;
          Threading::BlockInfo<data_size_t>(num_threads_, num_data, min_block_size_,
                                            max_block_size_, &n_data_block_, &data_block_size_);
          ResizeHistBuf(hist_buf, cur_multi_val_bin, origin_hist_data);
          OMP_INIT_EX();
          #pragma omp parallel for schedule(static) num_threads(num_threads_)
          for (int block_id = 0; block_id < n_data_block_; ++block_id) {
            OMP_LOOP_EX_BEGIN();
            data_size_t start = block_id * data_block_size_;
            data_size_t end = std::min(start + data_block_size_, num_data);
            ConstructHistogramsForBlock<USE_INDICES, ORDERED>(
              cur_multi_val_bin, start, end, data_indices, gradients, hessians,
              block_id, hist_buf
            );
            OMP_LOOP_EX_END();
          }
          OMP_THROW_EX();
          global_timer.Stop("Dataset::sparse_bin_histogram");

          global_timer.Start("Dataset::sparse_bin_histogram_merge");
          HistMerge(hist_buf);
          global_timer.Stop("Dataset::sparse_bin_histogram_merge");
          global_timer.Start("Dataset::sparse_bin_histogram_move");
          HistMove(*hist_buf);
          global_timer.Stop("Dataset::sparse_bin_histogram_move");
        }
      }

      template <bool USE_INDICES, bool ORDERED>
      void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
        data_size_t start, data_size_t end, const data_size_t* data_indices,
        const score_t* gradients, const score_t* hessians, int block_id,
        std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf) {
        hist_t* data_ptr = origin_hist_data_;
        if (block_id == 0) {
          if (is_use_subcol_) {
            data_ptr = hist_buf->data() + hist_buf->size() - 2 * static_cast<size_t>(num_bin_aligned_);
          }
        } else {
          data_ptr = hist_buf->data() +
            static_cast<size_t>(num_bin_aligned_) * (block_id - 1) * 2;
        }
        std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kHistBufferEntrySize);
        if (USE_INDICES) {
          if (ORDERED) {
            sub_multi_val_bin->ConstructHistogramOrdered(data_indices, start, end,
                                                    gradients, hessians, data_ptr);
          } else {
            sub_multi_val_bin->ConstructHistogram(data_indices, start, end, gradients,
                                              hessians, data_ptr);
          }
        } else {
          sub_multi_val_bin->ConstructHistogram(start, end, gradients, hessians,
                                            data_ptr);
        }
      }

      void CopyMultiValBinSubset(const std::vector<int>& group_feature_start,
        const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
        const std::vector<int8_t>& is_feature_used,
        const data_size_t* bagging_use_indices,
        data_size_t bagging_indices_cnt) {
        double sum_used_dense_ratio = 0.0;
        double sum_dense_ratio = 0.0;
        int num_used = 0;
        int total = 0;
        std::vector<int> used_feature_index;
        for (int i : feature_groups_contained_) {
          int f_start = group_feature_start[i];
          if (feature_groups[i]->is_multi_val_) {
            for (int j = 0; j < feature_groups[i]->num_feature_; ++j) {
              const auto dense_rate =
                  1.0 - feature_groups[i]->bin_mappers_[j]->sparse_rate();
              if (is_feature_used[f_start + j]) {
                ++num_used;
                used_feature_index.push_back(total);
                sum_used_dense_ratio += dense_rate;
              }
              sum_dense_ratio += dense_rate;
              ++total;
            }
          } else {
            bool is_group_used = false;
            double dense_rate = 0;
            for (int j = 0; j < feature_groups[i]->num_feature_; ++j) {
              if (is_feature_used[f_start + j]) {
                is_group_used = true;
              }
              dense_rate += 1.0 - feature_groups[i]->bin_mappers_[j]->sparse_rate();
            }
            if (is_group_used) {
              ++num_used;
              used_feature_index.push_back(total);
              sum_used_dense_ratio += dense_rate;
            }
            sum_dense_ratio += dense_rate;
            ++total;
          }
        }
        const double k_subfeature_threshold = 0.6;
        if (sum_used_dense_ratio >= sum_dense_ratio * k_subfeature_threshold) {
          // only need to copy subset
          if (is_use_subrow_ && !is_subrow_copied_) {
            if (multi_val_bin_subset_ == nullptr) {
              multi_val_bin_subset_.reset(multi_val_bin_->CreateLike(
                  bagging_indices_cnt, multi_val_bin_->num_bin(), total,
                  multi_val_bin_->num_element_per_row(), multi_val_bin_->offsets()));
            } else {
              multi_val_bin_subset_->ReSize(
                  bagging_indices_cnt, multi_val_bin_->num_bin(), total,
                  multi_val_bin_->num_element_per_row(), multi_val_bin_->offsets());
            }
            multi_val_bin_subset_->CopySubrow(
                multi_val_bin_.get(), bagging_use_indices,
                bagging_indices_cnt);
            // avoid to copy subset many times
            is_subrow_copied_ = true;
          }
        } else {
          is_use_subcol_ = true;
          std::vector<uint32_t> upper_bound;
          std::vector<uint32_t> lower_bound;
          std::vector<uint32_t> delta;
          std::vector<uint32_t> offsets;
          hist_move_src_.clear();
          hist_move_dest_.clear();
          hist_move_size_.clear();

          const int offset = multi_val_bin_->IsSparse() ? 1 : 0;
          int num_total_bin = offset;
          int new_num_total_bin = offset;
          offsets.push_back(static_cast<uint32_t>(new_num_total_bin));
          for (int i : feature_groups_contained_) {
            int f_start = group_feature_start[i];
            if (feature_groups[i]->is_multi_val_) {
              for (int j = 0; j < feature_groups[i]->num_feature_; ++j) {
                const auto& bin_mapper = feature_groups[i]->bin_mappers_[j];
                int cur_num_bin = bin_mapper->num_bin();
                if (bin_mapper->GetMostFreqBin() == 0) {
                  cur_num_bin -= offset;
                }
                num_total_bin += cur_num_bin;
                if (is_feature_used[f_start + j]) {
                  new_num_total_bin += cur_num_bin;
                  offsets.push_back(static_cast<uint32_t>(new_num_total_bin));
                  lower_bound.push_back(num_total_bin - cur_num_bin);
                  upper_bound.push_back(num_total_bin);

                  hist_move_src_.push_back(
                      (new_num_total_bin - cur_num_bin) * 2);
                  hist_move_dest_.push_back((num_total_bin - cur_num_bin) *
                                                      2);
                  hist_move_size_.push_back(cur_num_bin * 2);
                  delta.push_back(num_total_bin - new_num_total_bin);
                }
              }
            } else {
              bool is_group_used = false;
              for (int j = 0; j < feature_groups[i]->num_feature_; ++j) {
                if (is_feature_used[f_start + j]) {
                  is_group_used = true;
                  break;
                }
              }
              int cur_num_bin = feature_groups[i]->bin_offsets_.back() - offset;
              num_total_bin += cur_num_bin;
              if (is_group_used) {
                new_num_total_bin += cur_num_bin;
                offsets.push_back(static_cast<uint32_t>(new_num_total_bin));
                lower_bound.push_back(num_total_bin - cur_num_bin);
                upper_bound.push_back(num_total_bin);

                hist_move_src_.push_back(
                    (new_num_total_bin - cur_num_bin) * 2);
                hist_move_dest_.push_back((num_total_bin - cur_num_bin) *
                                                    2);
                hist_move_size_.push_back(cur_num_bin * 2);
                delta.push_back(num_total_bin - new_num_total_bin);
              }
            }
          }
          // avoid out of range
          lower_bound.push_back(num_total_bin);
          upper_bound.push_back(num_total_bin);
          data_size_t num_data = is_use_subrow_ ? bagging_indices_cnt : num_data_;
          if (multi_val_bin_subset_ == nullptr) {
            multi_val_bin_subset_.reset(multi_val_bin_->CreateLike(
                num_data, new_num_total_bin, num_used, sum_used_dense_ratio, offsets));
          } else {
            multi_val_bin_subset_->ReSize(num_data, new_num_total_bin,
                                                    num_used, sum_used_dense_ratio, offsets);
          }
          if (is_use_subrow_) {
            multi_val_bin_subset_->CopySubrowAndSubcol(
                multi_val_bin_.get(), bagging_use_indices,
                bagging_indices_cnt, used_feature_index, lower_bound,
                upper_bound, delta);
            // may need to recopy subset
            is_subrow_copied_ = false;
          } else {
            multi_val_bin_subset_->CopySubcol(
                multi_val_bin_.get(), used_feature_index, lower_bound, upper_bound, delta);
          }
        }
      }

      void SetUseSubrow(bool is_use_subrow) {
        is_use_subrow_ = is_use_subrow;
      }

      void SetSubrowCopied(bool is_subrow_copied) {
        is_subrow_copied_ = is_subrow_copied;
      }
  };

struct TrainingShareStates {
  int num_threads = 0;
  bool is_colwise = true;
  bool is_two_rowwise = false;
  bool is_constant_hessian = true;
  const data_size_t* bagging_use_indices;
  data_size_t bagging_indices_cnt;

  uint64_t num_hist_total_bin() { return num_hist_total_bin_; }

  const std::vector<uint32_t>& feature_hist_offsets() { return feature_hist_offsets_; }

  bool IsSparseRowwise() {
    return (multi_val_bin_wappers_.size() == 1 &&
      multi_val_bin_wappers_[0]->IsSparse());
  }

  virtual void SetMultiValBin(MultiValBin* bin, data_size_t num_data,
    const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    bool dense_only, bool sparse_only, uint32_t hist_start_pos) {
    num_threads = OMP_NUM_THREADS();
    if (bin == nullptr) {
      return;
    }
    std::vector<int> feature_groups_contained;
    for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
      const auto& feature_group = feature_groups[group];
      if (feature_group->is_multi_val_) {
        if (!dense_only) {
          feature_groups_contained.push_back(group);
        }
      } else if (!sparse_only) {
        feature_groups_contained.push_back(group);
      }
    }
    multi_val_bin_wappers_.emplace_back(new MultiValBinWrapper(
      bin, num_data, feature_groups_contained
    ));
    hist_data_offsets_.push_back(static_cast<size_t>(hist_start_pos));
  }

  virtual void CalcBinOffsets(const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    std::vector<uint32_t>* offsets, std::vector<uint32_t>* offsets2,
    uint32_t* hist_start_pos1, uint32_t* hist_start_pos2,
    bool is_col_wise, bool is_two_row_wise) {
    offsets->clear();
    offsets2->clear();
    feature_hist_offsets_.clear();
    if (is_two_row_wise) {
      is_two_row_wise = false;
      for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
        if (feature_groups[group]->is_multi_val_) {
          if (!feature_groups[group]->is_dense_multi_val_ && feature_groups.size() > 1) {
            is_two_row_wise = true;
          }
        }
      }
    }
    if (is_col_wise) {
      uint32_t cur_num_bin = 0;
      uint32_t hist_cur_num_bin = 0;
      for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
        const std::unique_ptr<FeatureGroup>& feature_group = feature_groups[group];
        if (feature_group->is_multi_val_) {
          if (feature_group->is_dense_multi_val_) {
            for (int i = 0; i < feature_group->num_feature_; ++i) {
              const std::unique_ptr<BinMapper>& bin_mapper = feature_group->bin_mappers_[i];
              if (group == 0 && i == 0 && bin_mapper->GetMostFreqBin() > 0) {
                cur_num_bin += 1;
                hist_cur_num_bin += 1;
              }
              offsets->push_back(cur_num_bin);
              feature_hist_offsets_.push_back(hist_cur_num_bin);
              int num_bin = bin_mapper->num_bin();
              hist_cur_num_bin += num_bin;
              if (bin_mapper->GetMostFreqBin() == 0) {
                feature_hist_offsets_.back() += 1;
              }
              cur_num_bin += num_bin;
            }
            offsets->push_back(cur_num_bin);
            CHECK(cur_num_bin == feature_group->bin_offsets_.back());
          } else {
            cur_num_bin += 1;
            hist_cur_num_bin += 1;
            for (int i = 0; i < feature_group->num_feature_; ++i) {
              offsets->push_back(cur_num_bin);
              feature_hist_offsets_.push_back(hist_cur_num_bin);
              const std::unique_ptr<BinMapper>& bin_mapper = feature_group->bin_mappers_[i];
              int num_bin = bin_mapper->num_bin();
              if (bin_mapper->GetMostFreqBin() == 0) {
                num_bin -= 1;
              }
              hist_cur_num_bin += num_bin;
              cur_num_bin += num_bin;
            }
            offsets->push_back(cur_num_bin);
            CHECK(cur_num_bin == feature_group->bin_offsets_.back());
          }
        } else {
          for (int i = 0; i < feature_group->num_feature_; ++i) {
            feature_hist_offsets_.push_back(hist_cur_num_bin + feature_group->bin_offsets_[i]);
          }
          hist_cur_num_bin += feature_group->bin_offsets_.back();
        }
      }
      feature_hist_offsets_.push_back(hist_cur_num_bin);
      num_hist_total_bin_ = static_cast<uint64_t>(feature_hist_offsets_.back());
      *hist_start_pos1 = 0;
      *hist_start_pos2 = 0;
    } else if (!is_two_row_wise) {
      double sum_dense_ratio = 0.0f;
      int ncol = 0;
      for (int gid = 0; gid < static_cast<int>(feature_groups.size()); ++gid) {
        if (feature_groups[gid]->is_multi_val_) {
          ncol += feature_groups[gid]->num_feature_;
        } else {
          ncol += 1;
        }
        for (int fid = 0; fid < feature_groups[gid]->num_feature_; ++fid) {
          const auto& bin_mapper = feature_groups[gid]->bin_mappers_[fid];
          sum_dense_ratio += 1.0f - bin_mapper->sparse_rate();
        }
      }
      sum_dense_ratio /= ncol;
      const bool is_sparse_row_wise = (1.0f - sum_dense_ratio) >=
        MultiValBin::multi_val_bin_sparse_threshold ? 1 : 0;
      if (is_sparse_row_wise) {
        int cur_num_bin = 1;
        uint32_t hist_cur_num_bin = 1;
        for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
          const std::unique_ptr<FeatureGroup>& feature_group = feature_groups[group];
          if (feature_group->is_multi_val_) {
            for (int i = 0; i < feature_group->num_feature_; ++i) {
              offsets->push_back(cur_num_bin);
              feature_hist_offsets_.push_back(hist_cur_num_bin);
              const std::unique_ptr<BinMapper>& bin_mapper = feature_group->bin_mappers_[i];
              int num_bin = bin_mapper->num_bin();
              if (bin_mapper->GetMostFreqBin() == 0) {
                num_bin -= 1;
              }
              cur_num_bin += num_bin;
              hist_cur_num_bin += num_bin;
            }
          } else {
            offsets->push_back(cur_num_bin);
            cur_num_bin += feature_group->bin_offsets_.back() - 1;
            for (int i = 0; i < feature_group->num_feature_; ++i) {
              feature_hist_offsets_.push_back(hist_cur_num_bin + feature_group->bin_offsets_[i] - 1);
            }
            hist_cur_num_bin += feature_group->bin_offsets_.back() - 1;
          }
        }
        offsets->push_back(cur_num_bin);
        feature_hist_offsets_.push_back(hist_cur_num_bin);
      } else {
        int cur_num_bin = 0;
        uint32_t hist_cur_num_bin = 0;
        for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
          const std::unique_ptr<FeatureGroup>& feature_group = feature_groups[group];
          if (feature_group->is_multi_val_) {
            for (int i = 0; i < feature_group->num_feature_; ++i) {
              const std::unique_ptr<BinMapper>& bin_mapper = feature_group->bin_mappers_[i];
              if (group == 0 && i == 0 && bin_mapper->GetMostFreqBin() > 0) {
                cur_num_bin += 1;
                hist_cur_num_bin += 1;
              }
              offsets->push_back(cur_num_bin);
              feature_hist_offsets_.push_back(hist_cur_num_bin);
              int num_bin = bin_mapper->num_bin();
              cur_num_bin += num_bin;
              hist_cur_num_bin += num_bin;
              if (bin_mapper->GetMostFreqBin() == 0) {
                feature_hist_offsets_.back() += 1;
              }
            }
          } else {
            offsets->push_back(cur_num_bin);
            cur_num_bin += feature_group->bin_offsets_.back();
            for (int i = 0; i < feature_group->num_feature_; ++i) {
              feature_hist_offsets_.push_back(hist_cur_num_bin + feature_group->bin_offsets_[i]);
            }
            hist_cur_num_bin += feature_group->bin_offsets_.back();
          }
        }
        offsets->push_back(cur_num_bin);
        feature_hist_offsets_.push_back(hist_cur_num_bin);
      }
      num_hist_total_bin_ = static_cast<uint64_t>(feature_hist_offsets_.back());
      *hist_start_pos1 = 0;
      *hist_start_pos2 = 0;
    } else {
      std::vector<uint32_t> feature_hist_offsets1, feature_hist_offsets2;
      int multi_val_group_id = -1;
      bool is_sparse_multi_val = false;
      std::vector<std::vector<int>> features_in_group;
      int feature_offset = 0;
      for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
        const std::unique_ptr<FeatureGroup>& feature_group = feature_groups[group];
        features_in_group.emplace_back(0);
        if (feature_group->is_multi_val_) {
          multi_val_group_id = group;
          if (!feature_group->is_dense_multi_val_) {
            is_sparse_multi_val = true;
          }
        }
        for (int i = 0; i < feature_group->num_feature_; ++i) {
          features_in_group.back().push_back(feature_offset + i);
        }
        feature_offset += feature_group->num_feature_;
      }
      CHECK(multi_val_group_id >= 0);
      CHECK(is_sparse_multi_val);
      int cur_num_bin = 1;
      int cur_num_bin2 = 0;
      uint32_t hist_cur_num_bin = 1;
      uint32_t hist_cur_num_bin2 = 0;
      const std::unique_ptr<FeatureGroup>& feature_group = feature_groups[multi_val_group_id];
      for (int i = 0; i < feature_group->num_feature_; ++i) {
        offsets->push_back(cur_num_bin);
        feature_hist_offsets1.push_back(hist_cur_num_bin);
        const std::unique_ptr<BinMapper>& bin_mapper = feature_group->bin_mappers_[i];
        int num_bin = bin_mapper->num_bin();
        if (bin_mapper->GetMostFreqBin() == 0) {
          num_bin -= 1;
        }
        cur_num_bin += num_bin;
        hist_cur_num_bin += num_bin;
      }
      offsets->push_back(cur_num_bin);
      feature_hist_offsets1.push_back(hist_cur_num_bin);
      for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
        if (group == multi_val_group_id) {
          continue;
        }
        const std::unique_ptr<FeatureGroup>& feature_group = feature_groups[group];
        CHECK(!feature_group->is_multi_val_);
        offsets2->push_back(cur_num_bin2);
        cur_num_bin2 += feature_group->bin_offsets_.back();
        for (int i = 0; i < feature_group->num_feature_; ++i) {
          feature_hist_offsets2.push_back(hist_cur_num_bin2 + feature_group->bin_offsets_[i]);
        }
        hist_cur_num_bin2 += feature_group->bin_offsets_.back();
      }
      offsets2->push_back(cur_num_bin2);
      feature_hist_offsets2.push_back(hist_cur_num_bin2);
      feature_hist_offsets_.clear();
      if (multi_val_group_id == 0) {
        // put histograms of all dense features in the back
        for (int i = 0; i < static_cast<int>(feature_hist_offsets1.size()) - 1; ++i) {
          feature_hist_offsets_.push_back(feature_hist_offsets1[i]);
        }
        for (int i = 0; i < static_cast<int>(feature_hist_offsets2.size()); ++i) {
          feature_hist_offsets_.push_back(feature_hist_offsets2[i] + feature_hist_offsets1.back());
        }
        *hist_start_pos1 = 0;
        *hist_start_pos2 = feature_hist_offsets1.back();
      } else {
        size_t cur_dense_feature = 0;
        // put histograms of all dense features in the front
        for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
          const std::unique_ptr<FeatureGroup>& feature_group = feature_groups[group];
          if (feature_group->is_multi_val_) {
            for (int i = 0; i < feature_group->num_feature_; ++i) {
              feature_hist_offsets_.push_back(feature_hist_offsets1[i] + feature_hist_offsets2.back());
            }
          } else {
            for (int i = 0; i < feature_group->num_feature_; ++i) {
              feature_hist_offsets_.push_back(feature_hist_offsets2[cur_dense_feature++]);
            }
          }
        }
        feature_hist_offsets_.push_back(feature_hist_offsets1.back() + feature_hist_offsets2.back());
        *hist_start_pos1 = feature_hist_offsets2.back();
        *hist_start_pos2 = 0;
      }
      num_hist_total_bin_ = static_cast<uint64_t>(feature_hist_offsets1.back() + feature_hist_offsets2.back());
    }
  }

  virtual void InitTrain(const std::vector<int>& group_feature_start,
        const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
        const std::vector<int8_t>& is_feature_used) {
    for (const auto& multi_val_bin_wrapper : multi_val_bin_wappers_) {
      multi_val_bin_wrapper->InitTrain(group_feature_start,
        feature_groups,
        is_feature_used,
        bagging_use_indices,
        bagging_indices_cnt);
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructHistograms(const data_size_t* data_indices,
                          data_size_t num_data,
                          const score_t* gradients,
                          const score_t* hessians,
                          hist_t* hist_data) {
    global_timer.Start("Dataset::sparse_bin_histogram");
    for (size_t i = 0; i < multi_val_bin_wappers_.size(); ++i) {
      const auto& multi_val_bin_wrapper = multi_val_bin_wappers_[i];
      multi_val_bin_wrapper->ConstructHistograms<USE_INDICES, ORDERED>(
        data_indices, num_data, gradients, hessians, &hist_buf_, hist_data + hist_data_offsets_[i] * 2
      );
    }
  }

  static TrainingShareStates* CreateTrainingShareStates(bool single_precision_hist_buffer);

  virtual ~TrainingShareStates() {}

  void SetUseSubrow(bool is_use_subrow) {
    for (auto& multi_val_bin_wrapper : multi_val_bin_wappers_) {
      multi_val_bin_wrapper->SetUseSubrow(is_use_subrow);
    }
  }

  void SetSubrowCopied(bool is_subrow_copied) {
    for (auto& multi_val_bin_wrapper : multi_val_bin_wappers_) {
      multi_val_bin_wrapper->SetSubrowCopied(is_subrow_copied);
    }
  }

protected:
  std::vector<uint32_t> feature_hist_offsets_;
  uint64_t num_hist_total_bin_ = 0;
  std::vector<size_t> hist_data_offsets_;
  std::vector<std::unique_ptr<MultiValBinWrapper>> multi_val_bin_wappers_;
  std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>> hist_buf_;
};

} // namespace LightGBM

#endif   // LightGBM_TRAIN_SHARE_STATES_H_
