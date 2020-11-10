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

struct TrainingShareStates {
  int num_threads = 0;
  bool is_colwise = true;
  bool is_two_rowwise = false;
  bool is_use_subcol = false;
  bool is_use_subcol_sparse = false;
  bool is_use_subrow = false;
  bool is_subrow_copied = false;
  bool is_constant_hessian = true;
  const data_size_t* bagging_use_indices;
  data_size_t bagging_indices_cnt;
  std::unique_ptr<MultiValBin> multi_val_bin;
  std::unique_ptr<MultiValBin> multi_val_bin_subset;
  std::unique_ptr<MultiValBin> multi_val_bin_sparse;
  std::unique_ptr<MultiValBin> multi_val_bin_sparse_subset;
  std::vector<uint32_t> hist_move_src;
  std::vector<uint32_t> hist_move_dest;
  std::vector<uint32_t> hist_move_size;
  std::vector<uint32_t> hist_move_src_sparse;
  std::vector<uint32_t> hist_move_dest_sparse;
  std::vector<uint32_t> hist_move_size_sparse;

  int num_hist_total_bin() { return num_hist_total_bin_; }

  const std::vector<uint32_t>& feature_hist_offsets() { return feature_hist_offsets_; }

  virtual void SetMultiValBin(MultiValBin* bin, data_size_t num_data) = 0;

  virtual void SetSparseMultiValBin(MultiValBin* /*bin*/, data_size_t /*num_data*/) {}

  virtual void HistMove() = 0;

  virtual void SparseHistMove() {}

  virtual void HistMerge() = 0;

  virtual void SparseHistMerge() {}

  virtual void ResizeHistBuf(hist_t* hist_data) = 0;

  virtual void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    bool use_indices, bool ordered) = 0;

  virtual void SparseConstructHistogramsForBlock(const MultiValBin* /*sub_multi_val_bin*/,
    data_size_t /*start*/, data_size_t /*end*/, const data_size_t* /*data_indices*/,
    const score_t* /*gradients*/, const score_t* /*hessians*/, int /*block_id*/,
    bool /*use_indices*/, bool /*ordered*/) {}

  virtual void CalcBinOffsets(const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    std::vector<uint32_t>* offsets, std::vector<uint32_t>* offsets2,
    bool is_col_wise, bool is_two_row_wise, bool is_sparse_row_wise) {
    offsets->clear();
    offsets2->clear();
    feature_hist_offsets_.clear();
    hist_start_pos_ = 0;
    sparse_hist_start_pos_ = 0;
    if (is_two_row_wise) {
      is_two_row_wise = false;
      //int multi_val_group = -1;
      for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
        if (feature_groups[group]->is_multi_val_) {
          if (!feature_groups[group]->is_dense_multi_val_ && feature_groups.size() > 1) {
            is_two_row_wise = true;
          }
          //multi_val_group = group;
        }
      }
      /*if (multi_val_group > 0 && !feature_groups[multi_val_group]->is_dense_multi_val_ &&
        feature_groups.size() == 1) {
        CHECK(is_sparse_row_wise == true);
      }
      if (multi_val_group > 0 && feature_groups[multi_val_group]->is_dense_multi_val_) {
        CHECK(is_sparse_row_wise == false);
      }*/
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
            Log::Warning("hist_cur_num_bin = %d, feature_group->bin_offsets_[i] = %d", hist_cur_num_bin, feature_group->bin_offsets_[i]);
          }
          hist_cur_num_bin += feature_group->bin_offsets_.back();
        }
      }
      feature_hist_offsets_.push_back(hist_cur_num_bin);
      num_hist_total_bin_ = feature_hist_offsets_.back();
    } else if (!is_two_row_wise) {
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
      num_hist_total_bin_ = feature_hist_offsets_.back();
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
      Log::Warning("multi_val_group_id = %d", multi_val_group_id);
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
        for (int i = 0; i < static_cast<int>(feature_hist_offsets1.size()) - 1; ++i) {
          feature_hist_offsets_.push_back(feature_hist_offsets1[i]);
        }
        for (int i = 0; i < static_cast<int>(feature_hist_offsets2.size()); ++i) {
          feature_hist_offsets_.push_back(feature_hist_offsets2[i] + feature_hist_offsets1.back());
        }
        sparse_hist_start_pos_ = 0;
        hist_start_pos_ = feature_hist_offsets1.back();
      } else {
        for (int i = 0; i < static_cast<int>(feature_hist_offsets2.size()) - 1; ++i) {
          feature_hist_offsets_.push_back(feature_hist_offsets2[i]);
        }
        for (int i = 0; i < static_cast<int>(feature_hist_offsets1.size()); ++i) {
          feature_hist_offsets_.push_back(feature_hist_offsets1[i] + feature_hist_offsets2.back());
        }
        hist_start_pos_ = 0;
        sparse_hist_start_pos_ = feature_hist_offsets2.back();
      }
      num_hist_total_bin_ = feature_hist_offsets1.back() + feature_hist_offsets2.back();
    }
  }

  virtual void InitTrain() {
    const auto cur_multi_val_bin = (is_use_subcol || is_use_subrow)
          ? multi_val_bin_subset.get()
          : multi_val_bin.get();
    if (cur_multi_val_bin != nullptr) {
      num_bin_ = cur_multi_val_bin->num_bin();
      num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
      min_block_size_ = std::min<int>(static_cast<int>(0.3f * num_bin_ /
        cur_multi_val_bin->num_element_per_row()) + 1, 1024);
    }
    const auto cur_multi_val_bin_sparse = (is_use_subcol_sparse || is_use_subrow)
          ? multi_val_bin_sparse_subset.get()
          : multi_val_bin_sparse.get();
    if (cur_multi_val_bin_sparse != nullptr) {
      sparse_num_bin_ = cur_multi_val_bin_sparse->num_bin();
      sparse_num_bin_aligned_ = (sparse_num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
      sparse_min_block_size_ = std::min<int>(static_cast<int>(0.3f * sparse_num_bin_ /
        cur_multi_val_bin_sparse->num_element_per_row()) + 1, 1024);
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructHistograms(const data_size_t* data_indices,
                          data_size_t num_data,
                          const score_t* gradients,
                          const score_t* hessians,
                          hist_t* hist_data) {
    global_timer.Start("Dataset::sparse_bin_histogram");
    const auto cur_multi_val_bin = (is_use_subcol || is_use_subrow)
          ? multi_val_bin_subset.get()
          : multi_val_bin.get();
    if (cur_multi_val_bin != nullptr) {
      n_data_block_ = 1;
      data_block_size_ = num_data;
      Threading::BlockInfo<data_size_t>(num_threads, num_data, min_block_size_,
                                        max_block_size_, &n_data_block_, &data_block_size_);
      ResizeHistBuf(hist_data);
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static) num_threads(num_threads)
      for (int block_id = 0; block_id < n_data_block_; ++block_id) {
        OMP_LOOP_EX_BEGIN();
        data_size_t start = block_id * data_block_size_;
        data_size_t end = std::min(start + data_block_size_, num_data);
        if (USE_INDICES) {
          if (ORDERED) {
            ConstructHistogramsForBlock(
              cur_multi_val_bin, start, end, data_indices, gradients, hessians,
              block_id, true, true
            );
          } else {
            ConstructHistogramsForBlock(
              cur_multi_val_bin, start, end, data_indices, gradients, hessians,
              block_id, true, false
            );
          }
        } else {
          ConstructHistogramsForBlock(
              cur_multi_val_bin, start, end, data_indices, gradients, hessians,
              block_id, false, false
            );
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      global_timer.Stop("Dataset::sparse_bin_histogram");

      global_timer.Start("Dataset::sparse_bin_histogram_merge");
      HistMerge();
      global_timer.Stop("Dataset::sparse_bin_histogram_merge");
      global_timer.Start("Dataset::sparse_bin_histogram_move");
      HistMove();
      global_timer.Stop("Dataset::sparse_bin_histogram_move");
    }

    const auto cur_multi_val_bin_sparse = (is_use_subcol_sparse || is_use_subrow)
          ? multi_val_bin_sparse_subset.get()
          : multi_val_bin_sparse.get();
    if (cur_multi_val_bin_sparse != nullptr) {
      n_data_block_ = 1;
      data_block_size_ = num_data;
      Threading::BlockInfo<data_size_t>(num_threads, num_data, sparse_min_block_size_,
                                        max_block_size_, &n_data_block_, &data_block_size_);
      ResizeHistBuf(hist_data);
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static) num_threads(num_threads)
      for (int block_id = 0; block_id < n_data_block_; ++block_id) {
        OMP_LOOP_EX_BEGIN();
        data_size_t start = block_id * data_block_size_;
        data_size_t end = std::min(start + data_block_size_, num_data);
        if (USE_INDICES) {
          if (ORDERED) {
            SparseConstructHistogramsForBlock(
              cur_multi_val_bin_sparse, start, end, data_indices, gradients, hessians,
              block_id, true, true
            );
          } else {
            SparseConstructHistogramsForBlock(
              cur_multi_val_bin_sparse, start, end, data_indices, gradients, hessians,
              block_id, true, false
            );
          }
        } else {
          SparseConstructHistogramsForBlock(
              cur_multi_val_bin_sparse, start, end, data_indices, gradients, hessians,
              block_id, false, false
            );
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      global_timer.Stop("Dataset::sparse_bin_histogram");

      global_timer.Start("Dataset::sparse_bin_histogram_merge");
      SparseHistMerge();
      global_timer.Stop("Dataset::sparse_bin_histogram_merge");
      global_timer.Start("Dataset::sparse_bin_histogram_move");
      SparseHistMove();
      global_timer.Stop("Dataset::sparse_bin_histogram_move");
    }
  }

  static TrainingShareStates* CreateTrainingShareStates(bool single_precision_hist_buffer);

  virtual ~TrainingShareStates() {}

protected:
  size_t kHistBufferEntrySize;
  int num_bin_aligned_;
  int num_bin_;
  int max_block_size_;
  int min_block_size_;
  int sparse_min_block_size_;
  int n_data_block_;
  int data_block_size_;
  hist_t* origin_hist_data_ = nullptr;
  int sparse_num_bin_aligned_;
  int sparse_num_bin_;
  hist_t* sparse_origin_hist_data_ = nullptr;
  std::vector<uint32_t> feature_hist_offsets_;
  int hist_start_pos_;
  int sparse_hist_start_pos_;
  int num_hist_total_bin_;
};

struct TrainingShareStatesFloat : public TrainingShareStates {
  std::vector<float, Common::AlignmentAllocator<float, kAlignedSize>> hist_buf;
  std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>> temp_buf;

  void SetMultiValBin(MultiValBin* bin, data_size_t num_data) override {
    num_threads = OMP_NUM_THREADS();
    kHistBufferEntrySize = 2 * sizeof(float);
    max_block_size_ = 100000;
    if (bin == nullptr) {
      return;
    }
    int num_blocks = (num_data + max_block_size_ - 1) / max_block_size_;
    num_blocks = std::max(num_threads, num_blocks);
    multi_val_bin.reset(bin);
    num_bin_aligned_ =
        (bin->num_bin() + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
    size_t new_size = static_cast<size_t>(num_bin_aligned_) * 2 * num_blocks;
    if (new_size > hist_buf.size()) {
      hist_buf.resize(static_cast<size_t>(num_bin_aligned_) * 2 * num_blocks);
    }
    if (is_use_subcol) {
      temp_buf.resize(static_cast<size_t>(num_bin_aligned_) * 2);
    }
  }

  void ResizeHistBuf(hist_t* hist_data) override {
    origin_hist_data_ = hist_data;
    size_t new_buf_size = static_cast<size_t>(n_data_block_) * static_cast<size_t>(num_bin_aligned_) * 2;
    if (hist_buf.size() < new_buf_size) {
      hist_buf.resize(new_buf_size);
    }
    if (temp_buf.size() < static_cast<size_t>(num_bin_aligned_) * 2 && is_use_subcol) {
      temp_buf.resize(static_cast<size_t>(num_bin_aligned_) * 2);
    }
  }

  void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    bool use_indices, bool ordered) override {
    float* data_ptr = hist_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * (block_id);
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kHistBufferEntrySize);
    if (use_indices) {
      if (ordered) {
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

  void HistMerge() override {
    int n_bin_block = 1;
    int bin_block_size = num_bin_;
    Threading::BlockInfo<data_size_t>(num_threads, num_bin_, 512, &n_bin_block,
                                    &bin_block_size);
    hist_t* dst = origin_hist_data_;
    if (is_use_subcol) {
      dst = temp_buf.data();
    }
    std::memset(reinterpret_cast<void*>(dst), 0, num_bin_ * kHistEntrySize);
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for (int t = 0; t < n_bin_block; ++t) {
      const int start = t * bin_block_size;
      const int end = std::min(start + bin_block_size, num_bin_);
      for (int tid = 0; tid < n_data_block_; ++tid) {
        auto src_ptr = hist_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * (tid);
        for (int i = start * 2; i < end * 2; ++i) {
          dst[i] += src_ptr[i];
        }
      }
    }
  }

  void HistMove() override {
    if (!is_use_subcol) {
      return;
    }
    hist_t* src = temp_buf.data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(hist_move_src.size()); ++i) {
      std::copy_n(src + hist_move_src[i], hist_move_size[i],
                  origin_hist_data_ + hist_move_dest[i]);
    }
  }
};

struct TrainingShareStatesFloatWithBuffer : public TrainingShareStatesFloat {

  void SetMultiValBin(MultiValBin* bin, data_size_t /*num_data*/) override {
    num_threads = OMP_NUM_THREADS();
    kHistBufferEntrySize = 2 * sizeof(float);
    max_block_size_ = 100000;
    if (bin == nullptr) {
      return;
    }
    multi_val_bin.reset(bin);
    num_bin_aligned_ =
        (bin->num_bin() + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
    const size_t thread_buf_size = static_cast<size_t>(num_bin_aligned_) * num_threads * 2;
    temp_buf.resize(thread_buf_size, 0.0f);
    hist_buf.resize(thread_buf_size, 0.0f);
  }

  void ResizeHistBuf(hist_t* hist_data) override {
    origin_hist_data_ = hist_data;
    const size_t new_thread_buf_size = static_cast<size_t>(num_bin_aligned_) * num_threads * 2;
    if (new_thread_buf_size > temp_buf.size()) {
      temp_buf.resize(new_thread_buf_size, 0.0f);
    }
    if (new_thread_buf_size > hist_buf.size()) {
      hist_buf.resize(new_thread_buf_size, 0.0f);
    }
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < static_cast<int>(temp_buf.size()); ++i) {
      temp_buf[i] = 0.0f;
    }
    if (!is_use_subcol) {
      #pragma omp parallel for schedule(static) num_threads(num_threads)
      for (int i = 0; i < num_bin_ * 2; ++i) {
        hist_data[i] = 0.0f;
      }
    }
  }

  void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int /*block_id*/,
    bool use_indices, bool ordered) override {
    int thread_id = omp_get_thread_num();
    float* data_ptr = hist_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * thread_id;
    const int num_bin = sub_multi_val_bin->num_bin();
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin * kHistBufferEntrySize);
    if (use_indices) {
      if (ordered) {
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
    double* thread_buf_ptr = origin_hist_data_;
    if (thread_id == 0) {
      if (is_use_subcol) {
        thread_buf_ptr = temp_buf.data();
      }
    } else {
      thread_buf_ptr = temp_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * (thread_id);
    }
    for (int i = 0; i < 2 * num_bin; ++i) {
      thread_buf_ptr[i] += data_ptr[i];
    }
  }

  void HistMerge() override {
    int n_bin_block = 1;
    int bin_block_size = num_bin_;
    Threading::BlockInfo<data_size_t>(num_threads, num_bin_, 512, &n_bin_block,
                                    &bin_block_size);
    hist_t* dst = origin_hist_data_;
    if (is_use_subcol) {
      dst = temp_buf.data();
    }
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for (int t = 0; t < n_bin_block; ++t) {
      const int start = t * bin_block_size;
      const int end = std::min(start + bin_block_size, num_bin_);
      for (int tid = 1; tid < num_threads; ++tid) {
        auto src_ptr = temp_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * (tid);
        for (int i = start * 2; i < end * 2; ++i) {
          dst[i] += src_ptr[i];
        }
      }
    }
  }

  void HistMove() override {
    if (!is_use_subcol) {
      return;
    }
    hist_t* src = temp_buf.data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(hist_move_src.size()); ++i) {
      std::copy_n(src + hist_move_src[i], hist_move_size[i],
                  origin_hist_data_ + hist_move_dest[i]);
    }
  }
};

struct TrainingShareStatesDouble : public TrainingShareStates {
  std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>> hist_buf;

  void SetMultiValBin(MultiValBin* bin, data_size_t num_data) override {
    num_threads = OMP_NUM_THREADS();
    kHistBufferEntrySize = 2 * sizeof(hist_t);
    max_block_size_ = num_data;
    if (bin == nullptr) {
      return;
    }
    multi_val_bin.reset(bin);
    num_bin_aligned_ =
        (bin->num_bin() + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
    size_t new_size = static_cast<size_t>(num_bin_aligned_) * 2 * num_threads;
    if (new_size > hist_buf.size()) {
      hist_buf.resize(new_size);
    }
  }

  void SetSparseMultiValBin(MultiValBin* bin, data_size_t /*num_data*/) override {
    if (bin == nullptr) {
      return;
    }
    multi_val_bin_sparse.reset(bin);
    sparse_num_bin_aligned_ =
        (bin->num_bin() + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
    size_t new_size = static_cast<size_t>(sparse_num_bin_aligned_) * 2 * num_threads;
    if (new_size > hist_buf.size()) {
      hist_buf.resize(new_size);
    }
  }

  void ResizeHistBuf(hist_t* hist_data) override {
    origin_hist_data_ = hist_data + hist_start_pos_ * 2;
    sparse_origin_hist_data_ = hist_data + sparse_hist_start_pos_ * 2;
    size_t new_buf_size = static_cast<size_t>(n_data_block_) * static_cast<size_t>(num_bin_aligned_) * 2;
    if (hist_buf.size() < new_buf_size) {
      hist_buf.resize(new_buf_size);
    }
    new_buf_size = static_cast<size_t>(n_data_block_) * static_cast<size_t>(sparse_num_bin_aligned_) * 2;
    if (hist_buf.size() < new_buf_size) {
      hist_buf.resize(new_buf_size);
    }
  }

  void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    bool use_indices, bool ordered) override {
    hist_t* data_ptr = origin_hist_data_;
    if (block_id == 0) {
      if (is_use_subcol) {
        data_ptr = hist_buf.data() + hist_buf.size() - 2 * static_cast<size_t>(num_bin_aligned_);
      }
    } else {
      data_ptr = hist_buf.data() +
        static_cast<size_t>(num_bin_aligned_) * (block_id - 1) * 2;
    }
    const int num_bin = sub_multi_val_bin->num_bin();
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin * kHistBufferEntrySize);
    if (use_indices) {
      if (ordered) {
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

  void SparseConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    bool use_indices, bool ordered) override {
    hist_t* data_ptr = sparse_origin_hist_data_;
    if (block_id == 0) {
      if (is_use_subcol_sparse) {
        data_ptr = hist_buf.data() + hist_buf.size() - 2 * static_cast<size_t>(sparse_num_bin_aligned_);
      }
    } else {
      data_ptr = hist_buf.data() +
        static_cast<size_t>(sparse_num_bin_aligned_) * (block_id - 1) * 2;
    }
    const int num_bin = sub_multi_val_bin->num_bin();
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin * kHistBufferEntrySize);
    if (use_indices) {
      if (ordered) {
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

  void HistMerge() override {
    int n_bin_block = 1;
    int bin_block_size = num_bin_;
    Threading::BlockInfo<data_size_t>(num_threads, num_bin_, 512, &n_bin_block,
                                    &bin_block_size);
    hist_t* dst = origin_hist_data_;
    if (is_use_subcol) {
      dst = hist_buf.data() + hist_buf.size() - 2 * static_cast<size_t>(num_bin_aligned_);
    }
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for (int t = 0; t < n_bin_block; ++t) {
      const int start = t * bin_block_size;
      const int end = std::min(start + bin_block_size, num_bin_);
      for (int tid = 1; tid < n_data_block_; ++tid) {
        auto src_ptr = hist_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * (tid - 1);
        for (int i = start * 2; i < end * 2; ++i) {
          dst[i] += src_ptr[i];
        }
      }
    }
  }

  void SparseHistMerge() override {
    int n_bin_block = 1;
    int bin_block_size = sparse_num_bin_;
    Threading::BlockInfo<data_size_t>(num_threads, sparse_num_bin_, 512, &n_bin_block,
                                    &bin_block_size);
    hist_t* dst = sparse_origin_hist_data_;
    if (is_use_subcol_sparse) {
      dst = hist_buf.data() + hist_buf.size() - 2 * static_cast<size_t>(sparse_num_bin_aligned_);
    }
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for (int t = 0; t < n_bin_block; ++t) {
      const int start = t * bin_block_size;
      const int end = std::min(start + bin_block_size, sparse_num_bin_);
      for (int tid = 1; tid < n_data_block_; ++tid) {
        auto src_ptr = hist_buf.data() + static_cast<size_t>(sparse_num_bin_aligned_) * 2 * (tid - 1);
        for (int i = start * 2; i < end * 2; ++i) {
          dst[i] += src_ptr[i];
        }
      }
    }
  }

  void HistMove() override {
    if (!is_use_subcol) {
      return;
    }
    hist_t* src = hist_buf.data() + hist_buf.size() - 2 * static_cast<size_t>(num_bin_aligned_);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(hist_move_src.size()); ++i) {
      std::copy_n(src + hist_move_src[i], hist_move_size[i],
                  origin_hist_data_ + hist_move_dest[i]);
    }
  }

  void SparseHistMove() override {
    if (!is_use_subcol_sparse) {
      return;
    }
    hist_t* src = hist_buf.data() + hist_buf.size() - 2 * static_cast<size_t>(sparse_num_bin_aligned_);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(hist_move_src_sparse.size()); ++i) {
      std::copy_n(src + hist_move_src_sparse[i], hist_move_size_sparse[i],
                  sparse_origin_hist_data_ + hist_move_dest_sparse[i]);
    }
  }
};

} // namespace LightGBM

#endif   // LightGBM_TRAIN_SHARE_STATES_H_
