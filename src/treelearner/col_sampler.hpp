/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_TREELEARNER_COL_SAMPLER_HPP_
#define LIGHTGBM_TREELEARNER_COL_SAMPLER_HPP_

#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/random.h>

#include <algorithm>
#include <vector>

namespace LightGBM {
class ColSampler {
 public:
  explicit ColSampler(const Config* config)
      : fraction_bytree_(config->feature_fraction),
        fraction_bynode_(config->feature_fraction_bynode),
        seed_(config->feature_fraction_seed),
        random_(config->feature_fraction_seed) {
  }

  static int GetCnt(size_t total_cnt, double fraction) {
    const int min = std::min(2, static_cast<int>(total_cnt));
    int used_feature_cnt = static_cast<int>(Common::RoundInt(total_cnt * fraction));
    return std::max(used_feature_cnt, min);
  }

  void SetTrainingData(const Dataset* train_data) {
    train_data_ = train_data;
    is_feature_used_.resize(train_data_->num_features(), 1);
    valid_feature_indices_ = train_data->ValidFeatureIndices();
    if (fraction_bytree_ >= 1.0f) {
      need_reset_bytree_ = false;
      used_cnt_bytree_ = static_cast<int>(valid_feature_indices_.size());
    } else {
      need_reset_bytree_ = true;
      used_cnt_bytree_ =
          GetCnt(valid_feature_indices_.size(), fraction_bytree_);
    }
    ResetByTree();
  }

  void SetConfig(const Config* config) {
    fraction_bytree_ = config->feature_fraction;
    fraction_bynode_ = config->feature_fraction_bynode;
    is_feature_used_.resize(train_data_->num_features(), 1);
    // seed is changed
    if (seed_ != config->feature_fraction_seed) {
      seed_ = config->feature_fraction_seed;
      random_ = Random(seed_);
    }
    if (fraction_bytree_ >= 1.0f) {
      need_reset_bytree_ = false;
      used_cnt_bytree_ = static_cast<int>(valid_feature_indices_.size());
    } else {
      need_reset_bytree_ = true;
      used_cnt_bytree_ =
          GetCnt(valid_feature_indices_.size(), fraction_bytree_);
    }
    ResetByTree();
  }

  void ResetByTree() {
    if (need_reset_bytree_) {
      std::memset(is_feature_used_.data(), 0,
                  sizeof(int8_t) * is_feature_used_.size());
      used_feature_indices_ = random_.Sample(
          static_cast<int>(valid_feature_indices_.size()), used_cnt_bytree_);
      int omp_loop_size = static_cast<int>(used_feature_indices_.size());

#pragma omp parallel for schedule(static, 512) if (omp_loop_size >= 1024)
      for (int i = 0; i < omp_loop_size; ++i) {
        int used_feature = valid_feature_indices_[used_feature_indices_[i]];
        int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
        is_feature_used_[inner_feature_index] = 1;
      }
    }
  }

  std::vector<int8_t> GetByNode() {
    if (fraction_bynode_ >= 1.0f) {
      return std::vector<int8_t>(train_data_->num_features(), 1);
    }
    std::vector<int8_t> ret(train_data_->num_features(), 0);
    if (need_reset_bytree_) {
      auto used_feature_cnt = GetCnt(used_feature_indices_.size(), fraction_bynode_);
      auto sampled_indices = random_.Sample(
          static_cast<int>(used_feature_indices_.size()), used_feature_cnt);
      int omp_loop_size = static_cast<int>(sampled_indices.size());
#pragma omp parallel for schedule(static, 512) if (omp_loop_size >= 1024)
      for (int i = 0; i < omp_loop_size; ++i) {
        int used_feature =
            valid_feature_indices_[used_feature_indices_[sampled_indices[i]]];
        int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
        ret[inner_feature_index] = 1;
      }
    } else {
      auto used_feature_cnt =
          GetCnt(valid_feature_indices_.size(), fraction_bynode_);
      auto sampled_indices = random_.Sample(
          static_cast<int>(valid_feature_indices_.size()), used_feature_cnt);
      int omp_loop_size = static_cast<int>(sampled_indices.size());
#pragma omp parallel for schedule(static, 512) if (omp_loop_size >= 1024)
      for (int i = 0; i < omp_loop_size; ++i) {
        int used_feature = valid_feature_indices_[sampled_indices[i]];
        int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
        ret[inner_feature_index] = 1;
      }
    }
    return ret;
  }

  const std::vector<int8_t>& is_feature_used_bytree() const {
    return is_feature_used_;
  }

  void SetIsFeatureUsedByTree(int fid, bool val) {
    is_feature_used_[fid] = val;
  }

 private:
  const Dataset* train_data_;
  double fraction_bytree_;
  double fraction_bynode_;
  bool need_reset_bytree_;
  int used_cnt_bytree_;
  int seed_;
  Random random_;
  std::vector<int8_t> is_feature_used_;
  std::vector<int> used_feature_indices_;
  std::vector<int> valid_feature_indices_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_TREELEARNER_COL_SAMPLER_HPP_
