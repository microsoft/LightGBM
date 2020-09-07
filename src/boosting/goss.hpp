/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_BOOSTING_GOSS_H_
#define LIGHTGBM_BOOSTING_GOSS_H_

#include <LightGBM/boosting.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>

#include "gbdt.h"
#include "score_updater.hpp"

namespace LightGBM {

class GOSS: public GBDT {
 public:
  /*!
  * \brief Constructor
  */
  GOSS() : GBDT() {
  }

  ~GOSS() {
  }

  void Init(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override {
    GBDT::Init(config, train_data, objective_function, training_metrics);
    ResetGoss();
    if (objective_function_ == nullptr) {
      // use customized objective function
      size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
      gradients_.resize(total_size, 0.0f);
      hessians_.resize(total_size, 0.0f);
    }
  }

  void ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
                         const std::vector<const Metric*>& training_metrics) override {
    GBDT::ResetTrainingData(train_data, objective_function, training_metrics);
    ResetGoss();
  }

  void ResetConfig(const Config* config) override {
    GBDT::ResetConfig(config);
    ResetGoss();
  }

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override {
    if (gradients != nullptr) {
      // use customized objective function
      CHECK(hessians != nullptr && objective_function_ == nullptr);
      int64_t total_size = static_cast<int64_t>(num_data_) * num_tree_per_iteration_;
      #pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < total_size; ++i) {
        gradients_[i] = gradients[i];
        hessians_[i] = hessians[i];
      }
      return GBDT::TrainOneIter(gradients_.data(), hessians_.data());
    } else {
      CHECK(hessians == nullptr);
      return GBDT::TrainOneIter(nullptr, nullptr);
    }
  }

  void ResetGoss() {
    CHECK_LE(config_->top_rate + config_->other_rate, 1.0f);
    CHECK(config_->top_rate > 0.0f && config_->other_rate > 0.0f);
    if (config_->bagging_freq > 0 && config_->bagging_fraction != 1.0f) {
      Log::Fatal("Cannot use bagging in GOSS");
    }
    Log::Info("Using GOSS");
    balanced_bagging_ = false;
    bag_data_indices_.resize(num_data_);
    bagging_runner_.ReSize(num_data_);
    bagging_rands_.clear();
    for (int i = 0;
         i < (num_data_ + bagging_rand_block_ - 1) / bagging_rand_block_; ++i) {
      bagging_rands_.emplace_back(config_->bagging_seed + i);
    }
    is_use_subset_ = false;
    if (config_->top_rate + config_->other_rate <= 0.5) {
      auto bag_data_cnt = static_cast<data_size_t>((config_->top_rate + config_->other_rate) * num_data_);
      bag_data_cnt = std::max(1, bag_data_cnt);
      tmp_subset_.reset(new Dataset(bag_data_cnt));
      tmp_subset_->CopyFeatureMapperFrom(train_data_);
      is_use_subset_ = true;
    }
    // flag to not bagging first
    bag_data_cnt_ = num_data_;
  }

  data_size_t BaggingHelper(data_size_t start, data_size_t cnt, data_size_t* buffer) override {
    if (cnt <= 0) {
      return 0;
    }
    std::vector<score_t> tmp_gradients(cnt, 0.0f);
    for (data_size_t i = 0; i < cnt; ++i) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + start + i;
        tmp_gradients[i] += std::fabs(gradients_[idx] * hessians_[idx]);
      }
    }
    data_size_t top_k = static_cast<data_size_t>(cnt * config_->top_rate);
    data_size_t other_k = static_cast<data_size_t>(cnt * config_->other_rate);
    top_k = std::max(1, top_k);
    ArrayArgs<score_t>::ArgMaxAtK(&tmp_gradients, 0, static_cast<int>(tmp_gradients.size()), top_k - 1);
    score_t threshold = tmp_gradients[top_k - 1];

    score_t multiply = static_cast<score_t>(cnt - top_k) / other_k;
    data_size_t cur_left_cnt = 0;
    data_size_t cur_right_pos = cnt;
    data_size_t big_weight_cnt = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
      auto cur_idx = start + i;
      score_t grad = 0.0f;
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + cur_idx;
        grad += std::fabs(gradients_[idx] * hessians_[idx]);
      }
      if (grad >= threshold) {
        buffer[cur_left_cnt++] = cur_idx;
        ++big_weight_cnt;
      } else {
        data_size_t sampled = cur_left_cnt - big_weight_cnt;
        data_size_t rest_need = other_k - sampled;
        data_size_t rest_all = (cnt - i) - (top_k - big_weight_cnt);
        double prob = (rest_need) / static_cast<double>(rest_all);
        if (bagging_rands_[cur_idx / bagging_rand_block_].NextFloat() < prob) {
          buffer[cur_left_cnt++] = cur_idx;
          for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
            size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + cur_idx;
            gradients_[idx] *= multiply;
            hessians_[idx] *= multiply;
          }
        } else {
          buffer[--cur_right_pos] = cur_idx;
        }
      }
    }
    return cur_left_cnt;
  }

  void Bagging(int iter) override {
    bag_data_cnt_ = num_data_;
    // not subsample for first iterations
    if (iter < static_cast<int>(1.0f / config_->learning_rate)) { return; }
    auto left_cnt = bagging_runner_.Run<true>(
        num_data_,
        [=](int, data_size_t cur_start, data_size_t cur_cnt, data_size_t* left,
            data_size_t*) {
          data_size_t cur_left_count = 0;
          cur_left_count = BaggingHelper(cur_start, cur_cnt, left);
          return cur_left_count;
        },
        bag_data_indices_.data());
    bag_data_cnt_ = left_cnt;
    // set bagging data to tree learner
    if (!is_use_subset_) {
      tree_learner_->SetBaggingData(nullptr, bag_data_indices_.data(), bag_data_cnt_);
    } else {
      // get subset
      tmp_subset_->ReSize(bag_data_cnt_);
      tmp_subset_->CopySubrow(train_data_, bag_data_indices_.data(),
                              bag_data_cnt_, false);
      tree_learner_->SetBaggingData(tmp_subset_.get(), bag_data_indices_.data(),
                                    bag_data_cnt_);
    }
  }

 protected:
  bool GetIsConstHessian(const ObjectiveFunction*) override {
    return false;
  }
};

}  // namespace LightGBM
#endif   // LIGHTGBM_BOOSTING_GOSS_H_
