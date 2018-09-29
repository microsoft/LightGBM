#ifndef LIGHTGBM_BOOSTING_RF_H_
#define LIGHTGBM_BOOSTING_RF_H_

#include <LightGBM/boosting.h>
#include <LightGBM/metric.h>
#include "score_updater.hpp"
#include "gbdt.h"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>

namespace LightGBM {
/*!
* \brief Rondom Forest implementation
*/
class RF: public GBDT {
public:

  RF() : GBDT() { 
    average_output_ = true;
  }

  ~RF() {}

  void Init(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override {
    CHECK(config->bagging_freq > 0 && config->bagging_fraction < 1.0f && config->bagging_fraction > 0.0f);
    CHECK(config->feature_fraction < 1.0f && config->feature_fraction > 0.0f);
    GBDT::Init(config, train_data, objective_function, training_metrics);

    if (num_init_iteration_ > 0) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        MultiplyScore(cur_tree_id, 1.0f / num_init_iteration_);
      }
    } else {
      CHECK(train_data->metadata().init_score() == nullptr);
    }
    // cannot use RF for multi-class. 
    CHECK(num_tree_per_iteration_ == num_class_);
    // not shrinkage rate for the RF
    shrinkage_rate_ = 1.0f;
    // only boosting one time
    GetRFTargets(train_data);
    if (is_use_subset_ && bag_data_cnt_ < num_data_) {
      tmp_grad_.resize(num_data_);
      tmp_hess_.resize(num_data_);
    }
    tmp_score_.resize(num_data_, 0.0);
  }

  void ResetConfig(const Config* config) override {
    CHECK(config->bagging_freq > 0 && config->bagging_fraction < 1.0f && config->bagging_fraction > 0.0f);
    CHECK(config->feature_fraction < 1.0f && config->feature_fraction > 0.0f);
    GBDT::ResetConfig(config);
    // not shrinkage rate for the RF
    shrinkage_rate_ = 1.0f;
  }

  void ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
                         const std::vector<const Metric*>& training_metrics) override {
    GBDT::ResetTrainingData(train_data, objective_function, training_metrics);
    if (iter_ + num_init_iteration_ > 0) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        train_score_updater_->MultiplyScore(1.0f / (iter_ + num_init_iteration_), cur_tree_id);
      }
    }
    // cannot use RF for multi-class.
    CHECK(num_tree_per_iteration_ == num_class_);
    // only boosting one time
    GetRFTargets(train_data);
    if (is_use_subset_ && bag_data_cnt_ < num_data_) {
      tmp_grad_.resize(num_data_);
      tmp_hess_.resize(num_data_);
    }
    tmp_score_.resize(num_data_, 0.0);
  }

  void GetRFTargets(const Dataset* train_data) {
    auto label_ptr = train_data->metadata().label();
    std::fill(hessians_.begin(), hessians_.end(), 1.0f);
    if (num_tree_per_iteration_ == 1) {
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static,1)
      for (data_size_t i = 0; i < train_data->num_data(); ++i) {
        OMP_LOOP_EX_BEGIN();
        score_t label = label_ptr[i];
        gradients_[i] = static_cast<score_t>(-label);
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    }
    else {
      std::fill(gradients_.begin(), gradients_.end(), 0.0f);
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static,1)
      for (data_size_t i = 0; i < train_data->num_data(); ++i) {
        OMP_LOOP_EX_BEGIN();
        score_t label = label_ptr[i];
        gradients_[i + static_cast<int>(label) * num_data_] = -1.0f;
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    }
  }

  void Boosting() override {

  }

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override {
    // bagging logic
    Bagging(iter_);
    CHECK(gradients == nullptr);
    CHECK(hessians == nullptr);

    gradients = gradients_.data();
    hessians = hessians_.data();
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      std::unique_ptr<Tree> new_tree(new Tree(2));
      size_t bias = static_cast<size_t>(cur_tree_id)* num_data_;
      auto grad = gradients + bias;
      auto hess = hessians + bias;

      // need to copy gradients for bagging subset.
      if (is_use_subset_ && bag_data_cnt_ < num_data_) {
        for (int i = 0; i < bag_data_cnt_; ++i) {
          tmp_grad_[i] = grad[bag_data_indices_[i]];
          tmp_hess_[i] = hess[bag_data_indices_[i]];
        }
        grad = tmp_grad_.data();
        hess = tmp_hess_.data();
      }
      new_tree.reset(tree_learner_->Train(grad, hess, is_constant_hessian_,
        forced_splits_json_));
      if (new_tree->num_leaves() > 1) {
        tree_learner_->RenewTreeOutput(new_tree.get(), objective_function_, tmp_score_.data(),
          num_data_, bag_data_indices_.data(), bag_data_cnt_);
        // update score
        MultiplyScore(cur_tree_id, (iter_ + num_init_iteration_));
        UpdateScore(new_tree.get(), cur_tree_id);
        MultiplyScore(cur_tree_id, 1.0 / (iter_ + num_init_iteration_ + 1));
      }
      // add model
      models_.push_back(std::move(new_tree));
    }
    ++iter_;
    return false;
  }

  void RollbackOneIter() override {
    if (iter_ <= 0) { return; }
    int cur_iter = iter_ + num_init_iteration_ - 1;
    // reset score
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      auto curr_tree = cur_iter * num_tree_per_iteration_ + cur_tree_id;
      models_[curr_tree]->Shrinkage(-1.0);
      MultiplyScore(cur_tree_id, (iter_ + num_init_iteration_));
      train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
      for (auto& score_updater : valid_score_updater_) {
        score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
      }
      MultiplyScore(cur_tree_id, 1.0f / (iter_ + num_init_iteration_ - 1));
    }
    // remove model
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      models_.pop_back();
    }
    --iter_;
  }

  void MultiplyScore(const int cur_tree_id, double val) {
    train_score_updater_->MultiplyScore(val, cur_tree_id);
    for (auto& score_updater : valid_score_updater_) {
      score_updater->MultiplyScore(val, cur_tree_id);
    }
  }

  void AddValidDataset(const Dataset* valid_data,
                       const std::vector<const Metric*>& valid_metrics) override {
    GBDT::AddValidDataset(valid_data, valid_metrics);
    if (iter_ + num_init_iteration_ > 0) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        valid_score_updater_.back()->MultiplyScore(1.0f / (iter_ + num_init_iteration_), cur_tree_id);
      }
    }
  }

  bool NeedAccuratePrediction() const override {
    // No early stopping for prediction
    return true;
  };

  std::vector<double> EvalOneMetric(const Metric* metric, const double* score) const override {
    return metric->Eval(score, nullptr);
  }

private:

  std::vector<score_t> tmp_grad_;
  std::vector<score_t> tmp_hess_;
  std::vector<double> tmp_score_;

};

}  // namespace LightGBM
#endif   // LIGHTGBM_BOOSTING_RF_H_
