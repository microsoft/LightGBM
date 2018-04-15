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

  void Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override {
    CHECK(config->bagging_freq > 0 && config->bagging_fraction < 1.0f && config->bagging_fraction > 0.0f);
    CHECK(config->tree_config.feature_fraction < 1.0f && config->tree_config.feature_fraction > 0.0f);
    GBDT::Init(config, train_data, objective_function, training_metrics);

    if (num_init_iteration_ > 0) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        MultiplyScore(cur_tree_id, 1.0f / num_init_iteration_);
      }
    } else {
      CHECK(train_data->metadata().init_score() == nullptr);
    }
    // cannot use RF for multi-class. 
    CHECK(num_tree_per_iteration_ == 1);
    // not shrinkage rate for the RF
    shrinkage_rate_ = 1.0f;
    // only boosting one time
    Boosting();
    if (is_use_subset_ && bag_data_cnt_ < num_data_) {
      size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
      tmp_grad_.resize(total_size);
      tmp_hess_.resize(total_size);
    }
  }

  void ResetConfig(const BoostingConfig* config) override {
    CHECK(config->bagging_freq > 0 && config->bagging_fraction < 1.0f && config->bagging_fraction > 0.0f);
    CHECK(config->tree_config.feature_fraction < 1.0f && config->tree_config.feature_fraction > 0.0f);
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
    CHECK(num_tree_per_iteration_ == 1);
    // only boosting one time
    Boosting();
    if (is_use_subset_ && bag_data_cnt_ < num_data_) {
      size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
      tmp_grad_.resize(total_size);
      tmp_hess_.resize(total_size);
    }
  }

  void Boosting() override {
    if (objective_function_ == nullptr) {
      Log::Fatal("No object function provided");
    }
    std::vector<double> tmp_score(num_tree_per_iteration_ * num_data_, 0.0f);
    objective_function_->
      GetGradients(tmp_score.data(), gradients_.data(), hessians_.data());
  }

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override {
    // bagging logic
    Bagging(iter_);
    if (gradients == nullptr || hessians == nullptr) {
      gradients = gradients_.data();
      hessians = hessians_.data();
    }

    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      std::unique_ptr<Tree> new_tree(new Tree(2));
      if (class_need_train_[cur_tree_id]) {
        size_t bias = static_cast<size_t>(cur_tree_id)* num_data_;

        auto grad = gradients + bias;
        auto hess = hessians + bias;

        // need to copy gradients for bagging subset.
        if (is_use_subset_ && bag_data_cnt_ < num_data_) {
          for (int i = 0; i < bag_data_cnt_; ++i) {
            tmp_grad_[bias + i] = grad[bag_data_indices_[i]];
            tmp_hess_[bias + i] = hess[bag_data_indices_[i]];
          }
          grad = tmp_grad_.data() + bias;
          hess = tmp_hess_.data() + bias;
        }

        new_tree.reset(tree_learner_->Train(grad, hess, is_constant_hessian_,
                       forced_splits_json_));
      }

      if (new_tree->num_leaves() > 1) {
        // update score
        MultiplyScore(cur_tree_id, (iter_ + num_init_iteration_));
        ConvertTreeOutput(new_tree.get());
        UpdateScore(new_tree.get(), cur_tree_id);
        MultiplyScore(cur_tree_id, 1.0 / (iter_ + num_init_iteration_ + 1));
      } else {
        // only add default score one-time
        if (!class_need_train_[cur_tree_id] && models_.size() < static_cast<size_t>(num_tree_per_iteration_)) {
          double output = class_default_output_[cur_tree_id];
          objective_function_->ConvertOutput(&output, &output);
          new_tree->AsConstantTree(output);
          train_score_updater_->AddScore(output, cur_tree_id);
          for (auto& score_updater : valid_score_updater_) {
            score_updater->AddScore(output, cur_tree_id);
          }
        }
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

  void ConvertTreeOutput(Tree* tree) {
    tree->Shrinkage(1.0f);
    for (int i = 0; i < tree->num_leaves(); ++i) {
      double output = tree->LeafOutput(i);
      objective_function_->ConvertOutput(&output, &output);
      tree->SetLeafOutput(i, output);
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

};

}  // namespace LightGBM
#endif   // LIGHTGBM_BOOSTING_RF_H_
