#include "gbdt.h"
#include "dart.h"

#include <LightGBM/utils/common.h>

#include <LightGBM/feature.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>

#include <ctime>

#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <utility>

namespace LightGBM {

DART::DART(){
}

DART::~DART(){
}

void DART::Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* object_function,
     const std::vector<const Metric*>& training_metrics) {
  GBDT::Init(config, train_data, object_function, training_metrics);
  gbdt_config_ = dynamic_cast<const GBDTConfig*>(config);
  drop_rate_ = gbdt_config_->drop_rate;
  random_for_drop_ = Random(gbdt_config_->dropping_seed);
}

bool DART::TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) {
  // boosting first
  if (gradient == nullptr || hessian == nullptr) {
    Boosting();
    gradient = gradients_;
    hessian = hessians_;
  }

  for (int curr_class = 0; curr_class < num_class_; ++curr_class){
    // bagging logic
    Bagging(iter_, curr_class);
    // train a new tree
    Tree * new_tree = tree_learner_[curr_class]->Train(gradient + curr_class * num_data_, hessian+ curr_class * num_data_);
    // if cannot learn a new tree, then stop
    if (new_tree->num_leaves() <= 1) {
      Log::Info("Can't training anymore, there isn't any leaf meets split requirements.");
      return true;
    }
    // add model
    models_.push_back(new_tree);
  }

  // select dropping trees and normalize once in one iteration
  double new_tree_shrinkage_rate = SelectDroppingTreesAndNormalize();
  size_t len = models_.size();
    
  for (int curr_class = 0; curr_class < num_class_; ++curr_class) {
    // shrinkage by learning rate
    models_[len - 1 - curr_class]->Shrinkage(new_tree_shrinkage_rate);
    // update score
    UpdateScore(models_[len - 1 - curr_class], curr_class);
  }
  
  bool is_met_early_stopping = false;
  // print message for metric
  if (is_eval) {
    is_met_early_stopping = OutputMetric(iter_ + 1);
  }
  ++iter_;
  if (is_met_early_stopping) {
    Log::Info("Early stopping at iteration %d, the best iteration round is %d",
      iter_, iter_ - early_stopping_round_);
    // pop last early_stopping_round_ models
    for (int i = 0; i < early_stopping_round_ * num_class_; ++i) {
      delete models_.back();
      models_.pop_back();
    }
  }
  return is_met_early_stopping;
}

void DART::UpdateScore(const Tree* tree, const int curr_class) {
  // update training score
  train_score_updater_->AddScore(tree, curr_class);
  // update validation score
  for (auto& score_updater : valid_score_updater_) {
    score_updater->AddScore(tree, curr_class);
  }
}

double DART::SelectDroppingTreesAndNormalize(){
    // if drop rate is too small, treat it as a standard gbdt
    if (drop_rate_ < kEpsilon) {
        return gbdt_config_->learning_rate;
    }
    // select dropping tree indexes based on drop_rate
    std::vector<int> drop_index;
    for (int i = 0; i < iter_; ++i){
        if (random_for_drop_.NextDouble() < drop_rate_) {
            drop_index.push_back(i);
        }
    }
    /*
     * according to the paper, assume number of dropping trees is k:
     * new_tree_shrinkage_rate is n = 1 / (1 + k)
     * drop_tree_shrinkage_rate is m = k / (1 + k)
     * first, shrink the dropping trees to m - 1 = -n
     * update the score, the dropping trees will weight 1 + (m - 1) = m
     * then, shrink the dropping trees to m / (m - 1) = -k
     */
    double k = static_cast<double>(drop_index.size());
    double new_tree_shrinkage_rate = 1.0 / (1.0 + k);
    for (int i: drop_index){
      for (int curr_class = 0; curr_class < num_class_; ++curr_class) {
        int curr_tree = i * num_class_ + curr_class;
        models_[curr_tree]->Shrinkage(-new_tree_shrinkage_rate);
        UpdateScore(models_[curr_tree], curr_class);
        models_[curr_tree]->Shrinkage(-k);
      }
    }
    return new_tree_shrinkage_rate;
}

}  // namespace LightGBM
