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
  // drop trees
  double shrinkage_rate = DroppingTrees();

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
    // shrink new tree
    new_tree->Shrinkage(shrinkage_rate);
    // update score
    UpdateScore(new_tree, curr_class);
    // add model
    models_.push_back(new_tree);
  }

  // normailize
  Normailize(shrinkage_rate);
  
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

double DART::DroppingTrees(){
  drop_index_.clear();
  // select dropping tree indexes based on drop_rate
  // if drop rate is too small, skip this step, drop one tree randomly
  if (drop_rate_ > kEpsilon) {
    for (size_t i = 0; i < static_cast<size_t>(iter_); ++i){
      if (random_for_drop_.NextDouble() < drop_rate_) {
        drop_index_.push_back(i);
      }
    }
  }
  // binomial-plus-one, at least one tree will be dropped
  if (drop_index_.empty()){
    drop_index_ = random_for_drop_.Sample(iter_, 1);
  }
  // drop trees
  for (int i: drop_index_){
    for (int curr_class = 0; curr_class < num_class_; ++curr_class) {
      int curr_tree = i * num_class_ + curr_class;
      models_[curr_tree]->Shrinkage(-1);
      UpdateScore(models_[curr_tree], curr_class);
    }
  }
  return 1.0 / (1.0 + drop_index_.size());
}

void DART::Normailize(double shrinkage_rate) {
  double k = static_cast<double>(drop_index_.size());
  for (int i: drop_index_){
    for (int curr_class = 0; curr_class < num_class_; ++curr_class) {
      int curr_tree = i * num_class_ + curr_class;
      models_[curr_tree]->Shrinkage(-k * shrinkage_rate);
      UpdateScore(models_[curr_tree], curr_class);
    }
  }
}

}  // namespace LightGBM
