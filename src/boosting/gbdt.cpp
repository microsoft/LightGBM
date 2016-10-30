#include "gbdt.h"

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

GBDT::GBDT()
  : tree_learner_(nullptr), train_score_updater_(nullptr),
  gradients_(nullptr), hessians_(nullptr),
  out_of_bag_data_indices_(nullptr), bag_data_indices_(nullptr) {
}

GBDT::~GBDT() {
  if (tree_learner_ != nullptr) { delete tree_learner_; }
  if (gradients_ != nullptr) { delete[] gradients_; }
  if (hessians_ != nullptr) { delete[] hessians_; }
  if (out_of_bag_data_indices_ != nullptr) { delete[] out_of_bag_data_indices_; }
  if (bag_data_indices_ != nullptr) { delete[] bag_data_indices_; }
  for (auto& tree : models_) {
    if (tree != nullptr) { delete tree; }
  }
  if (train_score_updater_ != nullptr) { delete train_score_updater_; }
  for (auto& score_tracker : valid_score_updater_) {
    if (score_tracker != nullptr) { delete score_tracker; }
  }
}

void GBDT::Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* object_function,
     const std::vector<const Metric*>& training_metrics) {
  gbdt_config_ = dynamic_cast<const GBDTConfig*>(config);
  iter_ = 0;
  max_feature_idx_ = 0;
  early_stopping_round_ = gbdt_config_->early_stopping_round;
  train_data_ = train_data;
  // create tree learner
  tree_learner_ =
    TreeLearner::CreateTreeLearner(gbdt_config_->tree_learner_type, gbdt_config_->tree_config);
  // init tree learner
  tree_learner_->Init(train_data_);
  object_function_ = object_function;
  // push training metrics
  for (const auto& metric : training_metrics) {
    training_metrics_.push_back(metric);
  }
  // create score tracker
  train_score_updater_ = new ScoreUpdater(train_data_);
  num_data_ = train_data_->num_data();
  // create buffer for gradients and hessians
  if (object_function_ != nullptr) {
    gradients_ = new score_t[num_data_];
    hessians_ = new score_t[num_data_];
  }

  // get max feature index
  max_feature_idx_ = train_data_->num_total_features() - 1;
  // get label index
  label_idx_ = train_data_->label_idx();
  // if need bagging, create buffer
  if (gbdt_config_->bagging_fraction < 1.0 && gbdt_config_->bagging_freq > 0) {
    out_of_bag_data_indices_ = new data_size_t[num_data_];
    bag_data_indices_ = new data_size_t[num_data_];
  } else {
    out_of_bag_data_cnt_ = 0;
    out_of_bag_data_indices_ = nullptr;
    bag_data_cnt_ = num_data_;
    bag_data_indices_ = nullptr;
  }
  // initialize random generator
  random_ = Random(gbdt_config_->bagging_seed);

}

void GBDT::AddDataset(const Dataset* valid_data,
         const std::vector<const Metric*>& valid_metrics) {
  // for a validation dataset, we need its score and metric
  valid_score_updater_.push_back(new ScoreUpdater(valid_data));
  valid_metrics_.emplace_back();
  best_iter_.emplace_back();
  best_score_.emplace_back();
  for (const auto& metric : valid_metrics) {
    valid_metrics_.back().push_back(metric);
    best_iter_.back().push_back(0);
    best_score_.back().push_back(-1);
  }
}


void GBDT::Bagging(int iter) {
  // if need bagging
  if (out_of_bag_data_indices_ != nullptr && iter % gbdt_config_->bagging_freq == 0) {
    // if doesn't have query data
    if (train_data_->metadata().query_boundaries() == nullptr) {
      bag_data_cnt_ =
        static_cast<data_size_t>(gbdt_config_->bagging_fraction * num_data_);
      out_of_bag_data_cnt_ = num_data_ - bag_data_cnt_;
      data_size_t cur_left_cnt = 0;
      data_size_t cur_right_cnt = 0;
      // random bagging, minimal unit is one record
      for (data_size_t i = 0; i < num_data_; ++i) {
        double prob =
          (bag_data_cnt_ - cur_left_cnt) / static_cast<double>(num_data_ - i);
        if (random_.NextDouble() < prob) {
          bag_data_indices_[cur_left_cnt++] = i;
        } else {
          out_of_bag_data_indices_[cur_right_cnt++] = i;
        }
      }
    } else {
      // if have query data
      const data_size_t* query_boundaries = train_data_->metadata().query_boundaries();
      data_size_t num_query = train_data_->metadata().num_queries();
      data_size_t bag_query_cnt =
          static_cast<data_size_t>(num_query * gbdt_config_->bagging_fraction);
      data_size_t cur_left_query_cnt = 0;
      data_size_t cur_left_cnt = 0;
      data_size_t cur_right_cnt = 0;
      // random bagging, minimal unit is one query
      for (data_size_t i = 0; i < num_query; ++i) {
        double prob =
            (bag_query_cnt - cur_left_query_cnt) / static_cast<double>(num_query - i);
        if (random_.NextDouble() < prob) {
          for (data_size_t j = query_boundaries[i]; j < query_boundaries[i + 1]; ++j) {
            bag_data_indices_[cur_left_cnt++] = j;
          }
          cur_left_query_cnt++;
        } else {
          for (data_size_t j = query_boundaries[i]; j < query_boundaries[i + 1]; ++j) {
            out_of_bag_data_indices_[cur_right_cnt++] = j;
          }
        }
      }
      bag_data_cnt_ = cur_left_cnt;
      out_of_bag_data_cnt_ = num_data_ - bag_data_cnt_;
    }
    Log::Info("re-bagging, using %d data to train", bag_data_cnt_);
    // set bagging data to tree learner
    tree_learner_->SetBaggingData(bag_data_indices_, bag_data_cnt_);
  }
}

void GBDT::UpdateScoreOutOfBag(const Tree* tree) {
  // we need to predict out-of-bag socres of data for boosting
  if (out_of_bag_data_indices_ != nullptr) {
    train_score_updater_->
      AddScore(tree, out_of_bag_data_indices_, out_of_bag_data_cnt_);
  }
}

bool GBDT::TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) {
  // boosting first
  if (gradient == nullptr || hessian == nullptr) {
    Boosting();
    gradient = gradients_;
    hessian = hessians_;
  }
  // bagging logic
  Bagging(iter_);
  // train a new tree
  Tree * new_tree = tree_learner_->Train(gradient, hessian);
  // if cannot learn a new tree, then stop
  if (new_tree->num_leaves() <= 1) {
    Log::Info("Can't training anymore, there isn't any leaf meets split requirements.");
    return true;
  }
  // shrinkage by learning rate
  new_tree->Shrinkage(gbdt_config_->learning_rate);
  // update score
  UpdateScore(new_tree);
  UpdateScoreOutOfBag(new_tree);
  bool is_met_early_stopping = false;
  // print message for metric
  if (is_eval) {
    is_met_early_stopping = OutputMetric(iter_ + 1);
  }
  // add model
  models_.push_back(new_tree);
  ++iter_;
  if (is_met_early_stopping) {
    Log::Info("Early stopping at iteration %d, the best iteration round is %d",
      iter_ + 1, iter_ + 1 - early_stopping_round_);
    // pop last early_stopping_round_ models
    for (int i = 0; i < early_stopping_round_; ++i) {
      delete models_.back();
      models_.pop_back();
    }
  }
  return is_met_early_stopping;
  
}

void GBDT::UpdateScore(const Tree* tree) {
  // update training score
  train_score_updater_->AddScore(tree_learner_);
  // update validation score
  for (auto& score_tracker : valid_score_updater_) {
    score_tracker->AddScore(tree);
  }
}

bool GBDT::OutputMetric(int iter) {
  bool ret = false;
  // print training metric
  for (auto& sub_metric : training_metrics_) {
    sub_metric->PrintAndGetLoss(iter, train_score_updater_->score());
  }
  // print validation metric
  for (size_t i = 0; i < valid_metrics_.size(); ++i) {
    for (size_t j = 0; j < valid_metrics_[i].size(); ++j) {
      score_t test_score_ = valid_metrics_[i][j]->PrintAndGetLoss(iter, valid_score_updater_[i]->score());
      if (!ret && early_stopping_round_ > 0){
        bool the_bigger_the_better_ = valid_metrics_[i][j]->the_bigger_the_better;
        if (best_score_[i][j] < 0
            || (!the_bigger_the_better_ && test_score_ < best_score_[i][j])
            || ( the_bigger_the_better_ && test_score_ > best_score_[i][j])){
            best_score_[i][j] = test_score_;
            best_iter_[i][j] = iter;
        }
        else {
          if (iter - best_iter_[i][j] >= early_stopping_round_) ret = true;
        }
      }
    }
  }
  return ret;
}

void GBDT::Boosting() {
  if (object_function_ == nullptr) {
    Log::Fatal("No object function provided");
  }
  // objective function will calculate gradients and hessians
  object_function_->
    GetGradients(train_score_updater_->score(), gradients_, hessians_);
}

void GBDT::SaveModelToFile(bool is_finish, const char* filename) {

  // first time to this function, open file
  if (saved_model_size_ == -1) {
    model_output_file_.open(filename);
    // output model type
    model_output_file_ << "gbdt" << label_idx_ << std::endl;
    // output label index
    model_output_file_ << "label_index=" << label_idx_ << std::endl;
    // output max_feature_idx
    model_output_file_ << "max_feature_idx=" << max_feature_idx_ << std::endl;
    // output sigmoid parameter
    model_output_file_ << "sigmoid=" << object_function_->GetSigmoid() << std::endl;
    model_output_file_ << std::endl;
    saved_model_size_ = 0;
  }
  // already saved
  if (!model_output_file_.is_open()) {
    return;
  }
  int rest = static_cast<int>(models_.size()) - early_stopping_round_;
  // output tree models
  for (int i = saved_model_size_; i < rest; ++i) {
    model_output_file_ << "Tree=" << i << std::endl;
    model_output_file_ << models_[i]->ToString() << std::endl;
  }

  if (rest > 0) {
    saved_model_size_ = rest;
  }

  model_output_file_.flush();
  // training finished, can close file
  if (is_finish) {
    for (int i = saved_model_size_; i < static_cast<int>(models_.size()); ++i) {
      model_output_file_ << "Tree=" << i << std::endl;
      model_output_file_ << models_[i]->ToString() << std::endl;
    }
    model_output_file_ << std::endl << FeatureImportance() << std::endl;
    model_output_file_.close();
  }
}

void GBDT::ModelsFromString(const std::string& model_str) {
  // use serialized string to restore this object
  models_.clear();
  std::vector<std::string> lines = Common::Split(model_str.c_str(), '\n');
  size_t i = 0;

  // get index of label
  while (i < lines.size()) {
    size_t find_pos = lines[i].find("label_index=");
    if (find_pos != std::string::npos) {
      std::vector<std::string> strs = Common::Split(lines[i].c_str(), '=');
      Common::Atoi(strs[1].c_str(), &label_idx_);
      ++i;
      break;
    } else {
      ++i;
    }
  }
  if (i == lines.size()) {
    Log::Fatal("Model file doesn't contain label index");
    return;
  }

  // get max_feature_idx first
  i = 0;
  while (i < lines.size()) {
    size_t find_pos = lines[i].find("max_feature_idx=");
    if (find_pos != std::string::npos) {
      std::vector<std::string> strs = Common::Split(lines[i].c_str(), '=');
      Common::Atoi(strs[1].c_str(), &max_feature_idx_);
      ++i;
      break;
    } else {
      ++i;
    }
  }
  if (i == lines.size()) {
    Log::Fatal("Model file doesn't contain max_feature_idx");
    return;
  }
  // get sigmoid parameter
  i = 0;
  while (i < lines.size()) {
    size_t find_pos = lines[i].find("sigmoid=");
    if (find_pos != std::string::npos) {
      std::vector<std::string> strs = Common::Split(lines[i].c_str(), '=');
      Common::Atof(strs[1].c_str(), &sigmoid_);
      ++i;
      break;
    } else {
      ++i;
    }
  }
  // if sigmoid doesn't exists
  if (i == lines.size()) {
    sigmoid_ = -1.0;
  }
  // get tree models
  i = 0;
  while (i < lines.size()) {
    size_t find_pos = lines[i].find("Tree=");
    if (find_pos != std::string::npos) {
      ++i;
      int start = static_cast<int>(i);
      while (i < lines.size() && lines[i].find("Tree=") == std::string::npos) { ++i; }
      int end = static_cast<int>(i);
      std::string tree_str = Common::Join(lines, start, end, '\n');
      models_.push_back(new Tree(tree_str));
    } else {
      ++i;
    }
  }
  Log::Info("%d models has been loaded\n", models_.size());
}

std::string GBDT::FeatureImportance() const {
  std::vector<size_t> feature_importances(max_feature_idx_ + 1, 0);
    for (size_t iter = 0; iter < models_.size(); ++iter) {
        for (int split_idx = 0; split_idx < models_[iter]->num_leaves() - 1; ++split_idx) {
            ++feature_importances[models_[iter]->split_feature_real(split_idx)];
        }
    }
    // store the importance first
    std::vector<std::pair<size_t, std::string>> pairs;
    for (size_t i = 0; i < feature_importances.size(); ++i) {
      pairs.emplace_back(feature_importances[i], train_data_->feature_names()[i]);
    }
    // sort the importance
    std::sort(pairs.begin(), pairs.end(),
      [](const std::pair<size_t, std::string>& lhs,
        const std::pair<size_t, std::string>& rhs) {
      return lhs.first > rhs.first;
    });
    std::stringstream str_buf;
    // write to model file
    str_buf << std::endl << "feature importances:" << std::endl;
    for (size_t i = 0; i < pairs.size(); ++i) {
      str_buf << pairs[i].second << "=" << std::to_string(pairs[i].first) << std::endl;
    }
    return str_buf.str();
}

double GBDT::PredictRaw(const double* value, size_t num_used_model) const {
  if (num_used_model < 0) {
    num_used_model = models_.size();
  }
  double ret = 0.0;
  for (size_t i = 0; i < num_used_model; ++i) {
    ret += models_[i]->Predict(value);
  }
  return ret;
}

double GBDT::Predict(const double* value, size_t num_used_model) const {
  if (num_used_model < 0) {
    num_used_model = models_.size();
  }
  double ret = 0.0;
  for (size_t i = 0; i < num_used_model; ++i) {
    ret += models_[i]->Predict(value);
  }
  // if need sigmoid transform
  if (sigmoid_ > 0) {
    ret = 1.0 / (1.0 + std::exp(- 2.0f * sigmoid_ * ret));
  }
  return ret;
}

std::vector<int> GBDT::PredictLeafIndex(const double* value, size_t num_used_model) const {
  if (num_used_model < 0) {
    num_used_model = models_.size();
  }
  std::vector<int> ret;
  for (size_t i = 0; i < num_used_model; ++i) {
    ret.push_back(models_[i]->PredictLeafIndex(value));
  }
  return ret;
}

}  // namespace LightGBM
