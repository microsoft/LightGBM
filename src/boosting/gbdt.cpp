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

namespace LightGBM {

GBDT::GBDT(const BoostingConfig* config)
  : tree_learner_(nullptr), train_score_updater_(nullptr),
  gradients_(nullptr), hessians_(nullptr),
  out_of_bag_data_indices_(nullptr), bag_data_indices_(nullptr) {
  max_feature_idx_ = 0;
  gbdt_config_ = dynamic_cast<const GBDTConfig*>(config);
  early_stopping_round_ = gbdt_config_->early_stopping_round;
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

void GBDT::Init(const Dataset* train_data, const ObjectiveFunction* object_function,
     const std::vector<const Metric*>& training_metrics, const char* output_model_filename) {
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
  gradients_ = new score_t[num_data_];
  hessians_ = new score_t[num_data_];

  // get max feature index
  max_feature_idx_ = train_data_->num_total_features() - 1;

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

  // open model output file
  #ifdef _MSC_VER
  fopen_s(&output_model_file, output_model_filename, "w");
  #else
  output_model_file = fopen(output_model_filename, "w");
  #endif
  // output models
  fprintf(output_model_file, "%s", this->ModelsToString().c_str());
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
    Log::Info("re-bagging, using %d data to train\n", bag_data_cnt_);
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

void GBDT::Train() {
  // training start time
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < gbdt_config_->num_iterations; ++iter) {
    // boosting first
    Boosting();
    // bagging logic
    Bagging(iter);
    // train a new tree
    Tree * new_tree = TrainOneTree();
    // if cannot learn a new tree, then stop
    if (new_tree->num_leaves() <= 1) {
      Log::Info("Can't training anymore, there isn't any leaf meets split requirements.\n");
      break;
    }
    // shrinkage by learning rate
    new_tree->Shrinkage(gbdt_config_->learning_rate);
    // update score
    UpdateScore(new_tree);
    UpdateScoreOutOfBag(new_tree);
    // print message for metric
    bool is_early_stopping = OutputMetric(iter + 1);
    // add model
    models_.push_back(new_tree);
    // save model to file per iteration
    if (early_stopping_round_ > 0){
        // if use early stopping, save previous model at (iter - early_stopping_round_) iteration
        if (iter >= early_stopping_round_){
            fprintf(output_model_file, "Tree=%d\n", iter - early_stopping_round_);
            Tree * printing_tree = models_.at(iter - early_stopping_round_);
            fprintf(output_model_file, "%s\n", printing_tree->ToString().c_str());
            fflush(output_model_file);
        }
    }
    else{
        fprintf(output_model_file, "Tree=%d\n", iter);
        fprintf(output_model_file, "%s\n", new_tree->ToString().c_str());
        fflush(output_model_file);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    // output used time per iteration
    Log::Info("%f seconds elapsed, finished %d iteration\n", std::chrono::duration<double,
                                     std::milli>(end_time - start_time) * 1e-3, iter + 1);
    if (is_early_stopping) {
        // close file with an early-stopping message
        Log::Stdout("early stopping at iteration %d, the best iteration round is %d", iter + 1, iter + 1 - early_stopping_round_);
        fclose(output_model_file);
        return;
    }
  }
  // close file
  if (early_stopping_round_ > 0) {
      // save remaining models
      for (int iter = gbdt_config_->num_iterations - early_stopping_round_; iter < static_cast<int>(models_.size()); ++iter){
        fprintf(output_model_file, "Tree=%d\n", iter);
        fprintf(output_model_file, "%s\n", models_.at(iter)->ToString().c_str());
      }
      fflush(output_model_file);
  }
  fclose(output_model_file);
}

Tree* GBDT::TrainOneTree() {
  return tree_learner_->Train(gradients_, hessians_);
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
  // objective function will calculate gradients and hessians
  object_function_->
    GetGradients(train_score_updater_->score(), gradients_, hessians_);
}


std::string GBDT::ModelsToString() const {
  // serialize this object to string
  std::stringstream ss;
  // output max_feature_idx
  ss << "max_feature_idx=" << max_feature_idx_ << std::endl;
  // output sigmoid parameter
  ss << "sigmoid=" << object_function_->GetSigmoid() << std::endl;
  ss << std::endl;

  // output tree models
  for (size_t i = 0; i < models_.size(); ++i) {
    ss << "Tree=" << i << std::endl;
    ss << models_[i]->ToString() << std::endl;
  }
  return ss.str();
}

void GBDT::ModelsFromString(const std::string& model_str, int num_used_model) {
  // use serialized string to restore this object
  models_.clear();
  std::vector<std::string> lines = Common::Split(model_str.c_str(), '\n');
  size_t i = 0;
  // get max_feature_idx first
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
    Log::Fatal("Model file doesn't contain max_feature_idx\n");
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
      if (num_used_model > 0 && models_.size() >= static_cast<size_t>(num_used_model)) {
        break;
      }
    } else {
      ++i;
    }
  }

  Log::Info("%d models has been loaded\n\n", models_.size());
}

double GBDT::PredictRaw(const double* value) const {
  double ret = 0.0;
  for (size_t i = 0; i < models_.size(); ++i) {
    ret += models_[i]->Predict(value);
  }
  return ret;
}

double GBDT::Predict(const double* value) const {
  double ret = 0.0;
  for (size_t i = 0; i < models_.size(); ++i) {
    ret += models_[i]->Predict(value);
  }
  // if need sigmoid transform
  if (sigmoid_ > 0) {
    ret = 1.0 / (1.0 + std::exp(- 2.0f * sigmoid_ * ret));
  }
  return ret;
}

}  // namespace LightGBM
