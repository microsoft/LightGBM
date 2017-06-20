#include "gbdt.h"

#include <LightGBM/utils/openmp_wrapper.h>

#include <LightGBM/utils/common.h>

#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/network.h>

#include <ctime>

#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <utility>

namespace LightGBM {

#ifdef TIMETAG
std::chrono::duration<double, std::milli> boosting_time;
std::chrono::duration<double, std::milli> train_score_time;
std::chrono::duration<double, std::milli> out_of_bag_score_time;
std::chrono::duration<double, std::milli> valid_score_time;
std::chrono::duration<double, std::milli> metric_time;
std::chrono::duration<double, std::milli> bagging_time;
std::chrono::duration<double, std::milli> sub_gradient_time;
std::chrono::duration<double, std::milli> tree_time;
#endif // TIMETAG

GBDT::GBDT()
  :iter_(0),
  train_data_(nullptr),
  objective_function_(nullptr),
  early_stopping_round_(0),
  max_feature_idx_(0),
  num_tree_per_iteration_(1),
  num_class_(1),
  num_iteration_for_pred_(0),
  shrinkage_rate_(0.1f),
  num_init_iteration_(0),
  boost_from_average_(false) {
  #pragma omp parallel
  #pragma omp master
  {
    num_threads_ = omp_get_num_threads();
  }
}

GBDT::~GBDT() {
  #ifdef TIMETAG
  Log::Info("GBDT::boosting costs %f", boosting_time * 1e-3);
  Log::Info("GBDT::train_score costs %f", train_score_time * 1e-3);
  Log::Info("GBDT::out_of_bag_score costs %f", out_of_bag_score_time * 1e-3);
  Log::Info("GBDT::valid_score costs %f", valid_score_time * 1e-3);
  Log::Info("GBDT::metric costs %f", metric_time * 1e-3);
  Log::Info("GBDT::bagging costs %f", bagging_time * 1e-3);
  Log::Info("GBDT::sub_gradient costs %f", sub_gradient_time * 1e-3);
  Log::Info("GBDT::tree costs %f", tree_time * 1e-3);
  #endif
}

void GBDT::Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
                const std::vector<const Metric*>& training_metrics) {
  iter_ = 0;
  num_iteration_for_pred_ = 0;
  max_feature_idx_ = 0;
  num_class_ = config->num_class;
  train_data_ = nullptr;
  gbdt_config_ = nullptr;
  tree_learner_ = nullptr;
  ResetTrainingData(config, train_data, objective_function, training_metrics);
}

void GBDT::ResetTrainingData(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
                             const std::vector<const Metric*>& training_metrics) {
  auto new_config = std::unique_ptr<BoostingConfig>(new BoostingConfig(*config));
  if (train_data_ != nullptr && !train_data_->CheckAlign(*train_data)) {
    Log::Fatal("cannot reset training data, since new training data has different bin mappers");
  }
  early_stopping_round_ = new_config->early_stopping_round;
  shrinkage_rate_ = new_config->learning_rate;

  objective_function_ = objective_function;
  num_tree_per_iteration_ = num_class_;
  if (objective_function_ != nullptr) {
    is_constant_hessian_ = objective_function_->IsConstantHessian();
    num_tree_per_iteration_ = objective_function_->NumTreePerIteration();
  } else {
    is_constant_hessian_ = false;
  }

  if (train_data_ != train_data && train_data != nullptr) {
    if (tree_learner_ == nullptr) {
      tree_learner_ = std::unique_ptr<TreeLearner>(TreeLearner::CreateTreeLearner(new_config->tree_learner_type, new_config->device_type, &new_config->tree_config));
    }
    // init tree learner
    tree_learner_->Init(train_data, is_constant_hessian_);

    // push training metrics
    training_metrics_.clear();
    for (const auto& metric : training_metrics) {
      training_metrics_.push_back(metric);
    }
    training_metrics_.shrink_to_fit();
    // not same training data, need reset score and others
    // create score tracker
    train_score_updater_.reset(new ScoreUpdater(train_data, num_tree_per_iteration_));
    // update score
    for (int i = 0; i < iter_; ++i) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        auto curr_tree = (i + num_init_iteration_) * num_tree_per_iteration_ + cur_tree_id;
        train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
      }
    }
    num_data_ = train_data->num_data();
    // create buffer for gradients and hessians
    if (objective_function_ != nullptr) {
      size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
      gradients_.resize(total_size);
      hessians_.resize(total_size);
    }
    // get max feature index
    max_feature_idx_ = train_data->num_total_features() - 1;
    // get label index
    label_idx_ = train_data->label_idx();
    // get feature names
    feature_names_ = train_data->feature_names();

    feature_infos_ = train_data->feature_infos();
  }

  if ((train_data_ != train_data && train_data != nullptr)
      || (gbdt_config_ != nullptr && gbdt_config_->bagging_fraction != new_config->bagging_fraction)) {
    // if need bagging, create buffer
    if (new_config->bagging_fraction < 1.0 && new_config->bagging_freq > 0) {
      bag_data_cnt_ =
        static_cast<data_size_t>(new_config->bagging_fraction * num_data_);
      bag_data_indices_.resize(num_data_);
      tmp_indices_.resize(num_data_);
      offsets_buf_.resize(num_threads_);
      left_cnts_buf_.resize(num_threads_);
      right_cnts_buf_.resize(num_threads_);
      left_write_pos_buf_.resize(num_threads_);
      right_write_pos_buf_.resize(num_threads_);
      double average_bag_rate = new_config->bagging_fraction / new_config->bagging_freq;
      int sparse_group = 0;
      for (int i = 0; i < train_data->num_feature_groups(); ++i) {
        if (train_data->FeatureGroupIsSparse(i)) {
          ++sparse_group;
        }
      }
      is_use_subset_ = false;
      const int group_threshold_usesubset = 100;
      const int sparse_group_threshold_usesubset = train_data->num_feature_groups() / 4;
      if (average_bag_rate <= 0.5 
          && (train_data->num_feature_groups() < group_threshold_usesubset || sparse_group < sparse_group_threshold_usesubset)) {
        tmp_subset_.reset(new Dataset(bag_data_cnt_));
        tmp_subset_->CopyFeatureMapperFrom(train_data);
        is_use_subset_ = true;
        Log::Debug("use subset for bagging");
      }
    } else {
      bag_data_cnt_ = num_data_;
      bag_data_indices_.clear();
      tmp_indices_.clear();
      is_use_subset_ = false;
    }
  }
  train_data_ = train_data;
  if (train_data_ != nullptr) {
    // reset config for tree learner
    tree_learner_->ResetConfig(&new_config->tree_config);
    class_need_train_ = std::vector<bool>(num_tree_per_iteration_, true);
    if (objective_function_ != nullptr && objective_function_->SkipEmptyClass()) {
      CHECK(num_tree_per_iteration_ == num_class_);
      // + 1 here for the binary classification
      class_default_output_ = std::vector<double>(num_tree_per_iteration_, 0.0f);
      auto label = train_data_->metadata().label();
      if (num_tree_per_iteration_ > 1) {
        // multi-class
        std::vector<data_size_t> cnt_per_class(num_tree_per_iteration_, 0);
        for (data_size_t i = 0; i < num_data_; ++i) {
          int index = static_cast<int>(label[i]);
          CHECK(index < num_tree_per_iteration_);
          ++cnt_per_class[index];
        }
        for (int i = 0; i < num_tree_per_iteration_; ++i) {
          if (cnt_per_class[i] == num_data_) {
            class_need_train_[i] = false;
            class_default_output_[i] = -std::log(kEpsilon);
          } else if (cnt_per_class[i] == 0) {
            class_need_train_[i] = false;
            class_default_output_[i] = -std::log(1.0f / kEpsilon - 1.0f);
          }
        }
      } else {
        // binary class
        data_size_t cnt_pos = 0;
        for (data_size_t i = 0; i < num_data_; ++i) {
          if (label[i] > 0) {
            ++cnt_pos;
          }
        }
        if (cnt_pos == 0) {
          class_need_train_[0] = false;
          class_default_output_[0] = -std::log(1.0f / kEpsilon - 1.0f);
        } else if (cnt_pos == num_data_) {
          class_need_train_[0] = false;
          class_default_output_[0] = -std::log(kEpsilon);
        }
      }
    }
  }
  gbdt_config_.reset(new_config.release());
}

void GBDT::AddValidDataset(const Dataset* valid_data,
                           const std::vector<const Metric*>& valid_metrics) {
  if (!train_data_->CheckAlign(*valid_data)) {
    Log::Fatal("cannot add validation data, since it has different bin mappers with training data");
  }
  // for a validation dataset, we need its score and metric
  auto new_score_updater = std::unique_ptr<ScoreUpdater>(new ScoreUpdater(valid_data, num_tree_per_iteration_));
  // update score
  for (int i = 0; i < iter_; ++i) {
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      auto curr_tree = (i + num_init_iteration_) * num_tree_per_iteration_ + cur_tree_id;
      new_score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
    }
  }
  valid_score_updater_.push_back(std::move(new_score_updater));
  valid_metrics_.emplace_back();
  if (early_stopping_round_ > 0) {
    best_iter_.emplace_back();
    best_score_.emplace_back();
    best_msg_.emplace_back();
  }
  for (const auto& metric : valid_metrics) {
    valid_metrics_.back().push_back(metric);
    if (early_stopping_round_ > 0) {
      best_iter_.back().push_back(0);
      best_score_.back().push_back(kMinScore);
      best_msg_.back().emplace_back();
    }
  }
  valid_metrics_.back().shrink_to_fit();
}

data_size_t GBDT::BaggingHelper(Random& cur_rand, data_size_t start, data_size_t cnt, data_size_t* buffer) {
  if (cnt <= 0) {
    return 0;
  }
  data_size_t bag_data_cnt =
    static_cast<data_size_t>(gbdt_config_->bagging_fraction * cnt);
  data_size_t cur_left_cnt = 0;
  data_size_t cur_right_cnt = 0;
  auto right_buffer = buffer + bag_data_cnt;
  // random bagging, minimal unit is one record
  for (data_size_t i = 0; i < cnt; ++i) {
    float prob =
      (bag_data_cnt - cur_left_cnt) / static_cast<float>(cnt - i);
    if (cur_rand.NextFloat() < prob) {
      buffer[cur_left_cnt++] = start + i;
    } else {
      right_buffer[cur_right_cnt++] = start + i;
    }
  }
  CHECK(cur_left_cnt == bag_data_cnt);
  return cur_left_cnt;
}

void GBDT::Bagging(int iter) {
  // if need bagging
  if (bag_data_cnt_ < num_data_ && iter % gbdt_config_->bagging_freq == 0) {
    const data_size_t min_inner_size = 1000;
    data_size_t inner_size = (num_data_ + num_threads_ - 1) / num_threads_;
    if (inner_size < min_inner_size) { inner_size = min_inner_size; }
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < num_threads_; ++i) {
      OMP_LOOP_EX_BEGIN();
      left_cnts_buf_[i] = 0;
      right_cnts_buf_[i] = 0;
      data_size_t cur_start = i * inner_size;
      if (cur_start > num_data_) { continue; }
      data_size_t cur_cnt = inner_size;
      if (cur_start + cur_cnt > num_data_) { cur_cnt = num_data_ - cur_start; }
      Random cur_rand(gbdt_config_->bagging_seed + iter * num_threads_ + i);
      data_size_t cur_left_count = BaggingHelper(cur_rand, cur_start, cur_cnt, tmp_indices_.data() + cur_start);
      offsets_buf_[i] = cur_start;
      left_cnts_buf_[i] = cur_left_count;
      right_cnts_buf_[i] = cur_cnt - cur_left_count;
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    data_size_t left_cnt = 0;
    left_write_pos_buf_[0] = 0;
    right_write_pos_buf_[0] = 0;
    for (int i = 1; i < num_threads_; ++i) {
      left_write_pos_buf_[i] = left_write_pos_buf_[i - 1] + left_cnts_buf_[i - 1];
      right_write_pos_buf_[i] = right_write_pos_buf_[i - 1] + right_cnts_buf_[i - 1];
    }
    left_cnt = left_write_pos_buf_[num_threads_ - 1] + left_cnts_buf_[num_threads_ - 1];

    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < num_threads_; ++i) {
      OMP_LOOP_EX_BEGIN();
      if (left_cnts_buf_[i] > 0) {
        std::memcpy(bag_data_indices_.data() + left_write_pos_buf_[i],
                    tmp_indices_.data() + offsets_buf_[i], left_cnts_buf_[i] * sizeof(data_size_t));
      }
      if (right_cnts_buf_[i] > 0) {
        std::memcpy(bag_data_indices_.data() + left_cnt + right_write_pos_buf_[i],
                    tmp_indices_.data() + offsets_buf_[i] + left_cnts_buf_[i], right_cnts_buf_[i] * sizeof(data_size_t));
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    bag_data_cnt_ = left_cnt;
    Log::Debug("Re-bagging, using %d data to train", bag_data_cnt_);
    // set bagging data to tree learner
    if (!is_use_subset_) {
      tree_learner_->SetBaggingData(bag_data_indices_.data(), bag_data_cnt_);
    } else {
      // get subset
      tmp_subset_->ReSize(bag_data_cnt_);
      tmp_subset_->CopySubset(train_data_, bag_data_indices_.data(), bag_data_cnt_, false);
      tree_learner_->ResetTrainingData(tmp_subset_.get());
    }
  }
}

void GBDT::UpdateScoreOutOfBag(const Tree* tree, const int cur_tree_id) {
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif
  // we need to predict out-of-bag scores of data for boosting
  if (num_data_ - bag_data_cnt_ > 0 && !is_use_subset_) {
    train_score_updater_->AddScore(tree, bag_data_indices_.data() + bag_data_cnt_, num_data_ - bag_data_cnt_, cur_tree_id);
  }
  #ifdef TIMETAG
  out_of_bag_score_time += std::chrono::steady_clock::now() - start_time;
  #endif
}

double LabelAverage(const float* label, data_size_t num_data) {
  double sum_label = 0.0f;
  #pragma omp parallel for schedule(static) reduction(+:sum_label)
  for (data_size_t i = 0; i < num_data; ++i) {
    sum_label += label[i];
  }
  double init_score = sum_label / num_data;
  if (Network::num_machines() > 1) {
    double global_init_score = 0.0f;
    Network::Allreduce(reinterpret_cast<char*>(&init_score),
                       sizeof(init_score), sizeof(init_score),
                       reinterpret_cast<char*>(&global_init_score),
                       [](const char* src, char* dst, int len) {
      int used_size = 0;
      const int type_size = sizeof(double);
      const double *p1;
      double *p2;
      while (used_size < len) {
        p1 = reinterpret_cast<const double *>(src);
        p2 = reinterpret_cast<double *>(dst);
        *p2 += *p1;
        src += type_size;
        dst += type_size;
        used_size += type_size;
      }
    });
    return global_init_score / Network::num_machines();
  } else {
    return init_score;
  }
}

bool GBDT::TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) {
  // boosting from average prediction. It doesn't work well for classification, remove it for now.
  if (models_.empty()
      && gbdt_config_->boost_from_average
      && !train_score_updater_->has_init_score()
      && num_class_ <= 1
      && objective_function_ != nullptr
      && objective_function_->BoostFromAverage()) {
    auto label = train_data_->metadata().label();
    double init_score = LabelAverage(label, num_data_);
    std::unique_ptr<Tree> new_tree(new Tree(2));
    new_tree->Split(0, 0, BinType::NumericalBin, 0, 0, 0, init_score, init_score, 0, 0, -1, 0, 0, 0);
    train_score_updater_->AddScore(init_score, 0);
    for (auto& score_updater : valid_score_updater_) {
      score_updater->AddScore(init_score, 0);
    }
    models_.push_back(std::move(new_tree));
    boost_from_average_ = true;
  }
  // boosting first
  if (gradient == nullptr || hessian == nullptr) {
    #ifdef TIMETAG
    auto start_time = std::chrono::steady_clock::now();
    #endif
    Boosting();
    gradient = gradients_.data();
    hessian = hessians_.data();
    #ifdef TIMETAG
    boosting_time += std::chrono::steady_clock::now() - start_time;
    #endif
  }
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif
  // bagging logic
  Bagging(iter_);
  #ifdef TIMETAG
  bagging_time += std::chrono::steady_clock::now() - start_time;
  #endif
  if (is_use_subset_ && bag_data_cnt_ < num_data_) {
    #ifdef TIMETAG
    start_time = std::chrono::steady_clock::now();
    #endif
    if (gradients_.empty()) {
      size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
      gradients_.resize(total_size);
      hessians_.resize(total_size);
    }
    // get sub gradients
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      size_t bias = static_cast<size_t>(cur_tree_id)* num_data_;
      // cannot multi-threading here.
      for (int i = 0; i < bag_data_cnt_; ++i) {
        gradients_[bias + i] = gradient[bias + bag_data_indices_[i]];
        hessians_[bias + i] = hessian[bias + bag_data_indices_[i]];
      }
    }
    gradient = gradients_.data();
    hessian = hessians_.data();
    #ifdef TIMETAG
    sub_gradient_time += std::chrono::steady_clock::now() - start_time;
    #endif
  }
  bool should_continue = false;
  for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
    #ifdef TIMETAG
    start_time = std::chrono::steady_clock::now();
    #endif
    std::unique_ptr<Tree> new_tree(new Tree(2));
    if (class_need_train_[cur_tree_id]) {
      size_t bias = static_cast<size_t>(cur_tree_id)* num_data_;
      new_tree.reset(
        tree_learner_->Train(gradient + bias, hessian + bias, is_constant_hessian_));
    }
    #ifdef TIMETAG
    tree_time += std::chrono::steady_clock::now() - start_time;
    #endif

    if (new_tree->num_leaves() > 1) {
      should_continue = true;
      // shrinkage by learning rate
      new_tree->Shrinkage(shrinkage_rate_);
      // update score
      UpdateScore(new_tree.get(), cur_tree_id);
      UpdateScoreOutOfBag(new_tree.get(), cur_tree_id);
    } else {
      // only add default score one-time
      if (!class_need_train_[cur_tree_id] && models_.size() < static_cast<size_t>(num_tree_per_iteration_)) {
        auto output = class_default_output_[cur_tree_id];
        new_tree->Split(0, 0, BinType::NumericalBin, 0, 0, 0,
                        output, output, 0, 0, -1, 0, 0, 0);
        train_score_updater_->AddScore(output, cur_tree_id);
        for (auto& score_updater : valid_score_updater_) {
          score_updater->AddScore(output, cur_tree_id);
        }
      }
    }
    // add model
    models_.push_back(std::move(new_tree));
  }
  if (!should_continue) {
    Log::Warning("Stopped training because there are no more leaves that meet the split requirements.");
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      models_.pop_back();
    }
    return true;
  }
  ++iter_;
  if (is_eval) {
    return EvalAndCheckEarlyStopping();
  } else {
    return false;
  }

}

void GBDT::RollbackOneIter() {
  if (iter_ <= 0) { return; }
  int cur_iter = iter_ + num_init_iteration_ - 1;
  // reset score
  for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
    auto curr_tree = cur_iter * num_tree_per_iteration_ + cur_tree_id;
    models_[curr_tree]->Shrinkage(-1.0);
    train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
    for (auto& score_updater : valid_score_updater_) {
      score_updater->AddScore(models_[curr_tree].get(), cur_tree_id);
    }
  }
  // remove model
  for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
    models_.pop_back();
  }
  --iter_;
}

bool GBDT::EvalAndCheckEarlyStopping() {
  bool is_met_early_stopping = false;
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif
  // print message for metric
  auto best_msg = OutputMetric(iter_);
  #ifdef TIMETAG
  metric_time += std::chrono::steady_clock::now() - start_time;
  #endif
  is_met_early_stopping = !best_msg.empty();
  if (is_met_early_stopping) {
    Log::Info("Early stopping at iteration %d, the best iteration round is %d",
              iter_, iter_ - early_stopping_round_);
    Log::Info("Output of best iteration round:\n%s", best_msg.c_str());
    // pop last early_stopping_round_ models
    for (int i = 0; i < early_stopping_round_ * num_tree_per_iteration_; ++i) {
      models_.pop_back();
    }
  }
  return is_met_early_stopping;
}

void GBDT::UpdateScore(const Tree* tree, const int cur_tree_id) {
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif
  // update training score
  if (!is_use_subset_) {
    train_score_updater_->AddScore(tree_learner_.get(), tree, cur_tree_id);
  } else {
    train_score_updater_->AddScore(tree, cur_tree_id);
  }
  #ifdef TIMETAG
  train_score_time += std::chrono::steady_clock::now() - start_time;
  #endif
  #ifdef TIMETAG
  start_time = std::chrono::steady_clock::now();
  #endif
  // update validation score
  for (auto& score_updater : valid_score_updater_) {
    score_updater->AddScore(tree, cur_tree_id);
  }
  #ifdef TIMETAG
  valid_score_time += std::chrono::steady_clock::now() - start_time;
  #endif
}

std::string GBDT::OutputMetric(int iter) {
  bool need_output = (iter % gbdt_config_->output_freq) == 0;
  std::string ret = "";
  std::stringstream msg_buf;
  std::vector<std::pair<size_t, size_t>> meet_early_stopping_pairs;
  // print training metric
  if (need_output) {
    for (auto& sub_metric : training_metrics_) {
      auto name = sub_metric->GetName();
      auto scores = sub_metric->Eval(train_score_updater_->score(), objective_function_);
      for (size_t k = 0; k < name.size(); ++k) {
        std::stringstream tmp_buf;
        tmp_buf << "Iteration:" << iter
          << ", training " << name[k]
          << " : " << scores[k];
        Log::Info(tmp_buf.str().c_str());
        if (early_stopping_round_ > 0) {
          msg_buf << tmp_buf.str() << std::endl;
        }
      }
    }
  }
  // print validation metric
  if (need_output || early_stopping_round_ > 0) {
    for (size_t i = 0; i < valid_metrics_.size(); ++i) {
      for (size_t j = 0; j < valid_metrics_[i].size(); ++j) {
        auto test_scores = valid_metrics_[i][j]->Eval(valid_score_updater_[i]->score(),
                                                      objective_function_);
        auto name = valid_metrics_[i][j]->GetName();
        for (size_t k = 0; k < name.size(); ++k) {
          std::stringstream tmp_buf;
          tmp_buf << "Iteration:" << iter
            << ", valid_" << i + 1 << " " << name[k]
            << " : " << test_scores[k];
          if (need_output) {
            Log::Info(tmp_buf.str().c_str());
          }
          if (early_stopping_round_ > 0) {
            msg_buf << tmp_buf.str() << std::endl;
          }
        }
        if (ret.empty() && early_stopping_round_ > 0) {
          auto cur_score = valid_metrics_[i][j]->factor_to_bigger_better() * test_scores.back();
          if (cur_score > best_score_[i][j]) {
            best_score_[i][j] = cur_score;
            best_iter_[i][j] = iter;
            meet_early_stopping_pairs.emplace_back(i, j);
          } else {
            if (iter - best_iter_[i][j] >= early_stopping_round_) { ret = best_msg_[i][j]; }
          }
        }
      }
    }
  }
  for (auto& pair : meet_early_stopping_pairs) {
    best_msg_[pair.first][pair.second] = msg_buf.str();
  }
  return ret;
}

/*! \brief Get eval result */
std::vector<double> GBDT::GetEvalAt(int data_idx) const {
  CHECK(data_idx >= 0 && data_idx <= static_cast<int>(valid_score_updater_.size()));
  std::vector<double> ret;
  if (data_idx == 0) {
    for (auto& sub_metric : training_metrics_) {
      auto scores = sub_metric->Eval(train_score_updater_->score(), objective_function_);
      for (auto score : scores) {
        ret.push_back(score);
      }
    }
  } else {
    auto used_idx = data_idx - 1;
    for (size_t j = 0; j < valid_metrics_[used_idx].size(); ++j) {
      auto test_scores = valid_metrics_[used_idx][j]->Eval(valid_score_updater_[used_idx]->score(),
                                                           objective_function_);
      for (auto score : test_scores) {
        ret.push_back(score);
      }
    }
  }
  return ret;
}

/*! \brief Get training scores result */
const double* GBDT::GetTrainingScore(int64_t* out_len) {
  *out_len = static_cast<int64_t>(train_score_updater_->num_data()) * num_class_;
  return train_score_updater_->score();
}

void GBDT::GetPredictAt(int data_idx, double* out_result, int64_t* out_len) {
  CHECK(data_idx >= 0 && data_idx <= static_cast<int>(valid_score_updater_.size()));

  const double* raw_scores = nullptr;
  data_size_t num_data = 0;
  if (data_idx == 0) {
    raw_scores = GetTrainingScore(out_len);
    num_data = train_score_updater_->num_data();
  } else {
    auto used_idx = data_idx - 1;
    raw_scores = valid_score_updater_[used_idx]->score();
    num_data = valid_score_updater_[used_idx]->num_data();
    *out_len = static_cast<int64_t>(num_data) * num_class_;
  }
  if (objective_function_ != nullptr) {
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_data; ++i) {
      std::vector<double> tree_pred(num_tree_per_iteration_);
      for (int j = 0; j < num_tree_per_iteration_; ++j) {
        tree_pred[j] = raw_scores[j * num_data + i];
      }
      std::vector<double> tmp_result(num_class_);
      objective_function_->ConvertOutput(tree_pred.data(), tmp_result.data());
      for (int j = 0; j < num_class_; ++j) {
        out_result[j * num_data + i] = static_cast<double>(tmp_result[j]);
      }
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_data; ++i) {
      std::vector<double> tmp_result(num_tree_per_iteration_);
      for (int j = 0; j < num_tree_per_iteration_; ++j) {
        out_result[j * num_data + i] = static_cast<double>(raw_scores[j * num_data + i]);
      }
    }
  }
}

void GBDT::Boosting() {
  if (objective_function_ == nullptr) {
    Log::Fatal("No object function provided");
  }
  // objective function will calculate gradients and hessians
  int64_t num_score = 0;
  objective_function_->
    GetGradients(GetTrainingScore(&num_score), gradients_.data(), hessians_.data());
}

std::string GBDT::DumpModel(int num_iteration) const {
  std::stringstream str_buf;

  str_buf << "{";
  str_buf << "\"name\":\"" << SubModelName() << "\"," << std::endl;
  str_buf << "\"num_class\":" << num_class_ << "," << std::endl;
  str_buf << "\"num_tree_per_iteration\":" << num_tree_per_iteration_ << "," << std::endl;
  str_buf << "\"label_index\":" << label_idx_ << "," << std::endl;
  str_buf << "\"max_feature_idx\":" << max_feature_idx_ << "," << std::endl;

  str_buf << "\"feature_names\":[\""
    << Common::Join(feature_names_, "\",\"") << "\"],"
    << std::endl;

  str_buf << "\"tree_info\":[";
  int num_used_model = static_cast<int>(models_.size());
  if (num_iteration > 0) {
    num_iteration += boost_from_average_ ? 1 : 0;
    num_used_model = std::min(num_iteration * num_tree_per_iteration_, num_used_model);
  }
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << ",";
    }
    str_buf << "{";
    str_buf << "\"tree_index\":" << i << ",";
    str_buf << models_[i]->ToJSON();
    str_buf << "}";
  }
  str_buf << "]" << std::endl;

  str_buf << "}" << std::endl;

  return str_buf.str();
}

std::string GBDT::ModelToIfElse(int num_iteration) const {
  std::stringstream str_buf;

  str_buf << "#include \"gbdt.h\"" << std::endl;
  str_buf << "#include <LightGBM/utils/common.h>" << std::endl;
  str_buf << "#include <LightGBM/objective_function.h>" << std::endl;
  str_buf << "#include <LightGBM/metric.h>" << std::endl;
  str_buf << "#include <LightGBM/prediction_early_stop.h>" << std::endl;
  str_buf << "#include <ctime>" << std::endl;
  str_buf << "#include <sstream>" << std::endl;
  str_buf << "#include <chrono>" << std::endl;
  str_buf << "#include <string>" << std::endl;
  str_buf << "#include <vector>" << std::endl;
  str_buf << "#include <utility>" << std::endl;
  str_buf << "namespace LightGBM {" << std::endl;

  int num_used_model = static_cast<int>(models_.size());
  if (num_iteration > 0) {
    num_iteration += boost_from_average_ ? 1 : 0;
    num_used_model = std::min(num_iteration * num_tree_per_iteration_, num_used_model);
  }

  // PredictRaw
  for (int i = 0; i < num_used_model; ++i) {
    str_buf << models_[i]->ToIfElse(i, false) << std::endl;
  }

  str_buf << "double (*PredictTreePtr[])(const double*) = { ";
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << " , ";
    }
    str_buf << "PredictTree" << i;
  }
  str_buf << " };" << std::endl << std::endl;

  std::stringstream pred_str_buf;

  pred_str_buf << "\t" << "int early_stop_round_counter = 0;" << std::endl;
  pred_str_buf << "\t" << "std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);" << std::endl;
  pred_str_buf << "\t" << "for (int i = 0; i < num_iteration_for_pred_; ++i) {" << std::endl;
  pred_str_buf << "\t\t" << "for (int k = 0; k < num_tree_per_iteration_; ++k) {" << std::endl;
  pred_str_buf << "\t\t\t" << "output[k] += (*PredictTreePtr[i * num_tree_per_iteration_ + k])(features);" << std::endl;
  pred_str_buf << "\t\t" << "}" << std::endl;
  pred_str_buf << "\t\t" << "++early_stop_round_counter;" << std::endl;
  pred_str_buf << "\t\t" << "if (early_stop->round_period == early_stop_round_counter) {" << std::endl;
  pred_str_buf << "\t\t\t" << "if (early_stop->callback_function(output, num_tree_per_iteration_))" << std::endl;
  pred_str_buf << "\t\t\t\t" << "return;" << std::endl;
  pred_str_buf << "\t\t\t" << "early_stop_round_counter = 0;" << std::endl;
  pred_str_buf << "\t\t" << "}" << std::endl;
  pred_str_buf << "\t" << "}" << std::endl;

  str_buf << "void GBDT::PredictRaw(const double* features, double *output, const PredictionEarlyStopInstance* early_stop) const {" << std::endl;
  str_buf << pred_str_buf.str();
  str_buf << "}" << std::endl;
  str_buf << std::endl;

  // Predict
  str_buf << "void GBDT::Predict(const double* features, double *output, const PredictionEarlyStopInstance* early_stop) const {" << std::endl;
  str_buf << "\t" << "PredictRaw(features, output, early_stop);" << std::endl;
  str_buf << "\t" << "if (objective_function_ != nullptr) {" << std::endl;
  str_buf << "\t\t" << "objective_function_->ConvertOutput(output, output);" << std::endl;
  str_buf << "\t" << "}" << std::endl;
  str_buf << "}" << std::endl;
  str_buf << std::endl;

  // PredictLeafIndex
  for (int i = 0; i < num_used_model; ++i) {
    str_buf << models_[i]->ToIfElse(i, true) << std::endl;
  }

  str_buf << "double (*PredictTreeLeafPtr[])(const double*) = { ";
  for (int i = 0; i < num_used_model; ++i) {
    if (i > 0) {
      str_buf << " , ";
    }
    str_buf << "PredictTree" << i << "Leaf";
  }
  str_buf << " };" << std::endl << std::endl;

  str_buf << "void GBDT::PredictLeafIndex(const double* features, double *output) const {" << std::endl;
  str_buf << "\t" << "int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;" << std::endl;
  str_buf << "\t" << "for (int i = 0; i < total_tree; ++i) {" << std::endl;
  str_buf << "\t\t" << "output[i] = (*PredictTreeLeafPtr[i])(features);" << std::endl;
  str_buf << "\t" << "}" << std::endl;
  str_buf << "}" << std::endl;

  str_buf << "}  // namespace LightGBM" << std::endl;

  return str_buf.str();
}

bool GBDT::SaveModelToIfElse(int num_iteration, const char* filename) const {
  /*! \brief File to write models */
  std::ofstream output_file;
  std::ifstream ifs(filename);
  if (ifs.good()) {
    std::string origin((std::istreambuf_iterator<char>(ifs)),
                       (std::istreambuf_iterator<char>()));
    output_file.open(filename);
    output_file << "#define USE_HARD_CODE 0" << std::endl;
    output_file << "#ifndef USE_HARD_CODE" << std::endl;
    output_file << origin << std::endl;
    output_file << "#else" << std::endl;
    output_file << ModelToIfElse(num_iteration);
    output_file << "#endif" << std::endl;
  } else {
    output_file.open(filename);
    output_file << ModelToIfElse(num_iteration);
  }

  ifs.close();
  output_file.close();

  return (bool)output_file;
}

std::string GBDT::SaveModelToString(int num_iteration) const {
  std::stringstream ss;

  // output model type
  ss << SubModelName() << std::endl;
  // output number of class
  ss << "num_class=" << num_class_ << std::endl;
  ss << "num_tree_per_iteration=" << num_tree_per_iteration_ << std::endl;
  // output label index
  ss << "label_index=" << label_idx_ << std::endl;
  // output max_feature_idx
  ss << "max_feature_idx=" << max_feature_idx_ << std::endl;
  // output objective
  if (objective_function_ != nullptr) {
    ss << "objective=" << objective_function_->ToString() << std::endl;
  }

  if (boost_from_average_) {
    ss << "boost_from_average" << std::endl;
  }

  ss << "feature_names=" << Common::Join(feature_names_, " ") << std::endl;

  ss << "feature_infos=" << Common::Join(feature_infos_, " ") << std::endl;

  ss << std::endl;
  int num_used_model = static_cast<int>(models_.size());
  if (num_iteration > 0) {
    num_iteration += boost_from_average_ ? 1 : 0;
    num_used_model = std::min(num_iteration * num_tree_per_iteration_, num_used_model);
  }
  // output tree models
  for (int i = 0; i < num_used_model; ++i) {
    ss << "Tree=" << i << std::endl;
    ss << models_[i]->ToString() << std::endl;
  }

  std::vector<std::pair<size_t, std::string>> pairs = FeatureImportance();
  ss << std::endl << "feature importances:" << std::endl;
  for (size_t i = 0; i < pairs.size(); ++i) {
    ss << pairs[i].second << "=" << std::to_string(pairs[i].first) << std::endl;
  }

  return ss.str();
}

bool GBDT::SaveModelToFile(int num_iteration, const char* filename) const {
  /*! \brief File to write models */
  std::ofstream output_file;
  output_file.open(filename);

  output_file << SaveModelToString(num_iteration);

  output_file.close();

  return (bool)output_file;
}

bool GBDT::LoadModelFromString(const std::string& model_str) {
  // use serialized string to restore this object
  models_.clear();
  std::vector<std::string> lines = Common::SplitLines(model_str.c_str());

  // get number of classes
  auto line = Common::FindFromLines(lines, "num_class=");
  if (line.size() > 0) {
    Common::Atoi(Common::Split(line.c_str(), '=')[1].c_str(), &num_class_);
  } else {
    Log::Fatal("Model file doesn't specify the number of classes");
    return false;
  }

  line = Common::FindFromLines(lines, "num_tree_per_iteration=");
  if (line.size() > 0) {
    Common::Atoi(Common::Split(line.c_str(), '=')[1].c_str(), &num_tree_per_iteration_);
  } else {
    num_tree_per_iteration_ = num_class_;
  }

  // get index of label
  line = Common::FindFromLines(lines, "label_index=");
  if (line.size() > 0) {
    Common::Atoi(Common::Split(line.c_str(), '=')[1].c_str(), &label_idx_);
  } else {
    Log::Fatal("Model file doesn't specify the label index");
    return false;
  }
  // get max_feature_idx first
  line = Common::FindFromLines(lines, "max_feature_idx=");
  if (line.size() > 0) {
    Common::Atoi(Common::Split(line.c_str(), '=')[1].c_str(), &max_feature_idx_);
  } else {
    Log::Fatal("Model file doesn't specify max_feature_idx");
    return false;
  }
  // get boost_from_average_
  line = Common::FindFromLines(lines, "boost_from_average");
  if (line.size() > 0) {
    boost_from_average_ = true;
  }
  // get feature names
  line = Common::FindFromLines(lines, "feature_names=");
  if (line.size() > 0) {
    feature_names_ = Common::Split(line.substr(std::strlen("feature_names=")).c_str(), ' ');
    if (feature_names_.size() != static_cast<size_t>(max_feature_idx_ + 1)) {
      Log::Fatal("Wrong size of feature_names");
      return false;
    }
  } else {
    Log::Fatal("Model file doesn't contain feature names");
    return false;
  }

  line = Common::FindFromLines(lines, "feature_infos=");
  if (line.size() > 0) {
    feature_infos_ = Common::Split(line.substr(std::strlen("feature_infos=")).c_str(), ' ');
    if (feature_infos_.size() != static_cast<size_t>(max_feature_idx_ + 1)) {
      Log::Fatal("Wrong size of feature_infos");
      return false;
    }
  } else {
    Log::Fatal("Model file doesn't contain feature infos");
    return false;
  }

  line = Common::FindFromLines(lines, "objective=");

  if (line.size() > 0) {
    auto str = Common::Split(line.c_str(), '=')[1];
    loaded_objective_.reset(ObjectiveFunction::CreateObjectiveFunction(str));
    objective_function_ = loaded_objective_.get();
  }

  // get tree models
  size_t i = 0;
  while (i < lines.size()) {
    size_t find_pos = lines[i].find("Tree=");
    if (find_pos != std::string::npos) {
      ++i;
      int start = static_cast<int>(i);
      while (i < lines.size() && lines[i].find("Tree=") == std::string::npos) { ++i; }
      int end = static_cast<int>(i);
      std::string tree_str = Common::Join<std::string>(lines, start, end, "\n");
      models_.emplace_back(new Tree(tree_str));
    } else {
      ++i;
    }
  }
  Log::Info("Finished loading %d models", models_.size());
  num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
  num_init_iteration_ = num_iteration_for_pred_;
  iter_ = 0;

  return true;
}

std::vector<std::pair<size_t, std::string>> GBDT::FeatureImportance() const {

  std::vector<size_t> feature_importances(max_feature_idx_ + 1, 0);
  for (size_t iter = 0; iter < models_.size(); ++iter) {
    for (int split_idx = 0; split_idx < models_[iter]->num_leaves() - 1; ++split_idx) {
      if (models_[iter]->split_gain(split_idx) > 0) {
        ++feature_importances[models_[iter]->split_feature(split_idx)];
      }
    }
  }
  // store the importance first
  std::vector<std::pair<size_t, std::string>> pairs;
  for (size_t i = 0; i < feature_importances.size(); ++i) {
    if (feature_importances[i] > 0) {
      pairs.emplace_back(feature_importances[i], feature_names_[i]);
    }
  }
  // sort the importance
  std::sort(pairs.begin(), pairs.end(),
            [](const std::pair<size_t, std::string>& lhs,
               const std::pair<size_t, std::string>& rhs) {
    return lhs.first > rhs.first;
  });
  return pairs;
}

}  // namespace LightGBM
