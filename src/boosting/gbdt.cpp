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
std::chrono::duration<double, std::milli> tree_time;
#endif // TIMETAG

GBDT::GBDT() : iter_(0),
    train_data_(nullptr),
    objective_function_(nullptr),
    early_stopping_round_(0),
    max_feature_idx_(0),
    num_tree_per_iteration_(1),
    num_class_(1),
    num_iteration_for_pred_(0),
    shrinkage_rate_(0.1f),
    num_init_iteration_(0),
    need_re_bagging_(false) {

  #pragma omp parallel
  #pragma omp master
  {
    num_threads_ = omp_get_num_threads();
  }
  average_output_ = false;
  tree_learner_ = nullptr;
}

GBDT::~GBDT() {
  #ifdef TIMETAG
  Log::Info("GBDT::boosting costs %f", boosting_time * 1e-3);
  Log::Info("GBDT::train_score costs %f", train_score_time * 1e-3);
  Log::Info("GBDT::out_of_bag_score costs %f", out_of_bag_score_time * 1e-3);
  Log::Info("GBDT::valid_score costs %f", valid_score_time * 1e-3);
  Log::Info("GBDT::metric costs %f", metric_time * 1e-3);
  Log::Info("GBDT::bagging costs %f", bagging_time * 1e-3);
  Log::Info("GBDT::tree costs %f", tree_time * 1e-3);
  #endif
}

void GBDT::Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
                const std::vector<const Metric*>& training_metrics) {
  CHECK(train_data != nullptr);
  CHECK(train_data->num_features() > 0);
  train_data_ = train_data;
  iter_ = 0;
  num_iteration_for_pred_ = 0;
  max_feature_idx_ = 0;
  num_class_ = config->num_class;
  gbdt_config_ = std::unique_ptr<BoostingConfig>(new BoostingConfig(*config));
  early_stopping_round_ = gbdt_config_->early_stopping_round;
  shrinkage_rate_ = gbdt_config_->learning_rate;

  objective_function_ = objective_function;
  num_tree_per_iteration_ = num_class_;
  if (objective_function_ != nullptr) {
    is_constant_hessian_ = objective_function_->IsConstantHessian();
    num_tree_per_iteration_ = objective_function_->NumModelPerIteration();
  } else {
    is_constant_hessian_ = false;
  }

  tree_learner_ = std::unique_ptr<TreeLearner>(TreeLearner::CreateTreeLearner(gbdt_config_->tree_learner_type, gbdt_config_->device_type, &gbdt_config_->tree_config));

  // init tree learner
  tree_learner_->Init(train_data_, is_constant_hessian_);

  // push training metrics
  training_metrics_.clear();
  for (const auto& metric : training_metrics) {
    training_metrics_.push_back(metric);
  }
  training_metrics_.shrink_to_fit();

  train_score_updater_.reset(new ScoreUpdater(train_data_, num_tree_per_iteration_));

  num_data_ = train_data_->num_data();
  // create buffer for gradients and hessians
  if (objective_function_ != nullptr) {
    size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
    gradients_.resize(total_size);
    hessians_.resize(total_size);
  }
  // get max feature index
  max_feature_idx_ = train_data_->num_total_features() - 1;
  // get label index
  label_idx_ = train_data_->label_idx();
  // get feature names
  feature_names_ = train_data_->feature_names();
  feature_infos_ = train_data_->feature_infos();

  // if need bagging, create buffer
  ResetBaggingConfig(gbdt_config_.get(), true);

  // reset config for tree learner
  class_need_train_ = std::vector<bool>(num_tree_per_iteration_, true);
  if (objective_function_ != nullptr && objective_function_->SkipEmptyClass()) {
    CHECK(num_tree_per_iteration_ == num_class_);

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

void GBDT::Boosting() {
  if (objective_function_ == nullptr) {
    Log::Fatal("No object function provided");
  }
  // objective function will calculate gradients and hessians
  int64_t num_score = 0;
  objective_function_->
    GetGradients(GetTrainingScore(&num_score), gradients_.data(), hessians_.data());
}

data_size_t GBDT::BaggingHelper(Random& cur_rand, data_size_t start, data_size_t cnt, data_size_t* buffer) {
  if (cnt <= 0) {
    return 0;
  }
  data_size_t bag_data_cnt = static_cast<data_size_t>(gbdt_config_->bagging_fraction * cnt);
  data_size_t cur_left_cnt = 0;
  data_size_t cur_right_cnt = 0;
  auto right_buffer = buffer + bag_data_cnt;
  // random bagging, minimal unit is one record
  for (data_size_t i = 0; i < cnt; ++i) {
    float prob = (bag_data_cnt - cur_left_cnt) / static_cast<float>(cnt - i);
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
  if ((bag_data_cnt_ < num_data_ && iter % gbdt_config_->bagging_freq == 0) 
      || need_re_bagging_) {
    need_re_bagging_ = false;
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

/* If the custom "average" is implemented it will be used inplace of the label average (if enabled)
*
* An improvement to this is to have options to explicitly choose
* (i) standard average
* (ii) custom average if available
* (iii) any user defined scalar bias (e.g. using a new option "init_score" that overrides (i) and (ii) )
*
* (i) and (ii) could be selected as say "auto_init_score" = 0 or 1 etc..
*
*/
double ObtainAutomaticInitialScore(const ObjectiveFunction* fobj, const float* label, data_size_t num_data) {
  double init_score = 0.0f;
  bool got_custom = false;
  if (fobj != nullptr) {
    got_custom = fobj->GetCustomAverage(&init_score);
  }
  if (!got_custom) {
    double sum_label = 0.0f;
    #pragma omp parallel for schedule(static) reduction(+:sum_label)
    for (data_size_t i = 0; i < num_data; ++i) {
      sum_label += label[i];
    }
    init_score = sum_label / num_data;
  }
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

void GBDT::Train(int snapshot_freq, const std::string& model_output_path) {
  bool is_finished = false;
  auto start_time = std::chrono::steady_clock::now();
  for (int iter = 0; iter < gbdt_config_->num_iterations && !is_finished; ++iter) {
    is_finished = TrainOneIter(nullptr, nullptr);
    if (!is_finished) {
      is_finished = EvalAndCheckEarlyStopping();
    }
    auto end_time = std::chrono::steady_clock::now();
    // output used time per iteration
    Log::Info("%f seconds elapsed, finished iteration %d", std::chrono::duration<double,
              std::milli>(end_time - start_time) * 1e-3, iter + 1);
    if (snapshot_freq > 0
        && (iter + 1) % snapshot_freq == 0) {
      std::string snapshot_out = model_output_path + ".snapshot_iter_" + std::to_string(iter + 1);
      SaveModelToFile(-1, snapshot_out.c_str());
    }
  }
  SaveModelToFile(-1, model_output_path.c_str());
}

double GBDT::BoostFromAverage() {
  // boosting from average label; or customized "average" if implemented for the current objective
  if (models_.empty()
      && gbdt_config_->boost_from_average
      && !train_score_updater_->has_init_score()
      && num_class_ <= 1
      && objective_function_ != nullptr
      && objective_function_->BoostFromAverage()) {

    auto label = train_data_->metadata().label();
    double init_score = ObtainAutomaticInitialScore(objective_function_, label, num_data_);
    if (std::fabs(init_score) > kEpsilon) {
      train_score_updater_->AddScore(init_score, 0);
      for (auto& score_updater : valid_score_updater_) {
        score_updater->AddScore(init_score, 0);
      }
      return init_score;
    }
  }
  return 0.0f;
}

bool GBDT::TrainOneIter(const score_t* gradients, const score_t* hessians) {
  auto init_score = BoostFromAverage();
  // boosting first
  if (gradients == nullptr || hessians == nullptr) {

    #ifdef TIMETAG
    auto start_time = std::chrono::steady_clock::now();
    #endif

    Boosting();
    gradients = gradients_.data();
    hessians = hessians_.data();

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

  bool should_continue = false;
  for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {

    #ifdef TIMETAG
    start_time = std::chrono::steady_clock::now();
    #endif

    std::unique_ptr<Tree> new_tree(new Tree(2));
    if (class_need_train_[cur_tree_id]) {
      size_t bias = static_cast<size_t>(cur_tree_id)* num_data_;
      auto grad = gradients + bias;
      auto hess = hessians + bias;

      // need to copy gradients for bagging subset.
      if (is_use_subset_ && bag_data_cnt_ < num_data_) {
        for (int i = 0; i < bag_data_cnt_; ++i) {
          gradients_[bias + i] = grad[bag_data_indices_[i]];
          hessians_[bias + i] = hess[bag_data_indices_[i]];
        }
        grad = gradients_.data() + bias;
        hess = hessians_.data() + bias;
      }

      new_tree.reset(tree_learner_->Train(grad, hess, is_constant_hessian_));
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
      if (std::fabs(init_score) > kEpsilon) {
        new_tree->AddBias(init_score);
      }
    } else {
      // only add default score one-time
      if (!class_need_train_[cur_tree_id] && models_.size() < static_cast<size_t>(num_tree_per_iteration_)) {
        auto output = class_default_output_[cur_tree_id];
        new_tree->AsConstantTree(output);
        // updates scores
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
  return false;
}

void GBDT::RollbackOneIter() {
  if (iter_ <= 0) { return; }
  // reset score
  for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
    auto curr_tree = models_.size() - num_tree_per_iteration_ + cur_tree_id;
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

    #ifdef TIMETAG
    train_score_time += std::chrono::steady_clock::now() - start_time;
    #endif

    #ifdef TIMETAG
    start_time = std::chrono::steady_clock::now();
    #endif

    // we need to predict out-of-bag scores of data for boosting
    if (num_data_ - bag_data_cnt_ > 0) {
      train_score_updater_->AddScore(tree, bag_data_indices_.data() + bag_data_cnt_, num_data_ - bag_data_cnt_, cur_tree_id);
    }

    #ifdef TIMETAG
    out_of_bag_score_time += std::chrono::steady_clock::now() - start_time;
    #endif

  } else {
    train_score_updater_->AddScore(tree, cur_tree_id);

    #ifdef TIMETAG
    train_score_time += std::chrono::steady_clock::now() - start_time;
    #endif
  }


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

std::vector<double> GBDT::EvalOneMetric(const Metric* metric, const double* score) const {
  return metric->Eval(score, objective_function_);
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
      auto scores = EvalOneMetric(sub_metric, train_score_updater_->score());
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
        auto test_scores = EvalOneMetric(valid_metrics_[i][j], valid_score_updater_[i]->score());
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
      auto scores = EvalOneMetric(sub_metric, train_score_updater_->score());
      for (auto score : scores) {
        ret.push_back(score);
      }
    }
  } else {
    auto used_idx = data_idx - 1;
    for (size_t j = 0; j < valid_metrics_[used_idx].size(); ++j) {
      auto test_scores = EvalOneMetric(valid_metrics_[used_idx][j], valid_score_updater_[used_idx]->score());
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

void GBDT::PredictContrib(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  int early_stop_round_counter = 0;
  // set zero
  const int num_features = max_feature_idx_ + 1;
  std::memset(output, 0, sizeof(double) * num_tree_per_iteration_ * (num_features + 1));
  for (int i = 0; i < num_iteration_for_pred_; ++i) {
    // predict all the trees for one iteration
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      models_[i * num_tree_per_iteration_ + k]->PredictContrib(features, num_features, output + k*(num_features + 1));
    }
    // check early stopping
    ++early_stop_round_counter;
    if (early_stop->round_period == early_stop_round_counter) {
      if (early_stop->callback_function(output, num_tree_per_iteration_)) {
        return;
      }
      early_stop_round_counter = 0;
    }
  }
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
  if (objective_function_ != nullptr && !average_output_) {
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

void GBDT::ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
                             const std::vector<const Metric*>& training_metrics) {

  if (train_data != train_data_ && !train_data_->CheckAlign(*train_data)) {
    Log::Fatal("cannot reset training data, since new training data has different bin mappers");
  }

  objective_function_ = objective_function;
  if (objective_function_ != nullptr) {
    is_constant_hessian_ = objective_function_->IsConstantHessian();
    CHECK(num_tree_per_iteration_ == objective_function_->NumModelPerIteration());
  } else {
    is_constant_hessian_ = false;
  }

  // push training metrics
  training_metrics_.clear();
  for (const auto& metric : training_metrics) {
    training_metrics_.push_back(metric);
  }
  training_metrics_.shrink_to_fit();

  if (train_data != train_data_) {
    train_data_ = train_data;
    // not same training data, need reset score and others
    // create score tracker
    train_score_updater_.reset(new ScoreUpdater(train_data_, num_tree_per_iteration_));

    // update score
    for (int i = 0; i < iter_; ++i) {
      for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
        auto curr_tree = (i + num_init_iteration_) * num_tree_per_iteration_ + cur_tree_id;
        train_score_updater_->AddScore(models_[curr_tree].get(), cur_tree_id);
      }
    }

    num_data_ = train_data_->num_data();

    // create buffer for gradients and hessians
    if (objective_function_ != nullptr) {
      size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
      gradients_.resize(total_size);
      hessians_.resize(total_size);
    }

    max_feature_idx_ = train_data_->num_total_features() - 1;
    label_idx_ = train_data_->label_idx();
    feature_names_ = train_data_->feature_names();
    feature_infos_ = train_data_->feature_infos();

    tree_learner_->ResetTrainingData(train_data);
    ResetBaggingConfig(gbdt_config_.get(), true);
  }
}

void GBDT::ResetConfig(const BoostingConfig* config) {
  auto new_config = std::unique_ptr<BoostingConfig>(new BoostingConfig(*config));
  early_stopping_round_ = new_config->early_stopping_round;
  shrinkage_rate_ = new_config->learning_rate;
  if (tree_learner_ != nullptr) {
    tree_learner_->ResetConfig(&new_config->tree_config);
  }
  if (train_data_ != nullptr) {
    ResetBaggingConfig(new_config.get(), false);
  }
  gbdt_config_.reset(new_config.release());
}

void GBDT::ResetBaggingConfig(const BoostingConfig* config, bool is_change_dataset) {
  // if need bagging, create buffer
  if (config->bagging_fraction < 1.0 && config->bagging_freq > 0) {
    bag_data_cnt_ =
      static_cast<data_size_t>(config->bagging_fraction * num_data_);
    bag_data_indices_.resize(num_data_);
    tmp_indices_.resize(num_data_);

    offsets_buf_.resize(num_threads_);
    left_cnts_buf_.resize(num_threads_);
    right_cnts_buf_.resize(num_threads_);
    left_write_pos_buf_.resize(num_threads_);
    right_write_pos_buf_.resize(num_threads_);

    double average_bag_rate = config->bagging_fraction / config->bagging_freq;
    int sparse_group = 0;
    for (int i = 0; i < train_data_->num_feature_groups(); ++i) {
      if (train_data_->FeatureGroupIsSparse(i)) {
        ++sparse_group;
      }
    }
    is_use_subset_ = false;
    const int group_threshold_usesubset = 100;
    const int sparse_group_threshold_usesubset = train_data_->num_feature_groups() / 4;
    if (average_bag_rate <= 0.5
        && (train_data_->num_feature_groups() < group_threshold_usesubset || sparse_group < sparse_group_threshold_usesubset)) {
      if (tmp_subset_ == nullptr || is_change_dataset) {
        tmp_subset_.reset(new Dataset(bag_data_cnt_));
        tmp_subset_->CopyFeatureMapperFrom(train_data_);
      }
      is_use_subset_ = true;
      Log::Debug("use subset for bagging");
    }

    if (is_change_dataset) {
      need_re_bagging_ = true;
    }

    if (is_use_subset_ && bag_data_cnt_ < num_data_) {
      if (objective_function_ == nullptr) {
        size_t total_size = static_cast<size_t>(num_data_) * num_tree_per_iteration_;
        gradients_.resize(total_size);
        hessians_.resize(total_size);
      }
    }
  } else {
    bag_data_cnt_ = num_data_;
    bag_data_indices_.clear();
    tmp_indices_.clear();
    is_use_subset_ = false;
  }
}

}  // namespace LightGBM
