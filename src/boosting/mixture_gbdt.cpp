/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Mixture-of-Experts GBDT extension for regime-switching models.
 */
#include "mixture_gbdt.h"

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace LightGBM {

constexpr double kEpsilon = 1e-12;

MixtureGBDT::MixtureGBDT()
    : num_experts_(4),
      train_data_(nullptr),
      objective_function_(nullptr),
      num_data_(0),
      iter_(0),
      max_feature_idx_(0),
      label_idx_(0) {
}

MixtureGBDT::~MixtureGBDT() {
}

void MixtureGBDT::Init(const Config* config, const Dataset* train_data,
                       const ObjectiveFunction* objective_function,
                       const std::vector<const Metric*>& training_metrics) {
  CHECK_NOTNULL(train_data);
  train_data_ = train_data;
  objective_function_ = objective_function;
  training_metrics_ = training_metrics;
  num_data_ = train_data_->num_data();
  iter_ = 0;

  // Store original config
  config_ = std::unique_ptr<Config>(new Config(*config));
  num_experts_ = config_->mixture_num_experts;

  // Get feature info
  max_feature_idx_ = train_data_->num_total_features() - 1;
  label_idx_ = train_data_->label_idx();
  feature_names_ = train_data_->feature_names();

  // Determine E-step loss type
  if (config_->mixture_e_step_loss == "auto") {
    // Infer from objective
    if (config_->objective == "regression_l1" || config_->objective == "l1" ||
        config_->objective == "mean_absolute_error" || config_->objective == "mae") {
      e_step_loss_type_ = "l1";
    } else if (config_->objective == "quantile") {
      e_step_loss_type_ = "quantile";
    } else {
      e_step_loss_type_ = "l2";  // default fallback
    }
  } else {
    e_step_loss_type_ = config_->mixture_e_step_loss;
  }

  Log::Info("MixtureGBDT: Initializing with %d experts, E-step loss type: %s",
            num_experts_, e_step_loss_type_.c_str());

  // Create expert config (same as original but for regression)
  expert_config_ = std::unique_ptr<Config>(new Config(*config));
  // Experts use the same objective as the mixture

  // Create gate config (multiclass classification)
  gate_config_ = std::unique_ptr<Config>(new Config(*config));
  gate_config_->objective = "multiclass";
  gate_config_->num_class = num_experts_;
  gate_config_->max_depth = config_->mixture_gate_max_depth;
  gate_config_->num_leaves = config_->mixture_gate_num_leaves;
  gate_config_->learning_rate = config_->mixture_gate_learning_rate;
  gate_config_->lambda_l2 = config_->mixture_gate_lambda_l2;

  // Initialize experts
  experts_.clear();
  experts_.reserve(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_.emplace_back(new GBDT());
    experts_[k]->Init(expert_config_.get(), train_data_, objective_function_, training_metrics_);
  }

  // Initialize gate
  // Note: Gate uses pseudo-labels, so we pass nullptr for objective
  gate_.reset(new GBDT());
  gate_->Init(gate_config_.get(), train_data_, nullptr, {});

  // Allocate buffers
  size_t nk = static_cast<size_t>(num_data_) * num_experts_;
  responsibilities_.resize(nk);
  expert_pred_.resize(nk);
  gate_proba_.resize(nk);
  yhat_.resize(num_data_);
  gradients_.resize(num_data_);
  hessians_.resize(num_data_);

  // Initialize responsibilities
  InitResponsibilities();

  Log::Info("MixtureGBDT: Initialization complete");
}

void MixtureGBDT::InitResponsibilities() {
  // Uniform initialization
  const double uniform_r = 1.0 / num_experts_;
  std::fill(responsibilities_.begin(), responsibilities_.end(), uniform_r);

  // TODO: Implement kmeans and residual_kmeans initialization
  if (config_->mixture_init != "uniform") {
    Log::Warning("MixtureGBDT: Only 'uniform' initialization is currently implemented. Using uniform.");
  }
}

void MixtureGBDT::Softmax(const double* scores, int n, double* probs) const {
  // Find max for numerical stability
  double max_score = scores[0];
  for (int i = 1; i < n; ++i) {
    if (scores[i] > max_score) max_score = scores[i];
  }

  // Compute exp and sum
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    probs[i] = std::exp(scores[i] - max_score);
    sum += probs[i];
  }

  // Normalize
  for (int i = 0; i < n; ++i) {
    probs[i] /= sum;
  }
}

double MixtureGBDT::ComputePointwiseLoss(double y, double pred) const {
  double diff = y - pred;
  if (e_step_loss_type_ == "l2") {
    return diff * diff;
  } else if (e_step_loss_type_ == "l1") {
    return std::fabs(diff);
  } else if (e_step_loss_type_ == "quantile") {
    // TODO: Get quantile alpha from config
    double alpha = 0.5;  // default median
    if (diff >= 0) {
      return alpha * diff;
    } else {
      return (alpha - 1.0) * diff;
    }
  }
  // Default to L2
  return diff * diff;
}

void MixtureGBDT::Forward() {
  // Get expert predictions
  for (int k = 0; k < num_experts_; ++k) {
    int64_t out_len;
    experts_[k]->GetPredictAt(0, expert_pred_.data() + k * num_data_, &out_len);
  }

  // Get gate probabilities (softmax of gate raw predictions)
  std::vector<double> gate_raw(static_cast<size_t>(num_data_) * num_experts_);
  int64_t out_len;
  gate_->GetPredictAt(0, gate_raw.data(), &out_len);

  // Apply softmax per sample
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    Softmax(gate_raw.data() + i * num_experts_, num_experts_,
            gate_proba_.data() + i * num_experts_);
  }

  // Compute combined prediction: yhat[i] = sum_k gate_proba[i,k] * expert_pred[i,k]
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    double sum = 0.0;
    for (int k = 0; k < num_experts_; ++k) {
      sum += gate_proba_[i * num_experts_ + k] * expert_pred_[k * num_data_ + i];
    }
    yhat_[i] = sum;
  }
}

void MixtureGBDT::EStep() {
  const label_t* labels = train_data_->metadata().label();
  const double alpha = config_->mixture_e_step_alpha;
  const double r_min = config_->mixture_r_min;

  std::vector<double> scores(num_experts_);

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) private(scores)
  for (data_size_t i = 0; i < num_data_; ++i) {
    // Compute scores: s_ik = log(gate_proba_ik + eps) - alpha * loss(y_i, expert_pred_ik)
    for (int k = 0; k < num_experts_; ++k) {
      double gate_prob = gate_proba_[i * num_experts_ + k];
      double expert_p = expert_pred_[k * num_data_ + i];
      double loss = ComputePointwiseLoss(labels[i], expert_p);
      scores[k] = std::log(gate_prob + kEpsilon) - alpha * loss;
    }

    // Apply softmax to get responsibilities
    Softmax(scores.data(), num_experts_, responsibilities_.data() + i * num_experts_);

    // Clip to r_min and renormalize
    double sum = 0.0;
    for (int k = 0; k < num_experts_; ++k) {
      size_t idx = i * num_experts_ + k;
      if (responsibilities_[idx] < r_min) {
        responsibilities_[idx] = r_min;
      }
      sum += responsibilities_[idx];
    }
    for (int k = 0; k < num_experts_; ++k) {
      responsibilities_[i * num_experts_ + k] /= sum;
    }
  }
}

void MixtureGBDT::SmoothResponsibilities() {
  if (config_->mixture_r_smoothing != "ema") {
    return;
  }

  const double lambda = config_->mixture_r_ema_lambda;
  if (lambda <= 0.0) {
    return;
  }

  // Apply EMA in row order (assumed to be time order)
  // r[i] = (1-lambda)*r[i] + lambda*r[i-1]
  for (data_size_t i = 1; i < num_data_; ++i) {
    double sum = 0.0;
    for (int k = 0; k < num_experts_; ++k) {
      size_t idx = i * num_experts_ + k;
      size_t prev_idx = (i - 1) * num_experts_ + k;
      responsibilities_[idx] = (1.0 - lambda) * responsibilities_[idx] +
                               lambda * responsibilities_[prev_idx];
      sum += responsibilities_[idx];
    }
    // Renormalize
    for (int k = 0; k < num_experts_; ++k) {
      responsibilities_[i * num_experts_ + k] /= sum;
    }
  }
}

void MixtureGBDT::MStepExperts() {
  const label_t* labels = train_data_->metadata().label();

  // Compute gradients and hessians for the mixture loss
  // Using the combined prediction yhat
  if (objective_function_ != nullptr) {
    objective_function_->GetGradients(yhat_.data(), gradients_.data(), hessians_.data());
  } else {
    // Default to MSE if no objective
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i) {
      double diff = yhat_[i] - labels[i];
      gradients_[i] = 2.0 * diff;
      hessians_[i] = 2.0;
    }
  }

  // Train each expert with responsibility-weighted gradients
  for (int k = 0; k < num_experts_; ++k) {
    std::vector<score_t> grad_k(num_data_);
    std::vector<score_t> hess_k(num_data_);

    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i) {
      double r_ik = responsibilities_[i * num_experts_ + k];
      grad_k[i] = static_cast<score_t>(r_ik * gradients_[i]);
      hess_k[i] = static_cast<score_t>(r_ik * hessians_[i]);
    }

    // Train one iteration with custom gradients
    experts_[k]->TrainOneIter(grad_k.data(), hess_k.data());
  }
}

void MixtureGBDT::MStepGate() {
  // Create pseudo-labels: z_i = argmax_k r_ik
  std::vector<label_t> pseudo_labels(num_data_);

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    int best_k = 0;
    double best_r = responsibilities_[i * num_experts_];
    for (int k = 1; k < num_experts_; ++k) {
      double r = responsibilities_[i * num_experts_ + k];
      if (r > best_r) {
        best_r = r;
        best_k = k;
      }
    }
    pseudo_labels[i] = static_cast<label_t>(best_k);
  }

  // Update gate's labels
  // Note: This requires modifying the dataset's labels, which is complex.
  // For now, we'll use the gate's TrainOneIter with custom gradients.
  // TODO: Implement proper label update for gate training

  // For multiclass, we compute softmax cross-entropy gradients
  std::vector<score_t> gate_grad(static_cast<size_t>(num_data_) * num_experts_);
  std::vector<score_t> gate_hess(static_cast<size_t>(num_data_) * num_experts_);

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    int label = static_cast<int>(pseudo_labels[i]);
    for (int k = 0; k < num_experts_; ++k) {
      size_t idx = i + k * num_data_;  // Gate uses class-major order
      double p = gate_proba_[i * num_experts_ + k];
      if (k == label) {
        gate_grad[idx] = static_cast<score_t>(p - 1.0);
      } else {
        gate_grad[idx] = static_cast<score_t>(p);
      }
      // Hessian for softmax cross-entropy: p * (1 - p)
      gate_hess[idx] = static_cast<score_t>(std::max(p * (1.0 - p), kEpsilon));
    }
  }

  // Train gate for specified iterations
  for (int g = 0; g < config_->mixture_gate_iters_per_round; ++g) {
    gate_->TrainOneIter(gate_grad.data(), gate_hess.data());
  }
}

bool MixtureGBDT::TrainOneIter(const score_t* gradients, const score_t* hessians) {
  // MixtureGBDT ignores external gradients/hessians
  // (custom objective is handled internally via responsibility weighting)
  (void)gradients;
  (void)hessians;

  // Forward pass
  Forward();

  // E-step: update responsibilities
  EStep();

  // Apply time-series smoothing if enabled
  SmoothResponsibilities();

  // M-step: update experts
  MStepExperts();

  // M-step: update gate
  MStepGate();

  ++iter_;

  // Check if we should continue
  // For now, always continue
  return false;
}

void MixtureGBDT::Train(int snapshot_freq, const std::string& model_output_path) {
  auto start_time = std::chrono::steady_clock::now();
  for (int iter = 0; iter < config_->num_iterations; ++iter) {
    TrainOneIter(nullptr, nullptr);

    auto end_time = std::chrono::steady_clock::now();
    Log::Info("MixtureGBDT: %f seconds elapsed, finished iteration %d",
              std::chrono::duration<double, std::milli>(end_time - start_time).count() * 1e-3,
              iter + 1);

    if (snapshot_freq > 0 && (iter + 1) % snapshot_freq == 0) {
      std::string snapshot_out = model_output_path + ".snapshot_iter_" + std::to_string(iter + 1);
      SaveModelToFile(0, -1, 0, snapshot_out.c_str());
    }
  }
}

int MixtureGBDT::GetCurrentIteration() const {
  return iter_;
}

void MixtureGBDT::RollbackOneIter() {
  if (iter_ > 0) {
    for (int k = 0; k < num_experts_; ++k) {
      experts_[k]->RollbackOneIter();
    }
    gate_->RollbackOneIter();
    --iter_;
  }
}

const double* MixtureGBDT::GetTrainingScore(int64_t* out_len) {
  *out_len = num_data_;
  return yhat_.data();
}

std::vector<double> MixtureGBDT::GetEvalAt(int data_idx) const {
  // TODO: Implement proper evaluation
  std::vector<double> result;
  return result;
}

int64_t MixtureGBDT::GetNumPredictAt(int data_idx) const {
  if (data_idx == 0) {
    return num_data_;
  }
  // TODO: Handle validation data
  return 0;
}

void MixtureGBDT::GetPredictAt(int data_idx, double* result, int64_t* out_len) {
  if (data_idx == 0) {
    std::copy(yhat_.begin(), yhat_.end(), result);
    *out_len = num_data_;
  }
}

int MixtureGBDT::NumPredictOneRow(int start_iteration, int num_iteration,
                                   bool is_pred_leaf, bool is_pred_contrib) const {
  (void)start_iteration;
  (void)num_iteration;
  (void)is_pred_leaf;
  (void)is_pred_contrib;
  // Return 1 for the combined prediction
  return 1;
}

void MixtureGBDT::Predict(const double* features, double* output,
                          const PredictionEarlyStopInstance* earlyStop) const {
  (void)earlyStop;

  // Get expert predictions
  std::vector<double> expert_preds(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->Predict(features, &expert_preds[k], nullptr);
  }

  // Get gate probabilities
  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), nullptr);

  std::vector<double> gate_prob(num_experts_);
  Softmax(gate_raw.data(), num_experts_, gate_prob.data());

  // Compute weighted sum
  double sum = 0.0;
  for (int k = 0; k < num_experts_; ++k) {
    sum += gate_prob[k] * expert_preds[k];
  }
  *output = sum;
}

void MixtureGBDT::PredictRaw(const double* features, double* output,
                             const PredictionEarlyStopInstance* earlyStop) const {
  Predict(features, output, earlyStop);
}

void MixtureGBDT::PredictByMap(const std::unordered_map<int, double>& features, double* output,
                               const PredictionEarlyStopInstance* early_stop) const {
  (void)early_stop;

  std::vector<double> expert_preds(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->PredictByMap(features, &expert_preds[k], nullptr);
  }

  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRawByMap(features, gate_raw.data(), nullptr);

  std::vector<double> gate_prob(num_experts_);
  Softmax(gate_raw.data(), num_experts_, gate_prob.data());

  double sum = 0.0;
  for (int k = 0; k < num_experts_; ++k) {
    sum += gate_prob[k] * expert_preds[k];
  }
  *output = sum;
}

void MixtureGBDT::PredictRawByMap(const std::unordered_map<int, double>& features, double* output,
                                  const PredictionEarlyStopInstance* early_stop) const {
  PredictByMap(features, output, early_stop);
}

void MixtureGBDT::PredictRegime(const double* features, int* output) const {
  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), nullptr);

  std::vector<double> gate_prob(num_experts_);
  Softmax(gate_raw.data(), num_experts_, gate_prob.data());

  // Find argmax
  int best_k = 0;
  double best_p = gate_prob[0];
  for (int k = 1; k < num_experts_; ++k) {
    if (gate_prob[k] > best_p) {
      best_p = gate_prob[k];
      best_k = k;
    }
  }
  *output = best_k;
}

void MixtureGBDT::PredictRegimeProba(const double* features, double* output) const {
  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), nullptr);
  Softmax(gate_raw.data(), num_experts_, output);
}

void MixtureGBDT::PredictExpertPred(const double* features, double* output) const {
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->Predict(features, &output[k], nullptr);
  }
}

void MixtureGBDT::PredictLeafIndex(const double* features, double* output) const {
  // Return leaf indices from all experts and gate
  // TODO: Implement properly
  (void)features;
  (void)output;
}

void MixtureGBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features,
                                        double* output) const {
  (void)features;
  (void)output;
}

void MixtureGBDT::PredictContrib(const double* features, double* output) const {
  // TODO: Implement SHAP for mixture model
  (void)features;
  (void)output;
}

void MixtureGBDT::PredictContribByMap(const std::unordered_map<int, double>& features,
                                       std::vector<std::unordered_map<int, double>>* output) const {
  (void)features;
  (void)output;
}

void MixtureGBDT::MergeFrom(const Boosting* other) {
  // TODO: Implement merge
  (void)other;
  Log::Fatal("MixtureGBDT::MergeFrom is not implemented");
}

void MixtureGBDT::ShuffleModels(int start_iter, int end_iter) {
  (void)start_iter;
  (void)end_iter;
  Log::Warning("MixtureGBDT::ShuffleModels is not supported");
}

void MixtureGBDT::ResetTrainingData(const Dataset* train_data,
                                    const ObjectiveFunction* objective_function,
                                    const std::vector<const Metric*>& training_metrics) {
  train_data_ = train_data;
  objective_function_ = objective_function;
  training_metrics_ = training_metrics;
  num_data_ = train_data_->num_data();

  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->ResetTrainingData(train_data_, objective_function_, training_metrics_);
  }
  gate_->ResetTrainingData(train_data_, nullptr, {});

  // Reallocate buffers
  size_t nk = static_cast<size_t>(num_data_) * num_experts_;
  responsibilities_.resize(nk);
  expert_pred_.resize(nk);
  gate_proba_.resize(nk);
  yhat_.resize(num_data_);
  gradients_.resize(num_data_);
  hessians_.resize(num_data_);

  InitResponsibilities();
}

void MixtureGBDT::ResetConfig(const Config* config) {
  config_ = std::unique_ptr<Config>(new Config(*config));
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->ResetConfig(config);
  }
  gate_->ResetConfig(gate_config_.get());
}

void MixtureGBDT::AddValidDataset(const Dataset* valid_data,
                                  const std::vector<const Metric*>& valid_metrics) {
  // TODO: Implement validation
  (void)valid_data;
  (void)valid_metrics;
}

void MixtureGBDT::RefitTree(const int* tree_leaf_prediction, const size_t nrow, const size_t ncol) {
  (void)tree_leaf_prediction;
  (void)nrow;
  (void)ncol;
  Log::Fatal("MixtureGBDT::RefitTree is not implemented");
}

std::string MixtureGBDT::DumpModel(int start_iteration, int num_iteration,
                                   int feature_importance_type) const {
  (void)start_iteration;
  (void)num_iteration;
  (void)feature_importance_type;
  // TODO: Implement JSON dump
  return "{}";
}

std::string MixtureGBDT::ModelToIfElse(int num_iteration) const {
  (void)num_iteration;
  Log::Fatal("MixtureGBDT::ModelToIfElse is not implemented");
  return "";
}

bool MixtureGBDT::SaveModelToIfElse(int num_iteration, const char* filename) const {
  (void)num_iteration;
  (void)filename;
  Log::Fatal("MixtureGBDT::SaveModelToIfElse is not implemented");
  return false;
}

bool MixtureGBDT::SaveModelToFile(int start_iteration, int num_iterations,
                                   int feature_importance_type, const char* filename) const {
  std::string model_str = SaveModelToString(start_iteration, num_iterations, feature_importance_type);
  if (model_str.empty()) {
    return false;
  }
  std::ofstream file(filename);
  if (!file.is_open()) {
    return false;
  }
  file << model_str;
  return true;
}

std::string MixtureGBDT::SaveModelToString(int start_iteration, int num_iterations,
                                            int feature_importance_type) const {
  std::stringstream ss;

  // Mixture header
  ss << "mixture\n";
  ss << "mixture_enable=1\n";
  ss << "mixture_num_experts=" << num_experts_ << "\n";
  ss << "mixture_e_step_alpha=" << config_->mixture_e_step_alpha << "\n";
  ss << "mixture_e_step_loss=" << e_step_loss_type_ << "\n";
  ss << "mixture_r_smoothing=" << config_->mixture_r_smoothing << "\n";
  ss << "mixture_r_ema_lambda=" << config_->mixture_r_ema_lambda << "\n";
  ss << "\n";

  // Gate model
  ss << "[gate_model]\n";
  ss << gate_->SaveModelToString(start_iteration, num_iterations, feature_importance_type);
  ss << "\n";

  // Expert models
  for (int k = 0; k < num_experts_; ++k) {
    ss << "[expert_model_" << k << "]\n";
    ss << experts_[k]->SaveModelToString(start_iteration, num_iterations, feature_importance_type);
    ss << "\n";
  }

  return ss.str();
}

bool MixtureGBDT::LoadModelFromString(const char* buffer, size_t len) {
  // TODO: Implement proper parsing
  (void)buffer;
  (void)len;
  Log::Fatal("MixtureGBDT::LoadModelFromString is not fully implemented");
  return false;
}

std::vector<double> MixtureGBDT::FeatureImportance(int num_iteration, int importance_type) const {
  // Sum importance across all experts
  std::vector<double> result(max_feature_idx_ + 1, 0.0);

  for (int k = 0; k < num_experts_; ++k) {
    auto expert_imp = experts_[k]->FeatureImportance(num_iteration, importance_type);
    for (size_t i = 0; i < expert_imp.size() && i < result.size(); ++i) {
      result[i] += expert_imp[i];
    }
  }

  return result;
}

double MixtureGBDT::GetUpperBoundValue() const {
  double max_val = -std::numeric_limits<double>::infinity();
  for (int k = 0; k < num_experts_; ++k) {
    max_val = std::max(max_val, experts_[k]->GetUpperBoundValue());
  }
  return max_val;
}

double MixtureGBDT::GetLowerBoundValue() const {
  double min_val = std::numeric_limits<double>::infinity();
  for (int k = 0; k < num_experts_; ++k) {
    min_val = std::min(min_val, experts_[k]->GetLowerBoundValue());
  }
  return min_val;
}

int MixtureGBDT::MaxFeatureIdx() const {
  return max_feature_idx_;
}

std::vector<std::string> MixtureGBDT::FeatureNames() const {
  return feature_names_;
}

int MixtureGBDT::LabelIdx() const {
  return label_idx_;
}

int MixtureGBDT::NumberOfTotalModel() const {
  int total = 0;
  for (int k = 0; k < num_experts_; ++k) {
    total += experts_[k]->NumberOfTotalModel();
  }
  total += gate_->NumberOfTotalModel();
  return total;
}

int MixtureGBDT::NumModelPerIteration() const {
  // One tree per expert per iteration, plus gate trees
  return num_experts_ + gate_->NumModelPerIteration();
}

int MixtureGBDT::NumberOfClasses() const {
  return 1;  // Regression output
}

bool MixtureGBDT::NeedAccuratePrediction() const {
  return true;
}

void MixtureGBDT::InitPredict(int start_iteration, int num_iteration, bool is_pred_contrib) {
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->InitPredict(start_iteration, num_iteration, is_pred_contrib);
  }
  gate_->InitPredict(start_iteration, num_iteration, is_pred_contrib);
}

double MixtureGBDT::GetLeafValue(int tree_idx, int leaf_idx) const {
  // Determine which model and which tree
  // TODO: Implement properly
  (void)tree_idx;
  (void)leaf_idx;
  return 0.0;
}

void MixtureGBDT::SetLeafValue(int tree_idx, int leaf_idx, double val) {
  (void)tree_idx;
  (void)leaf_idx;
  (void)val;
}

std::string MixtureGBDT::GetLoadedParam() const {
  return loaded_parameter_;
}

std::string MixtureGBDT::ParserConfigStr() const {
  if (!experts_.empty()) {
    return experts_[0]->ParserConfigStr();
  }
  return "";
}

}  // namespace LightGBM
