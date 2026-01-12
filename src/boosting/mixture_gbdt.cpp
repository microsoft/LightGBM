/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Mixture-of-Experts GBDT extension for regime-switching models.
 */
#include "mixture_gbdt.h"

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace LightGBM {

constexpr double kMixtureEpsilon = 1e-12;

MixtureGBDT::MixtureGBDT()
    : num_experts_(4),
      train_data_(nullptr),
      objective_function_(nullptr),
      num_data_(0),
      iter_(0),
      max_feature_idx_(0),
      label_idx_(0),
      use_markov_(false),
      use_momentum_(false),
      num_original_features_(0) {
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
  // Note: We pass nullptr for objective_function because we use custom gradients
  // (responsibility-weighted) in MStepExperts. The main objective is stored in
  // objective_function_ and used to compute gradients on yhat.
  Log::Debug("MixtureGBDT::Init - creating %d experts", num_experts_);
  experts_.clear();
  experts_.reserve(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    Log::Debug("MixtureGBDT::Init - creating expert %d", k);
    experts_.emplace_back(new GBDT());
    Log::Debug("MixtureGBDT::Init - initializing expert %d", k);
    experts_[k]->Init(expert_config_.get(), train_data_, nullptr, {});
    Log::Debug("MixtureGBDT::Init - expert %d initialized", k);
  }

  // Check smoothing modes
  use_markov_ = (config_->mixture_r_smoothing == "markov");
  use_momentum_ = (config_->mixture_r_smoothing == "momentum");
  num_original_features_ = train_data_->num_total_features();

  // Initialize gate
  // Note: Gate uses pseudo-labels, so we pass nullptr for objective
  Log::Debug("MixtureGBDT::Init - creating gate");
  gate_.reset(new GBDT());
  Log::Debug("MixtureGBDT::Init - initializing gate");

  if (use_markov_) {
    Log::Info("MixtureGBDT: Markov mode enabled (lambda=%.2f)",
              config_->mixture_smoothing_lambda);
  } else if (use_momentum_) {
    Log::Info("MixtureGBDT: Momentum mode enabled (lambda=%.2f)",
              config_->mixture_smoothing_lambda);
  }
  gate_->Init(gate_config_.get(), train_data_, nullptr, {});
  Log::Debug("MixtureGBDT::Init - gate initialized");

  // Allocate buffers
  size_t nk = static_cast<size_t>(num_data_) * num_experts_;
  responsibilities_.resize(nk);
  expert_pred_.resize(nk);
  gate_proba_.resize(nk);
  yhat_.resize(num_data_);
  gradients_.resize(num_data_);
  hessians_.resize(num_data_);

  // Initialize Markov-specific buffers
  if (use_markov_) {
    prev_gate_proba_.resize(nk);
    const double uniform_prob = 1.0 / num_experts_;
    std::fill(prev_gate_proba_.begin(), prev_gate_proba_.end(), uniform_prob);
  }

  // Initialize Momentum-specific buffers
  if (use_momentum_) {
    prev_responsibilities_.resize(nk);
    momentum_trend_.resize(nk);
    const double uniform_prob = 1.0 / num_experts_;
    std::fill(prev_responsibilities_.begin(), prev_responsibilities_.end(), uniform_prob);
    std::fill(momentum_trend_.begin(), momentum_trend_.end(), 0.0);
  }

  // Initialize responsibilities
  InitResponsibilities();

  Log::Info("MixtureGBDT: Initialization complete (smoothing=%s)",
            config_->mixture_r_smoothing.c_str());
}

void MixtureGBDT::InitResponsibilities() {
  const label_t* labels = train_data_->metadata().label();

  if (config_->mixture_init == "quantile") {
    // Quantile-based initialization: assign samples to experts based on label quantiles
    // This breaks symmetry by giving each expert a different subset of data
    Log::Info("MixtureGBDT: Using quantile-based initialization");

    // Sort indices by label value
    std::vector<data_size_t> sorted_indices(num_data_);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [labels](data_size_t a, data_size_t b) { return labels[a] < labels[b]; });

    // Assign to experts based on quantiles with soft boundaries
    const double base_r = 0.1 / num_experts_;  // Small base probability for all experts
    const double main_r = 1.0 - base_r * num_experts_;  // Main probability for assigned expert

    for (data_size_t rank = 0; rank < num_data_; ++rank) {
      data_size_t i = sorted_indices[rank];
      int assigned_expert = static_cast<int>(rank * num_experts_ / num_data_);
      if (assigned_expert >= num_experts_) assigned_expert = num_experts_ - 1;

      for (int k = 0; k < num_experts_; ++k) {
        if (k == assigned_expert) {
          responsibilities_[i * num_experts_ + k] = main_r + base_r;
        } else {
          responsibilities_[i * num_experts_ + k] = base_r;
        }
      }
    }
  } else if (config_->mixture_init == "random") {
    // Random initialization: randomly assign samples to experts
    Log::Info("MixtureGBDT: Using random initialization");

    std::mt19937 rng(config_->seed);
    std::uniform_int_distribution<int> dist(0, num_experts_ - 1);

    const double base_r = 0.1 / num_experts_;
    const double main_r = 1.0 - base_r * num_experts_;

    for (data_size_t i = 0; i < num_data_; ++i) {
      int assigned_expert = dist(rng);
      for (int k = 0; k < num_experts_; ++k) {
        if (k == assigned_expert) {
          responsibilities_[i * num_experts_ + k] = main_r + base_r;
        } else {
          responsibilities_[i * num_experts_ + k] = base_r;
        }
      }
    }
  } else {
    // Default: quantile-based initialization (better than uniform for MoE)
    // Use quantile as default because it ensures experts specialize on different
    // regions of the target distribution
    Log::Info("MixtureGBDT: Using default quantile-based initialization");

    std::vector<data_size_t> sorted_indices(num_data_);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [labels](data_size_t a, data_size_t b) { return labels[a] < labels[b]; });

    const double base_r = 0.1 / num_experts_;
    const double main_r = 1.0 - base_r * num_experts_;

    for (data_size_t rank = 0; rank < num_data_; ++rank) {
      data_size_t i = sorted_indices[rank];
      int assigned_expert = static_cast<int>(rank * num_experts_ / num_data_);
      if (assigned_expert >= num_experts_) assigned_expert = num_experts_ - 1;

      for (int k = 0; k < num_experts_; ++k) {
        if (k == assigned_expert) {
          responsibilities_[i * num_experts_ + k] = main_r + base_r;
        } else {
          responsibilities_[i * num_experts_ + k] = base_r;
        }
      }
    }
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
  // Note: GetPredictAt returns class-major order (all class 0, then class 1, etc.)
  std::vector<double> gate_raw(static_cast<size_t>(num_data_) * num_experts_);
  int64_t out_len;
  gate_->GetPredictAt(0, gate_raw.data(), &out_len);

  // Apply softmax per sample
  // gate_raw is in class-major order: gate_raw[k * num_data_ + i] = score for sample i, class k
  // gate_proba_ is in sample-major order: gate_proba_[i * num_experts_ + k]
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    // Copy to sample-major order for this sample
    std::vector<double> scores(num_experts_);
    for (int k = 0; k < num_experts_; ++k) {
      scores[k] = gate_raw[k * num_data_ + i];  // class-major indexing
    }
    Softmax(scores.data(), num_experts_,
            gate_proba_.data() + i * num_experts_);
  }

  // Markov mode: blend gate_proba with prev_gate_proba
  // This makes regime transitions smoother and dependent on previous state
  if (use_markov_) {
    const double lambda = config_->mixture_smoothing_lambda;
    if (lambda > 0.0) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        double sum = 0.0;
        for (int k = 0; k < num_experts_; ++k) {
          size_t idx = i * num_experts_ + k;
          // Blend: new_proba = (1-lambda) * current + lambda * prev
          gate_proba_[idx] = (1.0 - lambda) * gate_proba_[idx] +
                             lambda * prev_gate_proba_[idx];
          sum += gate_proba_[idx];
        }
        // Renormalize (should be close to 1 already, but for numerical stability)
        for (int k = 0; k < num_experts_; ++k) {
          gate_proba_[i * num_experts_ + k] /= sum;
        }
      }
    }

    // Update prev_gate_proba with current values (for next iteration)
    // Using row-wise copy: prev[i] = current[i-1] for time series
    // First row keeps its initial/previous value
    for (data_size_t i = num_data_ - 1; i > 0; --i) {
      for (int k = 0; k < num_experts_; ++k) {
        prev_gate_proba_[i * num_experts_ + k] = gate_proba_[(i - 1) * num_experts_ + k];
      }
    }
    // First row: use current gate_proba (no previous available in this batch)
    // This maintains consistency for the first sample
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

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    // Create local scores vector for each thread
    std::vector<double> scores(num_experts_);

    // Compute scores: s_ik = log(gate_proba_ik + eps) - alpha * loss(y_i, expert_pred_ik)
    for (int k = 0; k < num_experts_; ++k) {
      double gate_prob = gate_proba_[i * num_experts_ + k];
      double expert_p = expert_pred_[k * num_data_ + i];
      double loss = ComputePointwiseLoss(labels[i], expert_p);
      scores[k] = std::log(gate_prob + kMixtureEpsilon) - alpha * loss;
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
  const double lambda = config_->mixture_smoothing_lambda;
  if (lambda <= 0.0) {
    return;
  }

  if (config_->mixture_r_smoothing == "ema") {
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
  } else if (config_->mixture_r_smoothing == "momentum") {
    // Momentum smoothing: EMA with trend (direction of change)
    // extrapolated[i] = r[i-1] + lambda * (r[i-1] - r[i-2])
    // r_smooth[i] = (1-lambda)*r[i] + lambda*extrapolated[i]
    // This captures "inertia" - if regime is trending in a direction, continue that trend

    for (data_size_t i = 1; i < num_data_; ++i) {
      double sum = 0.0;
      for (int k = 0; k < num_experts_; ++k) {
        size_t idx = i * num_experts_ + k;
        size_t prev_idx = (i - 1) * num_experts_ + k;

        double extrapolated;
        if (i >= 2) {
          // Use trend from previous samples
          size_t prev2_idx = (i - 2) * num_experts_ + k;
          double trend = responsibilities_[prev_idx] - responsibilities_[prev2_idx];
          extrapolated = responsibilities_[prev_idx] + lambda * trend;
        } else {
          // Not enough history, just use previous value
          extrapolated = responsibilities_[prev_idx];
        }

        // Blend current with extrapolated
        responsibilities_[idx] = (1.0 - lambda) * responsibilities_[idx] +
                                 lambda * extrapolated;
        // Clip to valid range
        if (responsibilities_[idx] < 0.0) responsibilities_[idx] = 0.0;
        if (responsibilities_[idx] > 1.0) responsibilities_[idx] = 1.0;
        sum += responsibilities_[idx];
      }
      // Renormalize
      for (int k = 0; k < num_experts_; ++k) {
        responsibilities_[i * num_experts_ + k] /= (sum + kMixtureEpsilon);
      }
    }
  }
  // Note: Markov mode handles smoothing differently - it uses prev_proba as gate features
  // and updates prev_gate_proba_ after Forward() in TrainOneIter()
}

void MixtureGBDT::CreateGateDataset() {
  // For Markov mode, gate training uses original dataset but
  // gate probabilities are blended with previous probabilities
  // This is a simpler approach that avoids complex dataset reconstruction
  Log::Info("MixtureGBDT: Markov mode - gate will blend current proba with prev_proba");
  // No separate dataset needed - we use the original train_data_ for gate
  // and apply Markov blending in Forward()
}

void MixtureGBDT::UpdateGateDataset() {
  // In simplified Markov mode, we don't rebuild the dataset
  // Instead, we update prev_gate_proba_ after Forward() and blend in Forward()
  // This function is kept for potential future extension with full dataset rebuild
}

void MixtureGBDT::MStepExperts() {
  const label_t* labels = train_data_->metadata().label();

  // Each expert should optimize its OWN prediction toward the label
  // Gradient for expert k: r_ik * d_L(y_i, f_k(x_i)) / d_f_k(x_i)
  // This ensures experts specialize on different parts of the data

  // Train each expert with responsibility-weighted gradients
  // computed from the expert's OWN prediction (not mixture yhat)
  for (int k = 0; k < num_experts_; ++k) {
    std::vector<score_t> grad_k(num_data_);
    std::vector<score_t> hess_k(num_data_);

    // Get expert k's predictions
    const double* expert_k_pred = expert_pred_.data() + k * num_data_;

    if (objective_function_ != nullptr) {
      // Use objective function to compute gradients for expert k's predictions
      std::vector<double> expert_k_pred_vec(expert_k_pred, expert_k_pred + num_data_);
      std::vector<score_t> temp_grad(num_data_);
      std::vector<score_t> temp_hess(num_data_);
      objective_function_->GetGradients(expert_k_pred_vec.data(), temp_grad.data(), temp_hess.data());

      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        double r_ik = responsibilities_[i * num_experts_ + k];
        grad_k[i] = static_cast<score_t>(r_ik * temp_grad[i]);
        hess_k[i] = static_cast<score_t>(r_ik * temp_hess[i]);
      }
    } else {
      // Default to MSE: d/df (y - f)^2 = 2*(f - y)
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        double r_ik = responsibilities_[i * num_experts_ + k];
        double diff = expert_k_pred[i] - labels[i];
        grad_k[i] = static_cast<score_t>(r_ik * 2.0 * diff);
        hess_k[i] = static_cast<score_t>(r_ik * 2.0);
      }
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
      gate_hess[idx] = static_cast<score_t>(std::max(p * (1.0 - p), kMixtureEpsilon));
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

  Log::Debug("MixtureGBDT::TrainOneIter - starting iteration %d", iter_);

  // Forward pass - compute expert predictions and gate probabilities
  Forward();

  // E-step: update responsibilities
  // Skip E-step for first few iterations to allow experts to differentiate
  // with the initial (non-uniform) responsibilities.
  // Without this, all experts start with prediction=0, causing EStep to
  // compute uniform responsibilities and all experts train identically.
  const int warmup_iters = config_->mixture_warmup_iters;
  if (iter_ >= warmup_iters) {
    EStep();

    // Apply time-series smoothing if enabled
    SmoothResponsibilities();
  }

  // M-step: update experts
  MStepExperts();

  // M-step: update gate
  // Gate should always be trained, even during warmup.
  // During warmup, responsibilities are fixed from initialization (quantile-based),
  // so gate learns to predict these fixed responsibilities.
  // This is crucial for hard alpha (1.0) to work properly.
  MStepGate();

  ++iter_;
  Log::Debug("MixtureGBDT::TrainOneIter - completed iteration %d", iter_);

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
  // Create a no-op early stop instance if none provided
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());
  const PredictionEarlyStopInstance* early_stop_ptr = earlyStop ? earlyStop : &no_early_stop;

  // Get expert predictions
  std::vector<double> expert_preds(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->Predict(features, &expert_preds[k], early_stop_ptr);
  }

  // Get gate probabilities
  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), early_stop_ptr);

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
  // Create a no-op early stop instance if none provided
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());
  const PredictionEarlyStopInstance* early_stop_ptr = early_stop ? early_stop : &no_early_stop;

  std::vector<double> expert_preds(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->PredictByMap(features, &expert_preds[k], early_stop_ptr);
  }

  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRawByMap(features, gate_raw.data(), early_stop_ptr);

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
  // Create a no-op early stop instance
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());

  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), &no_early_stop);

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
  // Create a no-op early stop instance
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());

  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), &no_early_stop);
  Softmax(gate_raw.data(), num_experts_, output);
}

void MixtureGBDT::PredictExpertPred(const double* features, double* output) const {
  // Create a no-op early stop instance
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());

  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->Predict(features, &output[k], &no_early_stop);
  }
}

void MixtureGBDT::PredictWithPrevProba(const double* features, const double* prev_proba,
                                        double* output) const {
  // Create a no-op early stop instance
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());
  const PredictionEarlyStopInstance* early_stop_ptr = &no_early_stop;

  // Get expert predictions
  std::vector<double> expert_preds(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->Predict(features, &expert_preds[k], early_stop_ptr);
  }

  // Get current gate probabilities
  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), early_stop_ptr);

  std::vector<double> gate_prob(num_experts_);
  Softmax(gate_raw.data(), num_experts_, gate_prob.data());

  // Blend with prev_proba if provided and in Markov mode
  if (use_markov_ && prev_proba != nullptr) {
    const double lambda = config_->mixture_smoothing_lambda;
    if (lambda > 0.0) {
      double sum = 0.0;
      for (int k = 0; k < num_experts_; ++k) {
        gate_prob[k] = (1.0 - lambda) * gate_prob[k] + lambda * prev_proba[k];
        sum += gate_prob[k];
      }
      // Renormalize
      for (int k = 0; k < num_experts_; ++k) {
        gate_prob[k] /= sum;
      }
    }
  }

  // Compute weighted sum
  double sum = 0.0;
  for (int k = 0; k < num_experts_; ++k) {
    sum += gate_prob[k] * expert_preds[k];
  }
  *output = sum;
}

void MixtureGBDT::PredictRegimeProbaWithPrevProba(const double* features, const double* prev_proba,
                                                   double* output) const {
  // Create a no-op early stop instance
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());

  // Get current gate probabilities
  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), &no_early_stop);
  Softmax(gate_raw.data(), num_experts_, output);

  // Blend with prev_proba if provided and in Markov mode
  if (use_markov_ && prev_proba != nullptr) {
    const double lambda = config_->mixture_smoothing_lambda;
    if (lambda > 0.0) {
      double sum = 0.0;
      for (int k = 0; k < num_experts_; ++k) {
        output[k] = (1.0 - lambda) * output[k] + lambda * prev_proba[k];
        sum += output[k];
      }
      // Renormalize
      for (int k = 0; k < num_experts_; ++k) {
        output[k] /= sum;
      }
    }
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
  ss << "mixture_smoothing_lambda=" << config_->mixture_smoothing_lambda << "\n";
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
  std::string model_str(buffer, len);
  std::istringstream ss(model_str);
  std::string line;

  // Read header
  if (!std::getline(ss, line) || line != "mixture") {
    Log::Fatal("Invalid mixture model format: expected 'mixture' header");
    return false;
  }

  // Parse mixture parameters
  std::unordered_map<std::string, std::string> params;
  while (std::getline(ss, line)) {
    if (line.empty() || line[0] == '[') {
      break;
    }
    size_t eq_pos = line.find('=');
    if (eq_pos != std::string::npos) {
      std::string key = line.substr(0, eq_pos);
      std::string value = line.substr(eq_pos + 1);
      params[key] = value;
    }
  }

  // Extract parameters
  if (params.count("mixture_num_experts")) {
    num_experts_ = std::stoi(params["mixture_num_experts"]);
  }
  if (params.count("mixture_e_step_loss")) {
    e_step_loss_type_ = params["mixture_e_step_loss"];
  }

  // Store loaded parameters for GetLoadedParam (must be valid JSON)
  std::stringstream param_ss;
  param_ss << "{";
  bool first = true;
  for (const auto& kv : params) {
    if (!first) {
      param_ss << ", ";
    }
    first = false;
    param_ss << "\"" << kv.first << "\": ";
    // Try to detect numeric values
    bool is_numeric = !kv.second.empty() && (std::isdigit(kv.second[0]) || kv.second[0] == '-' || kv.second[0] == '.');
    if (is_numeric) {
      param_ss << kv.second;
    } else {
      param_ss << "\"" << kv.second << "\"";
    }
  }
  param_ss << "}";
  loaded_parameter_ = param_ss.str();

  // Find sections
  std::string gate_model_str;
  std::vector<std::string> expert_model_strs(num_experts_);

  // We need to re-parse from the section markers
  std::string remaining = model_str.substr(ss.tellg() > 0 ? static_cast<size_t>(ss.tellg()) - line.size() - 1 : 0);

  // Find [gate_model]
  size_t gate_start = remaining.find("[gate_model]");
  if (gate_start == std::string::npos) {
    Log::Fatal("Invalid mixture model format: [gate_model] section not found");
    return false;
  }
  gate_start += std::string("[gate_model]\n").length();

  // Find first expert
  size_t expert0_start = remaining.find("[expert_model_0]");
  if (expert0_start == std::string::npos) {
    Log::Fatal("Invalid mixture model format: [expert_model_0] section not found");
    return false;
  }

  gate_model_str = remaining.substr(gate_start, expert0_start - gate_start);

  // Parse expert models
  for (int k = 0; k < num_experts_; ++k) {
    std::string section_name = "[expert_model_" + std::to_string(k) + "]";
    size_t section_start = remaining.find(section_name);
    if (section_start == std::string::npos) {
      Log::Fatal("Invalid mixture model format: %s section not found", section_name.c_str());
      return false;
    }
    section_start += section_name.length() + 1;  // +1 for newline

    // Find next section or end
    size_t section_end;
    if (k < num_experts_ - 1) {
      std::string next_section = "[expert_model_" + std::to_string(k + 1) + "]";
      section_end = remaining.find(next_section);
    } else {
      section_end = remaining.length();
    }

    expert_model_strs[k] = remaining.substr(section_start, section_end - section_start);
  }

  // Create config for loading (minimal config)
  config_ = std::unique_ptr<Config>(new Config());
  config_->mixture_num_experts = num_experts_;

  // Initialize experts
  experts_.clear();
  experts_.reserve(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_.emplace_back(new GBDT());
    if (!experts_[k]->LoadModelFromString(expert_model_strs[k].c_str(), expert_model_strs[k].size())) {
      Log::Fatal("Failed to load expert model %d", k);
      return false;
    }
  }

  // Initialize gate
  gate_.reset(new GBDT());
  if (!gate_->LoadModelFromString(gate_model_str.c_str(), gate_model_str.size())) {
    Log::Fatal("Failed to load gate model");
    return false;
  }

  // Set feature info from first expert
  if (!experts_.empty()) {
    max_feature_idx_ = experts_[0]->MaxFeatureIdx();
    feature_names_ = experts_[0]->FeatureNames();
    label_idx_ = experts_[0]->LabelIdx();
  }

  Log::Info("MixtureGBDT: Loaded model with %d experts", num_experts_);
  return true;
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
  Log::Warning("MixtureGBDT::GetLoadedParam called, loaded_parameter_=%s", loaded_parameter_.c_str());
  return loaded_parameter_;
}

std::string MixtureGBDT::ParserConfigStr() const {
  if (!experts_.empty()) {
    return experts_[0]->ParserConfigStr();
  }
  return "";
}

}  // namespace LightGBM
