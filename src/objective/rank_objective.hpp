/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace LightGBM {

/*!
 * \brief Objective function for Ranking
 */
class RankingObjective : public ObjectiveFunction {
 public:
  explicit RankingObjective(const Config& config)
      : seed_(config.objective_seed) {}

  explicit RankingObjective(const std::vector<std::string>&) : seed_(0) {}

  ~RankingObjective() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    // get boundries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("Ranking tasks require query information");
    }
    num_queries_ = metadata.num_queries();
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
#pragma omp parallel for schedule(guided)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      const data_size_t start = query_boundaries_[i];
      const data_size_t cnt = query_boundaries_[i + 1] - query_boundaries_[i];
      GetGradientsForOneQuery(i, cnt, label_ + start, score + start,
                              gradients + start, hessians + start);
      if (weights_ != nullptr) {
        for (data_size_t j = 0; j < cnt; ++j) {
          gradients[start + j] =
              static_cast<score_t>(gradients[start + j] * weights_[start + j]);
          hessians[start + j] =
              static_cast<score_t>(hessians[start + j] * weights_[start + j]);
        }
      }
    }
  }

  virtual void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                       const label_t* label,
                                       const double* score, score_t* lambdas,
                                       score_t* hessians) const = 0;

  const char* GetName() const override = 0;

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool NeedAccuratePrediction() const override { return false; }

 protected:
  int seed_;
  data_size_t num_queries_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  /*! \brief Query boundaries */
  const data_size_t* query_boundaries_;
};

/*!
 * \brief Objective function for LambdaRank with NDCG
 */
class LambdarankNDCG : public RankingObjective {
 public:
  explicit LambdarankNDCG(const Config& config)
      : RankingObjective(config),
        sigmoid_(config.sigmoid),
        norm_(config.lambdarank_norm),
        truncation_level_(config.lambdarank_truncation_level) {
    label_gain_ = config.label_gain;
    // initialize DCG calculator
    DCGCalculator::DefaultLabelGain(&label_gain_);
    DCGCalculator::Init(label_gain_);
    sigmoid_table_.clear();
    inverse_max_dcgs_.clear();
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid param %f should be greater than zero", sigmoid_);
    }
  }

  explicit LambdarankNDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~LambdarankNDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    DCGCalculator::CheckMetadata(metadata, num_queries_);
    DCGCalculator::CheckLabel(label_, num_data_);
    inverse_max_dcgs_.resize(num_queries_);
#pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(
          truncation_level_, label_ + query_boundaries_[i],
          query_boundaries_[i + 1] - query_boundaries_[i]);

      if (inverse_max_dcgs_[i] > 0.0) {
        inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
      }
    }
    // construct Sigmoid table to speed up Sigmoid transform
    ConstructSigmoidTable();
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // get max DCG on current query
    const double inverse_max_dcg = inverse_max_dcgs_[query_id];
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
        sorted_idx.begin(), sorted_idx.end(),
        [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;
    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        } else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }
        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }
    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }
  }

  inline double GetSigmoid(double score) const {
    if (score <= min_sigmoid_input_) {
      // too small, use lower bound
      return sigmoid_table_[0];
    } else if (score >= max_sigmoid_input_) {
      // too large, use upper bound
      return sigmoid_table_[_sigmoid_bins - 1];
    } else {
      return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) *
                                                sigmoid_table_idx_factor_)];
    }
  }

  void ConstructSigmoidTable() {
    // get boundary
    min_sigmoid_input_ = min_sigmoid_input_ / sigmoid_ / 2;
    max_sigmoid_input_ = -min_sigmoid_input_;
    sigmoid_table_.resize(_sigmoid_bins);
    // get score to bin factor
    sigmoid_table_idx_factor_ =
        _sigmoid_bins / (max_sigmoid_input_ - min_sigmoid_input_);
    // cache
    for (size_t i = 0; i < _sigmoid_bins; ++i) {
      const double score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_;
      sigmoid_table_[i] = 1.0f / (1.0f + std::exp(score * sigmoid_));
    }
  }

  const char* GetName() const override { return "lambdarank"; }

 private:
  /*! \brief Sigmoid param */
  double sigmoid_;
  /*! \brief Normalize the lambdas or not */
  bool norm_;
  /*! \brief Truncation position for max DCG */
  int truncation_level_;
  /*! \brief Cache inverse max DCG, speed up calculation */
  std::vector<double> inverse_max_dcgs_;
  /*! \brief Cache result for sigmoid transform to speed up */
  std::vector<double> sigmoid_table_;
  /*! \brief Gains for labels */
  std::vector<double> label_gain_;
  /*! \brief Number of bins in simoid table */
  size_t _sigmoid_bins = 1024 * 1024;
  /*! \brief Minimal input of sigmoid table */
  double min_sigmoid_input_ = -50;
  /*! \brief Maximal input of Sigmoid table */
  double max_sigmoid_input_ = 50;
  /*! \brief Factor that covert score to bin in Sigmoid table */
  double sigmoid_table_idx_factor_;
};

/*!
 * \brief Implementation of the learning-to-rank objective function, XE_NDCG
 * [arxiv.org/abs/1911.09798].
 */
class RankXENDCG : public RankingObjective {
 public:
  explicit RankXENDCG(const Config& config) : RankingObjective(config) {}

  explicit RankXENDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~RankXENDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      rands_.emplace_back(seed_ + i);
    }
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // Skip groups with too few items.
    if (cnt <= 1) {
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = 0.0f;
        hessians[i] = 0.0f;
      }
      return;
    }

    // Turn scores into a probability distribution using Softmax.
    std::vector<double> rho(cnt, 0.0);
    Common::Softmax(score, rho.data(), cnt);

    // An auxiliary buffer of parameters used to form the ground-truth
    // distribution and compute the loss.
    std::vector<double> params(cnt);

    double inv_denominator = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
      params[i] = Phi(label[i], rands_[query_id].NextFloat());
      inv_denominator += params[i];
    }
    // sum_labels will always be positive number
    inv_denominator = 1. / std::max<double>(kEpsilon, inv_denominator);

    // Approximate gradients and inverse Hessian.
    // First order terms.
    double sum_l1 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = -params[i] * inv_denominator + rho[i];
      lambdas[i] = static_cast<score_t>(term);
      // Params will now store terms needed to compute second-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l1 += params[i];
    }
    // Second order terms.
    double sum_l2 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = rho[i] * (sum_l1 - params[i]);
      lambdas[i] += static_cast<score_t>(term);
      // Params will now store terms needed to compute third-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l2 += params[i];
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] += static_cast<score_t>(rho[i] * (sum_l2 - params[i]));
      hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
    }
  }

  double Phi(const label_t l, double g) const {
    return Common::Pow(2, static_cast<int>(l)) - g;
  }

  const char* GetName() const override { return "rank_xendcg"; }

 private:
  mutable std::vector<Random> rands_;
};

}  // namespace LightGBM
#endif  // LightGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
