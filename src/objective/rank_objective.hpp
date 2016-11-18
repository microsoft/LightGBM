#ifndef LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>

#include <cstdio>
#include <cstring>
#include <cmath>

#include <vector>
#include <algorithm>
#include <limits>

namespace LightGBM {
/*!
* \brief Objective function for Lambdrank with NDCG
*/
class LambdarankNDCG: public ObjectiveFunction {
public:
  explicit LambdarankNDCG(const ObjectiveConfig& config) {
    sigmoid_ = static_cast<score_t>(config.sigmoid);
    // initialize DCG calculator
    DCGCalculator::Init(config.label_gain);
    // copy lable gain to local
    for (auto gain : config.label_gain) {
      label_gain_.push_back(static_cast<score_t>(gain));
    }
    label_gain_.shrink_to_fit();
    // will optimize NDCG@optimize_pos_at_
    optimize_pos_at_ = config.max_position;
    sigmoid_table_.clear();
    inverse_max_dcgs_.clear();
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid param %f should be greater than zero", sigmoid_);
    }
  }
  ~LambdarankNDCG() {

  }
  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    // get boundries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("Lambdarank tasks require query information");
    }
    num_queries_ = metadata.num_queries();
    // cache inverse max DCG, avoid computation many times
    inverse_max_dcgs_.resize(num_queries_);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(optimize_pos_at_,
        label_ + query_boundaries_[i],
        query_boundaries_[i + 1] - query_boundaries_[i]);

      if (inverse_max_dcgs_[i] > 0.0) {
        inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
      }
    }
    // construct sigmoid table to speed up sigmoid transform
    ConstructSigmoidTable();
  }

  void GetGradients(const score_t* score, score_t* gradients,
                    score_t* hessians) const override {
    #pragma omp parallel for schedule(guided)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      GetGradientsForOneQuery(score, gradients, hessians, i);
    }
  }

  inline void GetGradientsForOneQuery(const score_t* score,
              score_t* lambdas, score_t* hessians, data_size_t query_id) const {
    // get doc boundary for current query
    const data_size_t start = query_boundaries_[query_id];
    const data_size_t cnt =
      query_boundaries_[query_id + 1] - query_boundaries_[query_id];
    // get max DCG on current query
    const score_t inverse_max_dcg = inverse_max_dcgs_[query_id];
    // add pointers with offset
    const float* label = label_ + start;
    score += start;
    lambdas += start;
    hessians += start;
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx;
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx.emplace_back(i);
    }
    std::sort(sorted_idx.begin(), sorted_idx.end(),
             [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const score_t best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const score_t wrost_score = score[sorted_idx[worst_idx]];
    // start accmulate lambdas by pairs
    for (data_size_t i = 0; i < cnt; ++i) {
      const data_size_t high = sorted_idx[i];
      const int high_label = static_cast<int>(label[high]);
      const score_t high_score = score[high];
      if (high_score == kMinScore) { continue; }
      const score_t high_label_gain = label_gain_[high_label];
      const score_t high_discount = DCGCalculator::GetDiscount(i);
      score_t high_sum_lambda = 0.0;
      score_t high_sum_hessian = 0.0;
      for (data_size_t j = 0; j < cnt; ++j) {
        // skip same data
        if (i == j) { continue; }

        const data_size_t low = sorted_idx[j];
        const int low_label = static_cast<int>(label[low]);
        const score_t low_score = score[low];
        // only consider pair with different label
        if (high_label <= low_label || low_score == kMinScore) { continue; }

        const score_t delta_score = high_score - low_score;

        const score_t low_label_gain = label_gain_[low_label];
        const score_t low_discount = DCGCalculator::GetDiscount(j);
        // get dcg gap
        const score_t dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const score_t paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        score_t delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (high_label != low_label && best_score != wrost_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }
        // calculate lambda for this pair
        score_t p_lambda = GetSigmoid(delta_score);
        score_t p_hessian = p_lambda * (2.0f - p_lambda);
        // update
        p_lambda *= -delta_pair_NDCG;
        p_hessian *= 2 * delta_pair_NDCG;
        high_sum_lambda += p_lambda;
        high_sum_hessian += p_hessian;
        lambdas[low] -= p_lambda;
        hessians[low] += p_hessian;
      }
      // update
      lambdas[high] += high_sum_lambda;
      hessians[high] += high_sum_hessian;
    }
    // if need weights
    if (weights_ != nullptr) {
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] *= weights_[start + i];
        hessians[i] *= weights_[start + i];
      }
    }
  }


  inline score_t GetSigmoid(score_t score) const {
    if (score <= min_sigmoid_input_) {
      // too small, use lower bound
      return sigmoid_table_[0];
    } else if (score >= max_sigmoid_input_) {
      // too big, use upper bound
      return sigmoid_table_[_sigmoid_bins - 1];
    } else {
      return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) * sigmoid_table_idx_factor_)];
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
      const score_t score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_;
      sigmoid_table_[i] = 2.0f / (1.0f + std::exp(2.0f * score * sigmoid_));
    }
  }

  const char* GetName() const override {
    return "lambdarank";
  }

private:
  /*! \brief Gains for labels */
  std::vector<score_t> label_gain_;
  /*! \brief Cache inverse max DCG, speed up calculation */
  std::vector<score_t> inverse_max_dcgs_;
  /*! \brief Simgoid param */
  score_t sigmoid_;
  /*! \brief Optimized NDCG@ */
  int optimize_pos_at_;
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
  /*! \brief Query boundries */
  const data_size_t* query_boundaries_;
  /*! \brief Cache result for sigmoid transform to speed up */
  std::vector<score_t> sigmoid_table_;
  /*! \brief Number of bins in simoid table */
  size_t _sigmoid_bins = 1024 * 1024;
  /*! \brief Minimal input of sigmoid table */
  score_t min_sigmoid_input_ = -50;
  /*! \brief Maximal input of sigmoid table */
  score_t max_sigmoid_input_ = 50;
  /*! \brief Factor that covert score to bin in sigmoid table */
  score_t sigmoid_table_idx_factor_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
