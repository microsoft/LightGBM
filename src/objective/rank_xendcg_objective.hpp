#ifndef LIGHTGBM_OBJECTIVE_RANK_XENDCG_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_XENDCG_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>

#include <limits>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <math.h>
#include <random>
#include <vector>
#include <random>

namespace LightGBM {
/*!
* \brief Implementation of the learning-to-rank objective function, XE_NDCG [arxiv.org/abs/1911.09798].
*/
class RankXENDCG: public ObjectiveFunction {
 public:
  explicit RankXENDCG(const Config& config) :
      generator_((std::random_device())()), distribution_(0.0, 1.0) {
    if (config.seed != 0) {
      generator_.seed(config.seed);
    }
  }

  explicit RankXENDCG(const std::vector<std::string>&) :
      generator_((std::random_device())()), distribution_(0.0, 1.0) {}

  ~RankXENDCG() {
  }
  void Init(const Metadata& metadata, data_size_t) override {
    // get label
    label_ = metadata.label();
    // get boundries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("RankXENDCG tasks require query information");
    }
    num_queries_ = metadata.num_queries();
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    #pragma omp parallel for schedule(guided)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      GetGradientsForOneQuery(score, gradients, hessians, i);
    }
  }

  inline void GetGradientsForOneQuery(
      const double* score,
      score_t* lambdas, score_t* hessians, data_size_t query_id) const {
    // get doc boundary for current query
    const data_size_t start = query_boundaries_[query_id];
    const data_size_t cnt =
      query_boundaries_[query_id + 1] - query_boundaries_[query_id];
    // add pointers with offset
    const label_t* label = label_ + start;
    score += start;
    lambdas += start;
    hessians += start;

    // Turn scores into a probability distribution using Softmax.
    std::vector<double> rho;
    for (data_size_t i = 0; i < cnt; ++i) {
      rho.emplace_back(score[i]);
    }

    double max_score = *std::max_element(
        std::begin(rho), std::end(rho));
    double sum_exp = 0.0f;
    for (data_size_t i = 0; i < cnt; ++i) {
      sum_exp += std::exp(rho[i] - max_score);
    }
    double eps = std::exp(-max_score) * 1e-20;
    double log_sum_exp = max_score + log(sum_exp + eps);
    for (data_size_t i = 0; i < cnt; ++i) {
      rho[i] = std::exp(rho[i]- log_sum_exp);
    }

    // Prepare a vector of gammas, a parameter of the loss.
    std::vector<double> gammas(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      gammas[i] = distribution_(generator_);
    }

    // Skip query if sum of labels is 0.
    float sum_labels = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
      sum_labels += phi(label[i], gammas[i]);
    }
    if (sum_labels == 0) {
      return;
    }

    // Approximate gradients and inverse Hessian.
    // First order terms.
    std::vector<double> L1s(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      L1s[i] = -phi(label[i], gammas[i])/sum_labels + rho[i];
    }
    // Second-order terms.
    std::vector<double> L2s(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      for (data_size_t j = 0; j < cnt; ++j) {
        if (i == j) continue;
        L2s[i] += L1s[j] / (1 - rho[j]);
      }
    }
    // Third-order terms.
    std::vector<double> L3s(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      for (data_size_t j = 0; j < cnt; ++j) {
        if (i == j) continue;
        L3s[i] += rho[j] * L2s[j] / (1 - rho[j]);
      }
    }

    // Finally, prepare lambdas and hessians.
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = static_cast<score_t>(
          L1s[i] + rho[i]*L2s[i] + rho[i]*L3s[i]);
      hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
    }
  }

  double phi(const label_t l, double g) const {
    return std::pow(2, l) - g;
  }

  const char* GetName() const override {
    return "rank_xendcg";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool NeedAccuratePrediction() const override { return false; }

 private:
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Query boundries */
  const data_size_t* query_boundaries_;

  // A pseudo-random number generator.
  mutable std::mt19937 generator_;
  // The Uniform distribution to sample from.
  mutable std::uniform_real_distribution<double> distribution_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_RANK_XENDCG_OBJECTIVE_HPP_
