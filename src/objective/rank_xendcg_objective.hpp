/*!
 * Copyright (c) 2019 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_RANK_XENDCG_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_XENDCG_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/random.h>

#include <string>
#include <vector>

namespace LightGBM {
/*!
* \brief Implementation of the learning-to-rank objective function, XE_NDCG [arxiv.org/abs/1911.09798].
*/
class RankXENDCG: public ObjectiveFunction {
 public:
  explicit RankXENDCG(const Config& config) {
    rand_ = new Random(config.objective_seed);
  }

  explicit RankXENDCG(const std::vector<std::string>&) {
    rand_ = new Random();
  }

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
    std::vector<double> rho(cnt);
    Common::Softmax(score, &rho[0], cnt);

    // Prepare a vector of gammas, a parameter of the loss.
    std::vector<double> gammas(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      gammas[i] = rand_->NextFloat();
    }

    // Skip query if sum of labels is 0.
    float sum_labels = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
      sum_labels += static_cast<float>(phi(label[i], gammas[i]));
    }
    if (std::fabs(sum_labels) < kEpsilon) {
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
    return Common::Pow(2, static_cast<int>(l)) - g;
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
  /*! \brief Pseudo-random number generator */
  Random* rand_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_RANK_XENDCG_OBJECTIVE_HPP_
