#ifndef LIGHTGBM_OBJECTIVE_NETFLIX_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_NETFLIX_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>

#include <cstring>
#include <cmath>
#include <vector>

#include "binary_objective.hpp"

namespace LightGBM {
/*!
* \brief Shifted Beta-Geometric objective function
*/
class sBGObjective: public ObjectiveFunction {
public:

  explicit sBGObjective(const Config& config) : config_(config) {
  }

  explicit sBGObjective(const std::vector<std::string>&) {
  }

  ~sBGObjective() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    is_censored_ = metadata.weights();
    weight_ = metadata.weights2();
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    #pragma omp parallel for schedule(runtime)
    for(data_size_t i = 0; i < num_data_; ++i){
        size_t idx1 = i;
        size_t idx2 = static_cast<size_t>(num_data_) + i;

        const int is_censored = static_cast<int>(is_censored_[i]);
        const double row_weight = weight_ == nullptr ? 1.0 : weight_[i];
        double alpha = std::exp(score[idx1]);
        double beta = std::exp(score[idx2]);
        double ab_inv = 1.0 / (alpha + beta);
        auto horizon = static_cast<int>(label_[i]);
        double ga; // gradient of pointiwise loss with respect to alpha
        double gb; // gradient of pointiwise loss with respect to beta
        double ha; // diagonal hessian of pointiwise loss with respect to alpha
        double hb; // diagonal hessian of pointiwise loss with respect to beta

        ha = - alpha * beta * ab_inv * ab_inv;
        hb = ha;
        if(is_censored == 0){
        ga = beta * ab_inv;
        gb = -ga;
        for(int j = 2; j < horizon + 1; ++j){
            double den = 1.0 / (alpha + beta + j - 1);
            double den_b = 1.0 / (beta + j - 2);
            double den2 = den * den;
            ga += - alpha * den;
            gb += beta * (den_b - den);
            ha += - alpha * (beta + j - 1) * den2;
            hb += beta * ((j -2) * den_b * den_b - (alpha + j -1) * den2);
        }
        } else {
        ga = -alpha * ab_inv;
        gb = -ga;
        for(int j = 2; j < horizon + 1; ++j){
            double den = 1.0 / (alpha + beta + j - 1);
            double den_b = 1.0 / (beta + j - 1);
            double den2 = den * den;
            ga += - alpha * den;
            gb += beta * (den_b - den);
            ha += - alpha * (beta + j -1) * den2;
            hb += beta * ((j - 1) * den_b * den_b - (alpha + j -1) * den2);
        }       
        }
        gradients[idx1] = -ga * row_weight;
        gradients[idx2] = -gb * row_weight;
        hessians[idx1] = -ha * row_weight;
        hessians[idx2] = -hb * row_weight;
    }
  }

  double BoostFromScore(int) const override {
    return 0.0;
  }

  void ConvertOutput(const double* input, double* output) const override {
      output[0] = exp(input[0]);
      output[1] = exp(input[1]);
  }

  const char* GetName() const override {
    return "sbg";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool SkipEmptyClass() const override { return false; }

  int NumModelPerIteration() const override { return 2; }

  int NumPredictOneRow() const override { return 2; }

  bool NeedAccuratePrediction() const override { return true; }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Is Censored for data */
  const label_t* is_censored_;
  /*! \brief per-row weight */
  const label_t* weight_;

  Config config_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_NETFLIX_OBJECTIVE_HPP_