/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_
#define LIGHTGBM_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_

#include "cuda_metric.hpp"
#include "../regression_metric.hpp"

#define EVAL_BLOCK_SIZE_REGRESSION_METRIC (1024)

namespace LightGBM {

// TODO(shiyu1994): merge CUDARegressionMetric and CUDABinaryLossMetric into CUDAPointWiseMetric
template <typename CUDAPointWiseLossCalculator>
class CUDARegressionMetric : public CUDAMetricInterface, public RegressionMetric<CUDAPointWiseLossCalculator> {
 public:
  explicit CUDARegressionMetric(const Config& config);

  ~CUDARegressionMetric();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;

  inline static double AverageLoss(double sum_loss, double sum_weights) {
    // need sqrt the result for RMSE loss
    return (sum_loss / sum_weights);
  }

  inline static double LossOnPoint(label_t /*label*/, double /*score*/, const Config& /*config*/) {
    Log::Fatal("Calling host LossOnPoint for a CUDA metric.");
    return 0.0f;
  }

 protected:
  void LaunchEvalKernel(const double* score) const;

  void LaunchEvalKernelInner(const double* score) const;

  __device__ inline static double SafeLogCUDA(const double x) {
    if (x > 0) {
      return log(x);
    } else {
      return -INFINITY;
    }
  }

  const label_t* cuda_label_;
  const label_t* cuda_weights_;
  double* cuda_score_convert_buffer_;
  double* cuda_sum_loss_buffer_;
  double* cuda_sum_loss_;
};

class CUDARMSEMetric : public CUDARegressionMetric<CUDARMSEMetric> {
 public:
  explicit CUDARMSEMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double /*alpha*/, const double /*fair_c*/, const double /*tweedie_variance_power*/) {
    return (score - label) * (score - label);
  }

  inline static double AverageLoss(double sum_loss, double sum_weights) {
    // need sqrt the result for RMSE loss
    return std::sqrt(sum_loss / sum_weights);
  }

  inline static const char* Name() {
    return "rmse";
  }
};

class CUDAL2Metric : public CUDARegressionMetric<CUDAL2Metric> {
 public:
  explicit CUDAL2Metric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double /*alpha*/, const double /*fair_c*/, const double /*tweedie_variance_power*/) {
    return (score - label)*(score - label);
  }

  inline static const char* Name() {
    return "l2";
  }
};

class CUDAQuantileMetric : public CUDARegressionMetric<CUDAQuantileMetric> {
 public:
  explicit CUDAQuantileMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double alpha, const double /*fair_c*/, const double /*tweedie_variance_power*/) {
    double delta = label - score;
    if (delta < 0) {
      return (alpha - 1.0f) * delta;
    } else {
      return alpha * delta;
    }
  }

  inline static const char* Name() {
    return "quantile";
  }
};

class CUDAL1Metric : public CUDARegressionMetric<CUDAL1Metric> {
 public:
  explicit CUDAL1Metric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double /*alpha*/, const double /*fair_c*/, const double /*tweedie_variance_power*/) {
    return fabs(score - label);
  }

  inline static const char* Name() {
    return "l1";
  }
};

class CUDAHuberLossMetric : public CUDARegressionMetric<CUDAHuberLossMetric> {
 public:
  explicit CUDAHuberLossMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double alpha, const double /*fair_c*/, const double /*tweedie_variance_power*/) {
    const double diff = score - label;
    if (fabs(diff) <= alpha) {
      return 0.5f * diff * diff;
    } else {
      return alpha * (fabs(diff) - 0.5f * alpha);
    }
  }

  inline static const char* Name() {
    return "huber";
  }
};

class CUDAFairLossMetric: public CUDARegressionMetric<CUDAFairLossMetric> {
 public:
  explicit CUDAFairLossMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double /*alpha*/, const double fair_c, const double /*tweedie_variance_power*/) {
    const double x = fabs(score - label);
    const double c = fair_c;
    return c * x - c * c * log(1.0f + x / c);
  }

  inline static const char* Name() {
    return "fair";
  }
};

class CUDAPoissonMetric: public CUDARegressionMetric<CUDAPoissonMetric> {
 public:
  explicit CUDAPoissonMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double /*alpha*/, const double /*fair_c*/, const double /*tweedie_variance_power*/) {
    const double eps = 1e-10f;
    if (score < eps) {
      score = eps;
    }
    return score - label * log(score);
  }

  inline static const char* Name() {
    return "poisson";
  }
};

class CUDAMAPEMetric : public CUDARegressionMetric<CUDAMAPEMetric> {
 public:
  explicit CUDAMAPEMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double /*alpha*/, const double /*fair_c*/, const double /*tweedie_variance_power*/) {
    return fabs((label - score)) / fmax(1.0f, fabs(label));
  }
  inline static const char* Name() {
    return "mape";
  }
};

class CUDAGammaMetric : public CUDARegressionMetric<CUDAGammaMetric> {
 public:
  explicit CUDAGammaMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double /*alpha*/, const double /*fair_c*/, const double /*tweedie_variance_power*/) {
    const double psi = 1.0;
    const double theta = -1.0 / score;
    const double a = psi;
    const double b = -SafeLogCUDA(-theta);
    const double c = 1. / psi * SafeLogCUDA(label / psi) - SafeLogCUDA(label) - 0;  // 0 = std::lgamma(1.0 / psi) = std::lgamma(1.0);
    return -((label * theta - b) / a + c);
  }
  inline static const char* Name() {
    return "gamma";
  }

  inline static void CheckLabel(label_t label) {
    CHECK_GT(label, 0);
  }
};

class CUDAGammaDevianceMetric : public CUDARegressionMetric<CUDAGammaDevianceMetric> {
 public:
  explicit CUDAGammaDevianceMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double /*alpha*/, const double /*fair_c*/, const double /*tweedie_variance_power*/) {
    const double epsilon = 1.0e-9;
    const double tmp = label / (score + epsilon);
    return tmp - SafeLogCUDA(tmp) - 1;
  }

  inline static const char* Name() {
    return "gamma_deviance";
  }

  inline static double AverageLoss(double sum_loss, double) {
    return sum_loss * 2;
  }

  inline static void CheckLabel(label_t label) {
    CHECK_GT(label, 0);
  }
};

class CUDATweedieMetric : public CUDARegressionMetric<CUDATweedieMetric> {
 public:
  explicit CUDATweedieMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, double score,
      const double /*alpha*/, const double /*fair_c*/, const double tweedie_variance_power) {
    const double rho = tweedie_variance_power;
    const double eps = 1e-10f;
    if (score < eps) {
      score = eps;
    }
    const double a = label * exp((1 - rho) * log(score)) / (1 - rho);
    const double b = exp((2 - rho) * log(score)) / (2 - rho);
    return -a + b;
  }

  inline static const char* Name() {
    return "tweedie";
  }
};

}  // namespace LightGBM

#endif  // LIGHTGBM_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_
