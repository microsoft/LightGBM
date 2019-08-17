/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_METRIC_REGRESSION_METRIC_HPP_
#define LIGHTGBM_METRIC_REGRESSION_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <algorithm>
#include <cmath>
#include <vector>

namespace LightGBM {
/*!
* \brief Metric for regression task.
* Use static class "PointWiseLossCalculator" to calculate loss point-wise
*/
template<typename PointWiseLossCalculator>
class RegressionMetric: public Metric {
 public:
  explicit RegressionMetric(const Config& config) :config_(config) {
  }

  virtual ~RegressionMetric() {
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back(PointWiseLossCalculator::Name());
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
      }
    }
    for (data_size_t i = 0; i < num_data_; ++i) {
      PointWiseLossCalculator::CheckLabel(label_[i]);
    }
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0f;
    if (objective == nullptr) {
      if (weights_ == nullptr) {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], score[i], config_);
        }
      } else {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], score[i], config_) * weights_[i];
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          double t = 0;
          objective->ConvertOutput(&score[i], &t);
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], t, config_);
        }
      } else {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // add loss
          double t = 0;
          objective->ConvertOutput(&score[i], &t);
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], t, config_) * weights_[i];
        }
      }
    }
    double loss = PointWiseLossCalculator::AverageLoss(sum_loss, sum_weights_);
    return std::vector<double>(1, loss);
  }

  inline static double AverageLoss(double sum_loss, double sum_weights) {
    return sum_loss / sum_weights;
  }

  inline static void CheckLabel(label_t) {
  }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weighs */
  const label_t* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of this test set */
  Config config_;
  std::vector<std::string> name_;
};

/*! \brief RMSE loss for regression task */
class RMSEMetric: public RegressionMetric<RMSEMetric> {
 public:
  explicit RMSEMetric(const Config& config) :RegressionMetric<RMSEMetric>(config) {}

  inline static double LossOnPoint(label_t label, double score, const Config&) {
    return (score - label)*(score - label);
  }

  inline static double AverageLoss(double sum_loss, double sum_weights) {
    // need sqrt the result for RMSE loss
    return std::sqrt(sum_loss / sum_weights);
  }

  inline static const char* Name() {
    return "rmse";
  }
};

/*! \brief L2 loss for regression task */
class L2Metric: public RegressionMetric<L2Metric> {
 public:
  explicit L2Metric(const Config& config) :RegressionMetric<L2Metric>(config) {}

  inline static double LossOnPoint(label_t label, double score, const Config&) {
    return (score - label)*(score - label);
  }

  inline static const char* Name() {
    return "l2";
  }
};

/*! \brief Quantile loss for regression task */
class QuantileMetric : public RegressionMetric<QuantileMetric> {
 public:
  explicit QuantileMetric(const Config& config) :RegressionMetric<QuantileMetric>(config) {
  }

  inline static double LossOnPoint(label_t label, double score, const Config& config) {
    double delta = label - score;
    if (delta < 0) {
      return (config.alpha - 1.0f) * delta;
    } else {
      return config.alpha * delta;
    }
  }

  inline static const char* Name() {
    return "quantile";
  }
};


/*! \brief L1 loss for regression task */
class L1Metric: public RegressionMetric<L1Metric> {
 public:
  explicit L1Metric(const Config& config) :RegressionMetric<L1Metric>(config) {}

  inline static double LossOnPoint(label_t label, double score, const Config&) {
    return std::fabs(score - label);
  }
  inline static const char* Name() {
    return "l1";
  }
};

/*! \brief Huber loss for regression task */
class HuberLossMetric: public RegressionMetric<HuberLossMetric> {
 public:
  explicit HuberLossMetric(const Config& config) :RegressionMetric<HuberLossMetric>(config) {
  }

  inline static double LossOnPoint(label_t label, double score, const Config& config) {
    const double diff = score - label;
    if (std::abs(diff) <= config.alpha) {
      return 0.5f * diff * diff;
    } else {
      return config.alpha * (std::abs(diff) - 0.5f * config.alpha);
    }
  }

  inline static const char* Name() {
    return "huber";
  }
};

/*! \brief Fair loss for regression task */
// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
class FairLossMetric: public RegressionMetric<FairLossMetric> {
 public:
  explicit FairLossMetric(const Config& config) :RegressionMetric<FairLossMetric>(config) {
  }

  inline static double LossOnPoint(label_t label, double score, const Config& config) {
    const double x = std::fabs(score - label);
    const double c = config.fair_c;
    return c * x - c * c * std::log(1.0f + x / c);
  }

  inline static const char* Name() {
    return "fair";
  }
};

/*! \brief Poisson regression loss for regression task */
class PoissonMetric: public RegressionMetric<PoissonMetric> {
 public:
  explicit PoissonMetric(const Config& config) :RegressionMetric<PoissonMetric>(config) {
  }

  inline static double LossOnPoint(label_t label, double score, const Config&) {
    const double eps = 1e-10f;
    if (score < eps) {
      score = eps;
    }
    return score - label * std::log(score);
  }
  inline static const char* Name() {
    return "poisson";
  }
};


/*! \brief Mape regression loss for regression task */
class MAPEMetric : public RegressionMetric<MAPEMetric> {
 public:
  explicit MAPEMetric(const Config& config) :RegressionMetric<MAPEMetric>(config) {
  }

  inline static double LossOnPoint(label_t label, double score, const Config&) {
    return std::fabs((label - score)) / std::max(1.0f, std::fabs(label));
  }
  inline static const char* Name() {
    return "mape";
  }
};

class GammaMetric : public RegressionMetric<GammaMetric> {
 public:
  explicit GammaMetric(const Config& config) :RegressionMetric<GammaMetric>(config) {
  }

  inline static double LossOnPoint(label_t label, double score, const Config&) {
    const double psi = 1.0;
    const double theta = -1.0 / score;
    const double a = psi;
    const double b = -Common::SafeLog(-theta);
    const double c = 1. / psi * Common::SafeLog(label / psi) - Common::SafeLog(label) - 0;  // 0 = std::lgamma(1.0 / psi) = std::lgamma(1.0);
    return -((label * theta - b) / a + c);
  }
  inline static const char* Name() {
    return "gamma";
  }

  inline static void CheckLabel(label_t label) {
    CHECK(label > 0);
  }
};


class GammaDevianceMetric : public RegressionMetric<GammaDevianceMetric> {
 public:
  explicit GammaDevianceMetric(const Config& config) :RegressionMetric<GammaDevianceMetric>(config) {
  }

  inline static double LossOnPoint(label_t label, double score, const Config&) {
    const double epsilon = 1.0e-9;
    const double tmp = label / (score + epsilon);
    return tmp - Common::SafeLog(tmp) - 1;
  }
  inline static const char* Name() {
    return "gamma_deviance";
  }
  inline static double AverageLoss(double sum_loss, double) {
    return sum_loss * 2;
  }
  inline static void CheckLabel(label_t label) {
    CHECK(label > 0);
  }
};

class TweedieMetric : public RegressionMetric<TweedieMetric> {
 public:
  explicit TweedieMetric(const Config& config) :RegressionMetric<TweedieMetric>(config) {
  }

  inline static double LossOnPoint(label_t label, double score, const Config& config) {
    const double rho = config.tweedie_variance_power;
    const double eps = 1e-10f;
    if (score < eps) {
      score = eps;
    }
    const double a = label * std::exp((1 - rho) * std::log(score)) / (1 - rho);
    const double b = std::exp((2 - rho) * std::log(score)) / (2 - rho);
    return -a + b;
  }
  inline static const char* Name() {
    return "tweedie";
  }
};


}  // namespace LightGBM
#endif   // LightGBM_METRIC_REGRESSION_METRIC_HPP_
