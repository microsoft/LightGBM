#ifndef LIGHTGBM_METRIC_REGRESSION_METRIC_HPP_
#define LIGHTGBM_METRIC_REGRESSION_METRIC_HPP_

#include <LightGBM/utils/log.h>

#include <LightGBM/metric.h>

#include <cmath>

namespace LightGBM {
/*!
* \brief Metric for regression task.
* Use static class "PointWiseLossCalculator" to calculate loss point-wise
*/
template<typename PointWiseLossCalculator>
class RegressionMetric: public Metric {
public:
  explicit RegressionMetric(const MetricConfig&) {

  }

  virtual ~RegressionMetric() {

  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  score_t factor_to_bigger_better() const override {
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
  }

  std::vector<double> Eval(const score_t* score) const override {
    double sum_loss = 0.0f;
    if (weights_ == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
      for (data_size_t i = 0; i < num_data_; ++i) {
        // add loss
        sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], score[i]);
      }
    } else {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
      for (data_size_t i = 0; i < num_data_; ++i) {
        // add loss
        sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], score[i]) * weights_[i];
      }
    }
    double loss = PointWiseLossCalculator::AverageLoss(sum_loss, sum_weights_);
    return std::vector<double>(1, loss);

  }

  inline static double AverageLoss(double sum_loss, double sum_weights) {
    return sum_loss / sum_weights;
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weighs */
  const float* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of this test set */
  std::vector<std::string> name_;
};

/*! \brief L2 loss for regression task */
class L2Metric: public RegressionMetric<L2Metric> {
public:
  explicit L2Metric(const MetricConfig& config) :RegressionMetric<L2Metric>(config) {}

  inline static score_t LossOnPoint(float label, score_t score) {
    return (score - label)*(score - label);
  }

  inline static double AverageLoss(double sum_loss, double sum_weights) {
    // need sqrt the result for L2 loss
    return std::sqrt(sum_loss / sum_weights);
  }

  inline static const char* Name() {
    return "l2";
  }
};

/*! \brief L1 loss for regression task */
class L1Metric: public RegressionMetric<L1Metric> {
public:
  explicit L1Metric(const MetricConfig& config) :RegressionMetric<L1Metric>(config) {}

  inline static score_t LossOnPoint(float label, score_t score) {
    return std::fabs(score - label);
  }
  inline static const char* Name() {
    return "l1";
  }
};

}  // namespace LightGBM
#endif   // LightGBM_METRIC_REGRESSION_METRIC_HPP_
