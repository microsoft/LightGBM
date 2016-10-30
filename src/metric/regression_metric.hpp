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

  const char* GetName() const override {
    return name_.c_str();
  }

  bool is_bigger_better() const override {
    return false;
  }

  void Init(const char* test_name, const Metadata& metadata, data_size_t num_data) override {
    std::stringstream str_buf;
    str_buf << test_name << "'s " << PointWiseLossCalculator::Name();
    name_ = str_buf.str();

    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    if (weights_ == nullptr) {
      sum_weights_ = static_cast<float>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  std::vector<float> Eval(const score_t* score) const override {
    score_t sum_loss = 0.0f;
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
    score_t loss = PointWiseLossCalculator::AverageLoss(sum_loss, sum_weights_);
    return std::vector<float>(1, loss);

  }

  inline static score_t AverageLoss(score_t sum_loss, score_t sum_weights) {
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
  float sum_weights_;
  /*! \brief Name of this test set */
  std::string name_;
};

/*! \brief L2 loss for regression task */
class L2Metric: public RegressionMetric<L2Metric> {
public:
  explicit L2Metric(const MetricConfig& config) :RegressionMetric<L2Metric>(config) {}

  inline static score_t LossOnPoint(float label, score_t score) {
    return (score - label)*(score - label);
  }

  inline static score_t AverageLoss(score_t sum_loss, score_t sum_weights) {
    // need sqrt the result for L2 loss
    return std::sqrt(sum_loss / sum_weights);
  }

  inline static const char* Name() {
    return "l2 loss";
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
    return "l1 loss";
  }
};

}  // namespace LightGBM
#endif   // LightGBM_METRIC_REGRESSION_METRIC_HPP_
