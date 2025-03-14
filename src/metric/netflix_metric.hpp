#ifndef LIGHTGBM_METRIC_NETFLIX_METRIC_HPP_
#define LIGHTGBM_METRIC_NETFLIX_METRIC_HPP_

#include <LightGBM/metric.h>

#include <LightGBM/utils/log.h>

#include <cmath>

namespace LightGBM {
/*!
* \brief Shift-Beta Geometric loss function
*/
class sBGMetric: public Metric {
public:
  explicit sBGMetric(const Config& config) :config_(config) {
  }

  virtual ~sBGMetric() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("sbg_loss");
    num_data_ = num_data;
    label_ = metadata.label();
    is_censored_ = metadata.weights();
    weights_ = metadata.weights2();
    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      double sum_weights = 0.0;
      #pragma omp parallel for schedule(runtime) reduction(+:sum_weights)
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights += weights_[i];
      }
      sum_weights_ = sum_weights;
    }
  }

  inline static double LossOnPoint(int horizon, double score1, double score2, int is_censored) {
    double alpha = std::exp(score1);
    double beta  = std::exp(score2);
    double p = alpha / (alpha + beta);
    double s = 1 - p;
    double sum = 0.0;
    for (data_size_t j = 2; j < horizon + 1; ++j) {
        p = p * (beta + j - 2)/(alpha + beta + j - 1 );
        s = s - p;
    }
    if (is_censored == 1) {
        sum = std::log(s);
    } else {
        sum = std::log(p);
    }
    return sum;
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0f;
    #pragma omp parallel for schedule(runtime) reduction(+:sum_loss)
    for (data_size_t i = 0; i < num_data_; ++i) {
        size_t idx1 = i;
        size_t idx2 = static_cast<size_t>(num_data_) + i;
        const double row_weight = weights_ == nullptr ? 1.0 : weights_[i];
        sum_loss += row_weight * LossOnPoint(static_cast<int>(label_[i]), score[idx1], score[idx2], static_cast<int>(is_censored_[i]));
    }
    return std::vector<double>(1, sum_loss / sum_weights_);

  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weighs */
  const label_t* is_censored_;
  /*! \brief Pointer of weighs */
  const label_t* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of this test set */
  Config config_;
  std::vector<std::string> name_;
};

}  // namespace LightGBM
#endif   // LightGBM_METRIC_NETFLIX_METRIC_HPP_