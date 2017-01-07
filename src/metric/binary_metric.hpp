#ifndef LIGHTGBM_METRIC_BINARY_METRIC_HPP_
#define LIGHTGBM_METRIC_BINARY_METRIC_HPP_

#include <LightGBM/utils/log.h>

#include <LightGBM/metric.h>

#include <algorithm>
#include <vector>
#include <sstream>

namespace LightGBM {

/*!
* \brief Metric for binary classification task.
* Use static class "PointWiseLossCalculator" to calculate loss point-wise
*/
template<typename PointWiseLossCalculator>
class BinaryMetric: public Metric {
public:
  explicit BinaryMetric(const MetricConfig& config) {
    sigmoid_ = static_cast<score_t>(config.sigmoid);
    if (sigmoid_ <= 0.0f) {
      Log::Fatal("Sigmoid parameter %f should greater than zero", sigmoid_);
    }
  }

  virtual ~BinaryMetric() {

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
      for (data_size_t i = 0; i < num_data; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  score_t factor_to_bigger_better() const override {
    return -1.0f;
  }

  std::vector<double> Eval(const score_t* score) const override {
    double sum_loss = 0.0f;
    if (weights_ == nullptr) {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
      for (data_size_t i = 0; i < num_data_; ++i) {
        // sigmoid transform
        score_t prob = 1.0f / (1.0f + std::exp(-2.0f * sigmoid_ * score[i]));
        // add loss
        sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], prob);
      }
    } else {
#pragma omp parallel for schedule(static) reduction(+:sum_loss)
      for (data_size_t i = 0; i < num_data_; ++i) {
        // sigmoid transform
        score_t prob = 1.0f / (1.0f + std::exp(-2.0f * sigmoid_ * score[i]));
        // add loss
        sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], prob) * weights_[i];
      }
    }
    double loss = sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
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
  /*! \brief Name of test set */
  std::vector<std::string> name_;
  /*! \brief Sigmoid parameter */
  score_t sigmoid_;
};

/*!
* \brief Log loss metric for binary classification task.
*/
class BinaryLoglossMetric: public BinaryMetric<BinaryLoglossMetric> {
public:
  explicit BinaryLoglossMetric(const MetricConfig& config) :BinaryMetric<BinaryLoglossMetric>(config) {}

  inline static score_t LossOnPoint(float label, score_t prob) {
    if (label == 0) {
      if (1.0f - prob > kEpsilon) {
        return -std::log(1.0f - prob);
      }
    } else {
      if (prob > kEpsilon) {
        return -std::log(prob);
      }
    }
    return -std::log(kEpsilon);
  }

  inline static const char* Name() {
    return "binary_logloss";
  }
};
/*!
* \brief Error rate metric for binary classification task.
*/
class BinaryErrorMetric: public BinaryMetric<BinaryErrorMetric> {
public:
  explicit BinaryErrorMetric(const MetricConfig& config) :BinaryMetric<BinaryErrorMetric>(config) {}

  inline static score_t LossOnPoint(float label, score_t prob) {
    if (prob <= 0.5f) {
      return label;
    } else {
      return 1.0f - label;
    }
  }

  inline static const char* Name() {
    return "binary_error";
  }
};

/*!
* \brief Auc Metric for binary classification task.
*/
class AUCMetric: public Metric {
public:
  explicit AUCMetric(const MetricConfig&) {

  }

  virtual ~AUCMetric() {
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  score_t factor_to_bigger_better() const override {
    return 1.0f;
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("auc");

    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();

    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }

  std::vector<double> Eval(const score_t* score) const override {
    // get indices sorted by score, descent order
    std::vector<data_size_t> sorted_idx;
    for (data_size_t i = 0; i < num_data_; ++i) {
      sorted_idx.emplace_back(i);
    }
    std::sort(sorted_idx.begin(), sorted_idx.end(), [score](data_size_t a, data_size_t b) {return score[a] > score[b]; });
    // temp sum of postive label
    double cur_pos = 0.0f;
    // total sum of postive label
    double sum_pos = 0.0f;
    // accumlate of auc
    double accum = 0.0f;
    // temp sum of negative label
    double cur_neg = 0.0f;
    score_t threshold = score[sorted_idx[0]];
    if (weights_ == nullptr) {  // no weights
      for (data_size_t i = 0; i < num_data_; ++i) {
        const float cur_label = label_[sorted_idx[i]];
        const score_t cur_score = score[sorted_idx[i]];
        // new threshold
        if (cur_score != threshold) {
          threshold = cur_score;
          // accmulate
          accum += cur_neg*(cur_pos * 0.5f + sum_pos);
          sum_pos += cur_pos;
          // reset
          cur_neg = cur_pos = 0.0f;
        }
        cur_neg += 1.0f - cur_label;
        cur_pos += cur_label;
      }
    } else {  // has weights
      for (data_size_t i = 0; i < num_data_; ++i) {
        const float cur_label = label_[sorted_idx[i]];
        const score_t cur_score = score[sorted_idx[i]];
        const float cur_weight = weights_[sorted_idx[i]];
        // new threshold
        if (cur_score != threshold) {
          threshold = cur_score;
          // accmulate
          accum += cur_neg*(cur_pos * 0.5f + sum_pos);
          sum_pos += cur_pos;
          // reset
          cur_neg = cur_pos = 0.0f;
        }
        cur_neg += (1.0f - cur_label)*cur_weight;
        cur_pos += cur_label*cur_weight;
      }
    }
    accum += cur_neg*(cur_pos * 0.5f + sum_pos);
    sum_pos += cur_pos;
    double auc = 1.0f;
    if (sum_pos > 0.0f && sum_pos != sum_weights_) {
      auc = accum / (sum_pos *(sum_weights_ - sum_pos));
    }
    return std::vector<double>(1, auc);
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
  /*! \brief Name of test set */
  std::vector<std::string> name_;
};

}  // namespace LightGBM
#endif   // LightGBM_METRIC_BINARY_METRIC_HPP_
