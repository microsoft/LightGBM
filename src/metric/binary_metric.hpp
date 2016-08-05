#ifndef LIGHTGBM_METRIC_BINARY_METRIC_HPP_
#define LIGHTGBM_METRIC_BINARY_METRIC_HPP_

#include <LightGBM/utils/log.h>

#include <LightGBM/metric.h>

#include <algorithm>
#include <vector>

namespace LightGBM {

/*!
* \brief Metric for binary classification task.
* Use static class "PointWiseLossCalculator" to calculate loss point-wise
*/
template<typename PointWiseLossCalculator>
class BinaryMetric: public Metric {
public:
  explicit BinaryMetric(const MetricConfig& config) {
    output_freq_ = config.output_freq;
    sigmoid_ = static_cast<score_t>(config.sigmoid);
    if (sigmoid_ <= 0.0f) {
      Log::Stderr("sigmoid param %f should greater than zero", sigmoid_);
    }
  }

  virtual ~BinaryMetric() {

  }

  void Init(const char* test_name, const Metadata& metadata, data_size_t num_data) override {
    name = test_name;
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

  void Print(int iter, const score_t* score) const override {
    score_t sum_loss = 0.0f;
    if (output_freq_ > 0 && iter % output_freq_ == 0) {
      if (weights_ == nullptr) {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // sigmoid transform
          score_t prob = 1.0f / (1.0f + std::exp(-sigmoid_ * score[i]));
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], prob);
        }
      } else {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          // sigmoid transform
          score_t prob = 1.0f / (1.0f + std::exp(-sigmoid_ * score[i]));
          // add loss
          sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], prob) * weights_[i];
        }
      }
      Log::Stdout("Iteration:%d, %s's %s: %f", iter, name, PointWiseLossCalculator::Name(), sum_loss / sum_weights_);
    }
  }

private:
  /*! \brief Output frequently */
  int output_freq_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weighs */
  const float* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of test set */
  const char* name;
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
    return "log loss";
  }
};
/*!
* \brief Error rate metric for binary classification task.
*/
class BinaryErrorMetric: public BinaryMetric<BinaryErrorMetric> {
public:
  explicit BinaryErrorMetric(const MetricConfig& config) :BinaryMetric<BinaryErrorMetric>(config) {}

  inline static score_t LossOnPoint(float label, score_t prob) {
    if (prob < 0.5f) {
      return label;
    } else {
      return 1.0f - label;
    }
  }

  inline static const char* Name() {
    return "error rate";
  }
};

/*!
* \brief Auc Metric for binary classification task.
*/
class AUCMetric: public Metric {
public:
  explicit AUCMetric(const MetricConfig& config) {
    output_freq_ = config.output_freq;
  }

  virtual ~AUCMetric() {
  }

  void Init(const char* test_name, const Metadata& metadata, data_size_t num_data) override {
    name = test_name;
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

  void Print(int iter, const score_t* score) const override {
    if (output_freq_ > 0 && iter % output_freq_ == 0) {
      // get indices sorted by score, descent order
      std::vector<data_size_t> sorted_idx;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sorted_idx.emplace_back(i);
      }
      std::sort(sorted_idx.begin(), sorted_idx.end(), [score](data_size_t a, data_size_t b) {return score[a] > score[b]; });
      // temp sum of postive label
      double cur_pos = 0.0;
      // total sum of postive label
      double sum_pos = 0.0;
      // accumlate of auc
      double accum = 0.0;
      // temp sum of negative label
      double cur_neg = 0.0;
      score_t threshold = score[sorted_idx[0]];
      if (weights_ == nullptr) {  // not weights
        for (data_size_t i = 0; i < num_data_; ++i) {
          const float cur_label = label_[sorted_idx[i]];
          const score_t cur_score = score[sorted_idx[i]];
          // new threshold
          if (cur_score != threshold) {
            threshold = cur_score;
            // accmulate
            accum += cur_neg*(cur_pos * 0.5 + sum_pos);
            sum_pos += cur_pos;
            // reset
            cur_neg = cur_pos = 0.0;
          }
          cur_neg += 1.0 - cur_label;
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
            accum += cur_neg*(cur_pos * 0.5 + sum_pos);
            sum_pos += cur_pos;
            // reset
            cur_neg = cur_pos = 0.0;
          }
          cur_neg += (1.0 - cur_label)*cur_weight;
          cur_pos += cur_label*cur_weight;
        }
      }
      accum += cur_neg*(cur_pos * 0.5 + sum_pos);
      sum_pos += cur_pos;
      double auc = 1.0;
      if (sum_pos > 0.0f && sum_pos != sum_weights_) {
        auc = accum / (sum_pos *(sum_weights_ - sum_pos));
      }
      Log::Stdout("iteration:%d, %s's %s: %f", iter, name, "auc", auc);
    }
  }

private:
  /*! \brief Output frequently */
  int output_freq_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weighs */
  const float* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of test set */
  const char* name;
};

}  // namespace LightGBM
#endif  #endif  // LightGBM_METRIC_BINARY_METRIC_HPP_
