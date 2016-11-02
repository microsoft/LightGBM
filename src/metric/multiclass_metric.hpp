#ifndef LIGHTGBM_METRIC_MULTICLASS_METRIC_HPP_
#define LIGHTGBM_METRIC_MULTICLASS_METRIC_HPP_

#include <LightGBM/utils/log.h>

#include <LightGBM/metric.h>

#include <cmath>

namespace LightGBM {
/*!
* \brief Metric for multiclass task.
* Use static class "PointWiseLossCalculator" to calculate loss point-wise
*/
template<typename PointWiseLossCalculator>
class MulticlassMetric: public Metric {
public:
  explicit MulticlassMetric(const MetricConfig& config) {
      num_class_ = config.num_class;
  }

  virtual ~MulticlassMetric() {

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
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights_ += weights_[i];
      }
    }
  }
  
  const char* GetName() const override {
    return name_.c_str();
  }

  bool is_bigger_better() const override {
    return false;
  }
  
  std::vector<double> Eval(const score_t* score) const override {
    double sum_loss = 0.0;
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static) reduction(+:sum_loss)
      for (data_size_t i = 0; i < num_data_; ++i) {
        std::vector<double> rec(num_class_);
        for (int k = 0; k < num_class_; ++k) {
          rec[k] = static_cast<double>(score[k * num_data_ + i]);
        }
        // add loss
        sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], rec);
      }
    } else {
      #pragma omp parallel for schedule(static) reduction(+:sum_loss)
      for (data_size_t i = 0; i < num_data_; ++i) {
        std::vector<double> rec(num_class_);
        for (int k = 0; k < num_class_; ++k) {
          rec[k] = static_cast<double>(score[k * num_data_ + i]);
        }
        // add loss
        sum_loss += PointWiseLossCalculator::LossOnPoint(label_[i], rec) * weights_[i];
      }
    }
    double loss = sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
  }

private:
  /*! \brief Output frequency */
  int output_freq_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Number of classes */
  int num_class_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weighs */
  const float* weights_;
  /*! \brief Sum weights */
  double sum_weights_;
  /*! \brief Name of this test set */
  std::string name_;
};

/*! \brief L2 loss for multiclass task */
class MultiErrorMetric: public MulticlassMetric<MultiErrorMetric> {
public:
  explicit MultiErrorMetric(const MetricConfig& config) :MulticlassMetric<MultiErrorMetric>(config) {}

  inline static score_t LossOnPoint(float label, std::vector<double> score) {
    size_t k = static_cast<size_t>(label);
    for (size_t i = 0; i < score.size(); ++i){
        if (i != k && score[i] > score[k]) {
            return 0.0f;
        }
    }
    return 1.0f;
  }

  inline static const char* Name() {
    return "multi error";
  }
};

/*! \brief Logloss for multiclass task */
class MultiLoglossMetric: public MulticlassMetric<MultiLoglossMetric> {
public:
  explicit MultiLoglossMetric(const MetricConfig& config) :MulticlassMetric<MultiLoglossMetric>(config) {}

  inline static score_t LossOnPoint(float label, std::vector<double> score) {
    size_t k = static_cast<size_t>(label);
    Common::Softmax(&score);
    if (score[k] > kEpsilon) {
      return static_cast<score_t>(-std::log(score[k]));
    } else {
      return -std::log(kEpsilon);
    }
  }
  
  inline static const char* Name() {
    return "multi logloss";
  }
};

}  // namespace LightGBM
#endif   // LightGBM_METRIC_MULTICLASS_METRIC_HPP_
