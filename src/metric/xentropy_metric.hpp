#ifndef LIGHTGBM_METRIC_XENTROPY_METRIC_HPP_
#define LIGHTGBM_METRIC_XENTROPY_METRIC_HPP_

#include <LightGBM/utils/log.h>
#include <LightGBM/utils/common.h>

#include <LightGBM/metric.h>

#include <algorithm>
#include <vector>
#include <sstream>

namespace LightGBM {

  // label should be in interval [0, 1];
  // prob should be in interval (0, 1); prob is clipped if needed
  inline static double XentLoss(float label, double prob) {
    const double log_arg_epsilon = 1.0e-12;
    double a = label;
    if (prob > log_arg_epsilon) {
      a *= std::log(prob);
    } else {
      a *= std::log(log_arg_epsilon);
    }
    double b = 1.0f - label;
    if (1.0f - prob > log_arg_epsilon) {
      b *= std::log(1.0f - prob);
    } else {
      b *= std::log(log_arg_epsilon);
    }
    return - (a + b);
  }

  // hhat >(=) 0 assumed; and weight > 0 required; but not checked here
  inline static double XentLoss1(float label, float weight, double hhat) {
    return XentLoss(label, 1.0f - std::exp(-weight * hhat));
  }

//
// CrossEntropyMetric : "xentropy" : (optional) weights are used linearly
//
class CrossEntropyMetric : public Metric {
public:
	explicit CrossEntropyMetric(const MetricConfig&) {}
  virtual ~CrossEntropyMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("xentropy");
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    // ensure that labels are in interval [0, 1], interval ends included
    if (label_ == nullptr) {
       Log::Fatal("[%s]: label nullptr", __func__);
    }
    for (data_size_t i = 0; i < num_data; ++i) {
    	if (label_[i] < 0.0f || label_[i] > 1.0f) {
    		Log::Fatal("[%s]: metric does not tolerate label [#%i] outside [0, 1]", __func__, i);
    	}
    }
    Log::Info("[%s:%s]: labels passed interval [0, 1] check (metric)",  GetName()[0].c_str(), __func__);

    if (weights_ == nullptr) {
      sum_weights_ = static_cast<double>(num_data_);
    } else {
      sum_weights_ = 0.0f;
      for (data_size_t i = 0; i < num_data; ++i) {
        sum_weights_ += weights_[i];
      }
    }
    // check weight sum
    if (sum_weights_ <= 0.0f) {
      Log::Fatal("[%s]: sum-of-weights = %f is non-positive", __func__, sum_weights_);
    } else {
      Log::Info("[%s:%s]: sum-of-weights = %f",  GetName()[0].c_str(), __func__, sum_weights_);
    }
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0f;
    if (objective == nullptr) {
      Log::Warning("[%s]: objective nullptr", __func__);
      if (weights_ == nullptr) {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          sum_loss += XentLoss(label_[i], score[i]); // NOTE: does not work unless score is a probability
        }
      } else {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          sum_loss += XentLoss(label_[i], score[i]) * weights_[i]; // NOTE: does not work unless score is a probability
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double p = 0;
          objective->ConvertOutput(&score[i], &p);
          sum_loss += XentLoss(label_[i], p);
        }
      } else {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double p = 0;
          objective->ConvertOutput(&score[i], &p);
          sum_loss += XentLoss(label_[i], p) * weights_[i];
        }
      }
    }
    double loss = sum_loss / sum_weights_;
    return std::vector<double>(1, loss);
  }

	const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f; // negative means smaller loss is better, positive means larger loss is better
  }

private:
  /*! \brief Number of data points */
  data_size_t num_data_;
  /*! \brief Pointer to label */
  const float* label_;
  /*! \brief Pointer to weights */
  const float* weights_;
  /*! \brief Sum of weights */
  double sum_weights_;
  /*! \brief Name of this metric */
  std::vector<std::string> name_;
};

//
// CrossEntropyMetric1 : "xentropy1" : (optional) weights have a different meaning than for "xentropy"
//
class CrossEntropyMetric1 : public Metric {
public:
  explicit CrossEntropyMetric1(const MetricConfig&) {}
  virtual ~CrossEntropyMetric1() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    name_.emplace_back("xentropy1");
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    // ensure that labels are in interval [0, 1], interval ends included
    if (label_ == nullptr) {
       Log::Fatal("[%s]: label nullptr", __PRETTY_FUNCTION__);
    }
    for (data_size_t i = 0; i < num_data; ++i) {
      if (label_[i] < 0.0f || label_[i] > 1.0f) {
        Log::Fatal("[%s]: metric does not tolerate label [#%i] outside [0, 1]", __PRETTY_FUNCTION__, i);
      }
    }
    Log::Info("[%s:%s:metric]: labels passed interval [0, 1] check",  GetName()[0].c_str(), __func__);

    // check all weights are strictly positive; throw error if not
    if (weights_ != nullptr) {
      for (data_size_t i = 0; i < num_data; ++i) {
        if (weights_[i] <= 0) {
          Log::Fatal("[%s]: weight [#%i] required to be positive", __PRETTY_FUNCTION__, i);
        }
      }
    } else {
      // safe to not do anything
    }
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override {
    double sum_loss = 0.0f;
    if (objective == nullptr) {
      Log::Warning("[%s]: objective nullptr", __func__); // Q: when can this happen ?
      if (weights_ == nullptr) {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double hhat = std::log(1.0f + std::exp(score[i])); // auto-convert
          sum_loss += XentLoss1(label_[i], 1.0f, hhat);
          //double p = 1.0f / (1.0f + std::exp(-score[i])); // auto-convert
          //sum_loss += XentLoss(label_[i], p);
        }
      } else {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double hhat = std::log(1.0f + std::exp(score[i])); // auto-convert
          sum_loss += XentLoss1(label_[i], weights_[i], hhat);
        }
      }
    } else {
      if (weights_ == nullptr) {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double hhat = 0;
          objective->ConvertOutput(&score[i], &hhat); // NOTE: this only works if objective = "xentropy1"
          sum_loss += XentLoss1(label_[i], 1.0f, hhat);
        }
      } else {
        #pragma omp parallel for schedule(static) reduction(+:sum_loss)
        for (data_size_t i = 0; i < num_data_; ++i) {
          double hhat = 0;
          objective->ConvertOutput(&score[i], &hhat); // NOTE: this only works if objective = "xentropy1"
          sum_loss += XentLoss1(label_[i], weights_[i], hhat);
        }
      }
    }
    return std::vector<double>(1, sum_loss / static_cast<double>(num_data_));
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

private:
  /*! \brief Number of data points */
  data_size_t num_data_;
  /*! \brief Pointer to label */
  const float* label_;
  /*! \brief Pointer to weights */
  const float* weights_;
  /*! \brief Name of this metric */
  std::vector<std::string> name_;
};

} // end namespace LightGBM

#endif // end #ifndef LIGHTGBM_METRIC_XENTROPY_METRIC_HPP_
