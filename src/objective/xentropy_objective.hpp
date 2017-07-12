#ifndef LIGHTGBM_OBJECTIVE_XENTROPY_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_XENTROPY_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>

#include <cstring>
#include <cmath>

namespace LightGBM {
/*!
* \brief Objective function for cross-entropy (with optional linear weights)
*/
class CrossEntropy: public ObjectiveFunction {
public:
  explicit CrossEntropy(const ObjectiveConfig&) {
  }

  explicit CrossEntropy(const std::vector<std::string>&) {
  }

  ~CrossEntropy() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    // ensure that labels are in interval [0, 1], interval ends included
    for (data_size_t i = 0; i < num_data; ++i) {
      if (label_[i] < 0.0f || label_[i] > 1.0f) {
        Log::Fatal("[%s]: does not tolerate label [#%i] outside [0, 1]",  GetName(), i);
      }
    }

    Log::Info("[%s:%s]: labels passed interval [0, 1] check (objective)",  GetName(), __func__);

    if (weights_ != nullptr) {
      // ensure that every weight is non-negative; and count number of zero weights
      data_size_t cnt_zero_weights = 0;
      for (data_size_t i = 0; i < num_data; ++i) {
        if (weights_[i] == 0.0f) {
          ++cnt_zero_weights;
        } else if (weights_[i] < 0.0f) {
          Log::Fatal("[%s]: does not tolerate negative weight [#%i]", GetName(), i);
        }
      }
      if (cnt_zero_weights > 0) {
        Log::Warning("[%s]: counted #%i zero weights",  GetName(), cnt_zero_weights);
      }
    }
    
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      // compute pointwise gradients and hessians with implied unit weights
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double z = 1.0f / (1.0f + std::exp(-score[i]));
        gradients[i] = static_cast<score_t>(z - label_[i]);
        hessians[i] = static_cast<score_t>(z * (1.0f - z));
      }
    } else {
      // compute pointwise gradients and hessians with given weights
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double z = 1.0f / (1.0f + std::exp(-score[i]));
        gradients[i] = static_cast<score_t>((z - label_[i]) * weights_[i]);
        hessians[i] = static_cast<score_t>(z * (1.0f - z) * weights_[i]);
      }
    }
  }

  const char* GetName() const override {
    return "xentropy";
  }

  void ConvertOutput(const double* input, double* output) const override {
    output[0] = 1.0f / (1.0f + std::exp(-input[0]));
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool BoostFromAverage() const override { return true; }

private:
  /*! \brief Number of data points */
  data_size_t num_data_;
  /*! \brief Pointer for label */
  const float* label_;
  /*! \brief Weights for data */
  const float* weights_;
};

/*!
* \brief Objective function for 1st variant of cross-entropy (with optional "nonlinear" weights)
*/
class CrossEntropy1: public ObjectiveFunction {
public:
  explicit CrossEntropy1(const ObjectiveConfig&) {
    min_weight_ = max_weight_ = 0.0f;
  }

  explicit CrossEntropy1(const std::vector<std::string>&) {
  }

  ~CrossEntropy1() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
    // ensure that labels are in interval [0, 1], interval ends included
    for (data_size_t i = 0; i < num_data; ++i) {
      if (label_[i] < 0.0f || label_[i] > 1.0f) {
        Log::Fatal("[%s]: does not tolerate label [#%i] outside [0, 1]",  GetName(), i);
      }
    }
    Log::Info("[%s:%s]: labels passed interval [0, 1] check (objective)",  GetName(), __func__);
    if (weights_ != nullptr) {
      min_weight_ = weights_[0];
      max_weight_ = weights_[0];
      // ensure that every weight is strictly positive; and determine ratio : largest/smallest weight
      for (data_size_t i = 0; i < num_data; ++i) {
        if (weights_[i] <= 0.0f) {
          Log::Fatal("[%s]: does not tolerate non-positive weight [#%i]", GetName(), i);
        }
        if (weights_[i] < min_weight_) min_weight_ = weights_[i];
        if (weights_[i] > max_weight_) max_weight_ = weights_[i];
      }
      // Issue an info statement about this ratio; and possibly warn if excessively large ?!
      double weight_ratio = max_weight_ / min_weight_;
      Log::Info("[%s:%s]: min, max weights = %f, %f; ratio = %f", 
                GetName(), __func__,
                min_weight_, max_weight_,
                weight_ratio);
    } else {
      // all weights are implied to be unity; no need to do anything 
    }
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      // compute pointwise gradients and hessians with implied unit weights; exactly equivalent to CrossEntropy with unit weights
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double z = 1.0f / (1.0f + std::exp(-score[i]));
        gradients[i] = static_cast<score_t>(z - label_[i]);
        hessians[i] = static_cast<score_t>(z * (1.0f - z));
      }
    } else {
      // compute pointwise gradients and hessians with given weights
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double w = weights_[i];
        const double y = label_[i];
        const double epf = std::exp(score[i]);
        const double hhat = std::log(1.0f + epf);
        const double z = 1.0f - std::exp(-w*hhat);
        const double enf = 1.0f / epf; // = std::exp(-score[i]);
        gradients[i] = static_cast<score_t>((1.0f - y / z) * w / (1.0f + enf));
        const double c = 1.0f / (1.0f - z);
        double d = 1.0f + epf;
        const double a = w * epf / (d * d);
        d = c - 1.0f;
        const double b = (c / (d * d) ) * (1.0f + w * epf - c);
        hessians[i] = static_cast<score_t>(a * (1.0f + y * b));
      }
    }
  }

  const char* GetName() const override {
    return "xentropy1";
  }

  // ATTENTION: this is NOT the true probability conversion for a point (unless all weights=1).
  // Let output from this member be z; then prob(z) = 1-(1-z)^w is the probability for a point with weight w>0.
  // But this member has no access to w; so this has to be done in the associated metric class.

  //
  // ATTENTION: the function output is the "normalized exponential parameter" > 0, not the probability
  //
  void ConvertOutput(const double* input, double* output) const override {
  //  output[0] = 1.0f / (1.0f + std::exp(-input[0]));
    output[0] = std::log(1.0f + std::exp(input[0]));
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  // TODO: default boost from different (weighted) average; not quite the same ...
  bool BoostFromAverage() const override { return true; }

private:
  /*! \brief Number of data points */
  data_size_t num_data_;
  /*! \brief Pointer for label */
  const float* label_;
  /*! \brief Weights for data */
  const float* weights_;
  /*! \brief Minimum weight found during init */
  float min_weight_;
  /*! \brief Maximum weight found during init */
  float max_weight_;
};

}  // end namespace LightGBM

#endif   // end #ifndef LIGHTGBM_OBJECTIVE_XENTROPY_OBJECTIVE_HPP_
