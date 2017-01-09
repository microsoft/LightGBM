#ifndef LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>
#include <LightGBM/utils/common.h>

namespace LightGBM {
/*!
* \brief Objective function for regression
*/
class RegressionL2loss: public ObjectiveFunction {
public:
  explicit RegressionL2loss(const ObjectiveConfig&) {
  }

  ~RegressionL2loss() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const score_t* score, score_t* gradients,
    score_t* hessians) const override {
    if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = (score[i] - label_[i]);
        hessians[i] = 1.0;
      }
    } else {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = (score[i] - label_[i]) * weights_[i];
        hessians[i] = weights_[i];
      }
    }
  }

  const char* GetName() const override {
    return "regression";
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
};

/*!
* \brief L1 regression loss
*/
class RegressionL1loss: public ObjectiveFunction {
public:
  explicit RegressionL1loss(const ObjectiveConfig&) {}

  ~RegressionL1loss() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const score_t* score, score_t* gradients,
    score_t* hessians) const override {
    if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const score_t diff = score[i] - label_[i];
        if (diff >= 0.0f) {
          gradients[i] = 1.0f;
        } else {
          gradients[i] = -1.0f;
        }
        hessians[i] = static_cast<score_t>(Common::ApproximateHessianWithGaussian(score[i], label_[i], gradients[i]));
      }
    } else {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const score_t diff = score[i] - label_[i];
        if (diff >= 0.0f) {
          gradients[i] = weights_[i];
        } else {
          gradients[i] = -weights_[i];
        }
        hessians[i] = static_cast<score_t>(Common::ApproximateHessianWithGaussian(score[i], label_[i], gradients[i], weights_[i]));
      }
    }
  }

  const char* GetName() const override {
    return "regression_l1";
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
};

/*!
* \brief Huber regression loss
*/
class RegressionHuberLoss: public ObjectiveFunction {
public:
  explicit RegressionHuberLoss(const ObjectiveConfig& config) {
    delta_ = static_cast<score_t>(config.huber_delta);
  }

  ~RegressionHuberLoss() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const score_t* score, score_t* gradients,
    score_t* hessians) const override {
    if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const score_t diff = score[i] - label_[i];

        if (std::abs(diff) <= delta_) {
          gradients[i] = diff;
          hessians[i] = 1.0f;
        } else {
          if (diff >= 0.0f) {
            gradients[i] = delta_;
          } else {
            gradients[i] = -delta_;
          }
          hessians[i] = static_cast<score_t>(Common::ApproximateHessianWithGaussian(score[i], label_[i], gradients[i]));
        }
      }
    } else {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const score_t diff = score[i] - label_[i];

        if (std::abs(diff) <= delta_) {
          gradients[i] = diff * weights_[i];
          hessians[i] = weights_[i];
        } else {
          if (diff >= 0.0f) {
            gradients[i] = delta_ * weights_[i];
          } else {
            gradients[i] = -delta_ * weights_[i];
          }
          hessians[i] = static_cast<score_t>(Common::ApproximateHessianWithGaussian(score[i], label_[i], gradients[i], weights_[i]));
        }
      }
    }
  }

  const char* GetName() const override {
    return "huber";
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
  /*! \brief delta for Huber loss */
  score_t delta_;
};


// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
class RegressionFairLoss: public ObjectiveFunction {
public:
  explicit RegressionFairLoss(const ObjectiveConfig& config) {
    c_ = static_cast<score_t>(config.fair_c);
  }

  ~RegressionFairLoss() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const score_t* score, score_t* gradients,
    score_t* hessians) const override {
    if (weights_ == nullptr) {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const score_t x = score[i] - label_[i];
        gradients[i] = c_ * x / (std::fabs(x) + c_);
        hessians[i] = c_ * c_ / ((std::fabs(x) + c_) * (std::fabs(x) + c_));
      }
    } else {
#pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const score_t x = score[i] - label_[i];
        gradients[i] = c_ * x / (std::fabs(x) + c_);
        gradients[i] *= weights_[i];
        hessians[i] = c_ * c_ / ((std::fabs(x) + c_) * (std::fabs(x) + c_));
        hessians[i] *= weights_[i];
      }
    }
  }

  const char* GetName() const override {
    return "fair";
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
  /*! \brief c for Fair loss */
  score_t c_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
