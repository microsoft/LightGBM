#ifndef LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>

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


class RegressionL1loss: public ObjectiveFunction {
public:
  explicit RegressionL1loss(const ObjectiveConfig& config) {}

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
        const double diff = score[i] - label_[i];
        if (diff >= 0.0) {
          gradients[i] = 1.0;
        } else {
          gradients[i] = -1.0;
        }
        hessians[i] = 0.0;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];
        if (diff >= 0.0) {
          gradients[i] = weights_[i];
        } else {
          gradients[i] = -weights_[i];
        }
        hessians[i] = 0.0;
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


class RegressionLHuberLoss: public ObjectiveFunction {
public:
  explicit RegressionLHuberLoss(const ObjectiveConfig& config) {
    delta_ = config.huber_delta;
  }

  ~RegressionLHuberLoss() {
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
        const double diff = score[i] - label_[i];
        if (std::abs(diff) <= delta_) {
          gradients[i] = diff;
          hessians[i] = 1.0;
        } else {
            if (diff >= 0.0) {
              gradients[i] = delta_;
            } else {
              gradients[i] = -delta_;
            }
            hessians[i] = 0.0;
        }
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];
        if (std::abs(diff) <= delta_) {
          gradients[i] = diff * weights_[i];
          hessians[i] = weights_[i];
        } else {
            if (diff >= 0.0) {
              gradients[i] = delta_ * weights_[i];
            } else {
              gradients[i] = -delta_ * weights_[i];
            }
            hessians[i] = 0.0;
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
  double delta_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
