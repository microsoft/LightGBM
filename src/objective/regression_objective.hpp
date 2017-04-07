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

  explicit RegressionL2loss(const std::vector<std::string>&) {

  }

  ~RegressionL2loss() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>(score[i] - label_[i]);
        hessians[i] = 1.0f;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>(score[i] - label_[i]) * weights_[i];
        hessians[i] = weights_[i];
      }
    }
  }

  const char* GetName() const override {
    return "regression";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool IsConstantHessian() const override {
    if (weights_ == nullptr) {
      return true;
    } else {
      return false;
    }
  }

  bool BoostFromAverage() const override { return true; }

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
  explicit RegressionL1loss(const ObjectiveConfig& config) {
    eta_ = static_cast<double>(config.gaussian_eta);
  }

  explicit RegressionL1loss(const std::vector<std::string>&) {

  }

  ~RegressionL1loss() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];
        if (diff >= 0.0f) {
          gradients[i] = 1.0f;
        } else {
          gradients[i] = -1.0f;
        }
        hessians[i] = static_cast<score_t>(Common::ApproximateHessianWithGaussian(score[i], label_[i], gradients[i], eta_));
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];
        if (diff >= 0.0f) {
          gradients[i] = weights_[i];
        } else {
          gradients[i] = -weights_[i];
        }
        hessians[i] = static_cast<score_t>(Common::ApproximateHessianWithGaussian(score[i], label_[i], gradients[i], eta_, weights_[i]));
      }
    }
  }

  const char* GetName() const override {
    return "regression_l1";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool BoostFromAverage() const override { return true; }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
  /*! \brief a parameter to control the width of Gaussian function to approximate hessian */
  double eta_;
};

/*!
* \brief Huber regression loss
*/
class RegressionHuberLoss: public ObjectiveFunction {
public:
  explicit RegressionHuberLoss(const ObjectiveConfig& config) {
    delta_ = static_cast<double>(config.huber_delta);
    eta_ = static_cast<double>(config.gaussian_eta);
  }

  explicit RegressionHuberLoss(const std::vector<std::string>&) {

  }

  ~RegressionHuberLoss() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];

        if (std::abs(diff) <= delta_) {
          gradients[i] = static_cast<score_t>(diff);
          hessians[i] = 1.0f;
        } else {
          if (diff >= 0.0f) {
            gradients[i] = static_cast<score_t>(delta_);
          } else {
            gradients[i] = static_cast<score_t>(-delta_);
          }
          hessians[i] = static_cast<score_t>(Common::ApproximateHessianWithGaussian(score[i], label_[i], gradients[i], eta_));
        }
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];

        if (std::abs(diff) <= delta_) {
          gradients[i] = static_cast<score_t>(diff * weights_[i]);
          hessians[i] = weights_[i];
        } else {
          if (diff >= 0.0f) {
            gradients[i] = static_cast<score_t>(delta_ * weights_[i]);
          } else {
            gradients[i] = static_cast<score_t>(-delta_ * weights_[i]);
          }
          hessians[i] = static_cast<score_t>(Common::ApproximateHessianWithGaussian(score[i], label_[i], gradients[i], eta_, weights_[i]));
        }
      }
    }
  }

  const char* GetName() const override {
    return "huber";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool BoostFromAverage() const override { return true; }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
  /*! \brief delta for Huber loss */
  double delta_;
  /*! \brief a parameter to control the width of Gaussian function to approximate hessian */
  double eta_;
};


// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
class RegressionFairLoss: public ObjectiveFunction {
public:
  explicit RegressionFairLoss(const ObjectiveConfig& config) {
    c_ = static_cast<double>(config.fair_c);
  }

  explicit RegressionFairLoss(const std::vector<std::string>&) {

  }

  ~RegressionFairLoss() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double x = score[i] - label_[i];
        gradients[i] = static_cast<score_t>(c_ * x / (std::fabs(x) + c_));
        hessians[i] = static_cast<score_t>(c_ * c_ / ((std::fabs(x) + c_) * (std::fabs(x) + c_)));
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double x = score[i] - label_[i];
        gradients[i] = static_cast<score_t>(c_ * x / (std::fabs(x) + c_) * weights_[i]);
        hessians[i] = static_cast<score_t>(c_ * c_ / ((std::fabs(x) + c_) * (std::fabs(x) + c_)) * weights_[i]);
      }
    }
  }

  const char* GetName() const override {
    return "fair";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool BoostFromAverage() const override { return true; }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
  /*! \brief c for Fair loss */
  double c_;
};


/*!
* \brief Objective function for Poisson regression
*/
class RegressionPoissonLoss: public ObjectiveFunction {
public:
  explicit RegressionPoissonLoss(const ObjectiveConfig& config) {
    max_delta_step_ = static_cast<double>(config.poisson_max_delta_step);
  }

  explicit RegressionPoissonLoss(const std::vector<std::string>&) {

  }

  ~RegressionPoissonLoss() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>(score[i] - label_[i]);
        hessians[i] = static_cast<score_t>(score[i] + max_delta_step_);
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>((score[i] - label_[i]) * weights_[i]);
        hessians[i] = static_cast<score_t>((score[i] + max_delta_step_) * weights_[i]);
      }
    }
  }

  const char* GetName() const override {
    return "poisson";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool BoostFromAverage() const override { return true; }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
  /*! \brief used to safeguard optimization */
  double max_delta_step_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
