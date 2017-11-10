#ifndef LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_

#include <LightGBM/meta.h>

#include <LightGBM/objective_function.h>
#include <LightGBM/utils/common.h>

namespace LightGBM {

/*!
* \brief Objective function for regression
*/
class RegressionL2loss: public ObjectiveFunction {
public:
  explicit RegressionL2loss(const ObjectiveConfig& config) {
    sqrt_ = config.reg_sqrt;
  }

  explicit RegressionL2loss(const std::vector<std::string>& strs) {
    sqrt_ = false;
    for (auto str : strs) {
      if (str == std::string("sqrt")) {
        sqrt_ = true;
      }
    }
  }

  ~RegressionL2loss() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    if (sqrt_) {
      trans_label_.resize(num_data_);
      for (data_size_t i = 0; i < num_data; ++i) {
        trans_label_[i] = std::copysign(std::sqrt(std::fabs(label_[i])), label_[i]);
      }
      label_ = trans_label_.data();
    }
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

  void ConvertOutput(const double* input, double* output) const override {
    if (sqrt_) {
      output[0] = std::copysign(input[0] * input[0], input[0]);
    } else {
      output[0] = input[0];
    }
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    if (sqrt_) {
      str_buf << " sqrt";
    }
    return str_buf.str();
  }

  bool IsConstantHessian() const override {
    if (weights_ == nullptr) {
      return true;
    } else {
      return false;
    }
  }

  bool BoostFromAverage() const override { 
    if (sqrt_) {
      return false;
    } else {
      return true;
    }
  }

protected:
  bool sqrt_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
  std::vector<float> trans_label_;
};

/*!
* \brief L1 regression loss
*/
class RegressionL1loss: public RegressionL2loss {
public:
  explicit RegressionL1loss(const ObjectiveConfig& config): RegressionL2loss(config) {
    eta_ = static_cast<double>(config.gaussian_eta);
  }

  explicit RegressionL1loss(const std::vector<std::string>& strs): RegressionL2loss(strs) {

  }

  ~RegressionL1loss() {}

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

  bool IsConstantHessian() const override {
    return false;
  }

private:
  double eta_;
};

/*!
* \brief Huber regression loss
*/
class RegressionHuberLoss: public RegressionL2loss {
public:
  explicit RegressionHuberLoss(const ObjectiveConfig& config): RegressionL2loss(config) {
    alpha_ = static_cast<double>(config.alpha);
    eta_ = static_cast<double>(config.gaussian_eta);
  }

  explicit RegressionHuberLoss(const std::vector<std::string>& strs): RegressionL2loss(strs) {

  }

  ~RegressionHuberLoss() {
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];

        if (std::abs(diff) <= alpha_) {
          gradients[i] = static_cast<score_t>(diff);
          hessians[i] = 1.0f;
        } else {
          if (diff >= 0.0f) {
            gradients[i] = static_cast<score_t>(alpha_);
          } else {
            gradients[i] = static_cast<score_t>(-alpha_);
          }
          hessians[i] = static_cast<score_t>(Common::ApproximateHessianWithGaussian(score[i], label_[i], gradients[i], eta_));
        }
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];

        if (std::abs(diff) <= alpha_) {
          gradients[i] = static_cast<score_t>(diff * weights_[i]);
          hessians[i] = weights_[i];
        } else {
          if (diff >= 0.0f) {
            gradients[i] = static_cast<score_t>(alpha_ * weights_[i]);
          } else {
            gradients[i] = static_cast<score_t>(-alpha_ * weights_[i]);
          }
          hessians[i] = static_cast<score_t>(Common::ApproximateHessianWithGaussian(score[i], label_[i], gradients[i], eta_, weights_[i]));
        }
      }
    }
  }

  const char* GetName() const override {
    return "huber";
  }

  bool IsConstantHessian() const override {
    return false;
  }

private:
  /*! \brief delta for Huber loss */
  double alpha_;
  /*! \brief a parameter to control the width of Gaussian function to approximate hessian */
  double eta_;
};


// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
class RegressionFairLoss: public RegressionL2loss {
public:
  explicit RegressionFairLoss(const ObjectiveConfig& config): RegressionL2loss(config) {
    c_ = static_cast<double>(config.fair_c);
  }

  explicit RegressionFairLoss(const std::vector<std::string>& strs): RegressionL2loss(strs) {

  }

  ~RegressionFairLoss() {}

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

  bool IsConstantHessian() const override {
    return false;
  }

private:
  /*! \brief c for Fair loss */
  double c_;
};


/*!
* \brief Objective function for Poisson regression
*/
class RegressionPoissonLoss: public RegressionL2loss {
public:
  explicit RegressionPoissonLoss(const ObjectiveConfig& config): RegressionL2loss(config) {
    max_delta_step_ = static_cast<double>(config.poisson_max_delta_step);
  }

  explicit RegressionPoissonLoss(const std::vector<std::string>& strs): RegressionL2loss(strs) {

  }

  ~RegressionPoissonLoss() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RegressionL2loss::Init(metadata, num_data);
    // Safety check of labels
    float miny;
    double sumy;
    Common::ObtainMinMaxSum(label_, num_data_, &miny, (float*)nullptr, &sumy);
    if (miny < 0.0f) {
      Log::Fatal("[%s]: at least one target label is negative.", GetName());
    }
    if (sumy == 0.0f) {
      Log::Fatal("[%s]: sum of labels is zero.", GetName());
    }
  }

  /* Parametrize with unbounded internal score "f"; then
   *  loss = exp(f) - label * f
   *  grad = exp(f) - label
   *  hess = exp(f)
   *
   * And the output is exp(f); so the associated metric get s=exp(f)
   * so that its loss = s - label * log(s); a little awkward maybe.
   *
   */
  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double ef = std::exp(score[i]);
        gradients[i] = static_cast<score_t>(ef - label_[i]);
        hessians[i] = static_cast<score_t>(ef);
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double ef = std::exp(score[i]);
        gradients[i] = static_cast<score_t>((ef - label_[i]) * weights_[i]);
        hessians[i] = static_cast<score_t>(ef * weights_[i]);
      }
    }
  }

  void ConvertOutput(const double* input, double* output) const override {
    RegressionL2loss::ConvertOutput(input, output);
    output[0] = std::exp(input[0]);
  }

  const char* GetName() const override {
    return "poisson";
  }

  bool GetCustomAverage(double *initscore) const override {
    if (initscore == nullptr) return false;
    double sumw = 0.0f;
    double sumy = 0.0f;
    if (weights_ == nullptr) {
      for (data_size_t i = 0; i < num_data_; i++) {
        sumy += label_[i];
      }
      sumw = static_cast<double>(num_data_);
    } else {
      for (data_size_t i = 0; i < num_data_; i++) {
        sumy += weights_[i] * label_[i];
        sumw += weights_[i];
      }
    }
    const double yavg = sumy / sumw;
    *initscore = std::log(yavg);
    Log::Info("[%s:%s]: yavg=%f -> initscore=%f",  GetName(), __func__, yavg, *initscore);
    return true;
  }

  bool IsConstantHessian() const override {
    return false;
  }

private:
  /*! \brief used to safeguard optimization */
  double max_delta_step_;
};

class RegressionQuantileloss : public RegressionL2loss {
public:
  explicit RegressionQuantileloss(const ObjectiveConfig& config): RegressionL2loss(config) {
    alpha_ = static_cast<score_t>(config.alpha);
  }

  explicit RegressionQuantileloss(const std::vector<std::string>& strs): RegressionL2loss(strs) {

  }

  ~RegressionQuantileloss() {}

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        score_t delta = static_cast<score_t>(score[i] - label_[i]);
        if (delta >= 0) {
          gradients[i] = (1.0f - alpha_);
        } else {
          gradients[i] = -alpha_;
        }
        hessians[i] = 1.0f;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        score_t delta = static_cast<score_t>(score[i] - label_[i]);
        if (delta >= 0) {
          gradients[i] = (1.0f - alpha_) * weights_[i];
        } else {
          gradients[i] = -alpha_ * weights_[i];
        }
        hessians[i] = weights_[i];
      }
    }
  }

  const char* GetName() const override {
    return "quantile";
  }

private:
  score_t alpha_;
};

class RegressionQuantileL2loss : public RegressionL2loss {
public:
  explicit RegressionQuantileL2loss(const ObjectiveConfig& config) : RegressionL2loss(config) {
    alpha_ = static_cast<score_t>(config.alpha);
  }

  explicit RegressionQuantileL2loss(const std::vector<std::string>& strs) : RegressionL2loss(strs) {

  }

  ~RegressionQuantileL2loss() {}

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        score_t delta = static_cast<score_t>(score[i] - label_[i]);
        if (delta > 0) {
          gradients[i] = (1.0f - alpha_) * delta;
          hessians[i] = (1.0f - alpha_);
        } else {
          gradients[i] = alpha_ * delta;
          hessians[i] = alpha_;
        }

      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        score_t delta = static_cast<score_t>(score[i] - label_[i]);
        if (delta > 0) {
          gradients[i] = (1.0f - alpha_) * delta * weights_[i];
          hessians[i] = (1.0f - alpha_) * weights_[i];
        } else {
          gradients[i] = alpha_ * delta * weights_[i];
          hessians[i] = alpha_ * weights_[i];
        }
      }
    }
  }

  bool IsConstantHessian() const override {
    return false;
  }

  const char* GetName() const override {
    return "quantile_l2";
  }

private:
  score_t alpha_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
