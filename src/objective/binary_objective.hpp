#ifndef LIGHTGBM_OBJECTIVE_BINARY_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_BINARY_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>

#include <cstring>
#include <cmath>

namespace LightGBM {
/*!
* \brief Objective function for binary classification
*/
class BinaryLogloss: public ObjectiveFunction {
public:
  explicit BinaryLogloss(const ObjectiveConfig& config) {
    is_unbalance_ = config.is_unbalance;
    sigmoid_ = static_cast<double>(config.sigmoid);
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid parameter %f should be greater than zero", sigmoid_);
    }
    scale_pos_weight_ = static_cast<double>(config.scale_pos_weight);
  }
  ~BinaryLogloss() {}
  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
    data_size_t cnt_positive = 0;
    data_size_t cnt_negative = 0;
    // count for positive and negative samples
    #pragma omp parallel for schedule(static) reduction(+:cnt_positive, cnt_negative)
    for (data_size_t i = 0; i < num_data_; ++i) {
      if (label_[i] > 0) {
        ++cnt_positive;
      } else {
        ++cnt_negative;
      }
    }
    Log::Info("Number of positive: %d, number of negative: %d", cnt_positive, cnt_negative);
    // use -1 for negative class, and 1 for positive class
    label_val_[0] = -1;
    label_val_[1] = 1;
    // weight for label
    label_weights_[0] = 1.0f;
    label_weights_[1] = 1.0f;
    // if using unbalance, change the labels weight
    if (is_unbalance_ && cnt_positive > 0 && cnt_negative > 0) {
      if (cnt_positive > cnt_negative) {
        label_weights_[1] = 1.0f;
        label_weights_[0] = static_cast<double>(cnt_positive) / cnt_negative;
      } else {
        label_weights_[1] = static_cast<double>(cnt_negative) / cnt_positive;
        label_weights_[0] = 1.0f;
      }
    }
    label_weights_[1] *= scale_pos_weight_;
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        // get label and label weights
        const int is_pos = label_[i] > 0;
        const int label = label_val_[is_pos];
        const double label_weight = label_weights_[is_pos];
        // calculate gradients and hessians
        const double response = -label * sigmoid_ / (1.0f + std::exp(label * sigmoid_ * score[i]));
        const double abs_response = fabs(response);
        gradients[i] = static_cast<score_t>(response * label_weight);
        hessians[i] = static_cast<score_t>(abs_response * (sigmoid_ - abs_response) * label_weight);
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        // get label and label weights
        const int is_pos = label_[i] > 0;
        const int label = label_val_[is_pos];
        const double label_weight = label_weights_[is_pos];
        // calculate gradients and hessians
        const double response = -label * sigmoid_ / (1.0f + std::exp(label * sigmoid_ * score[i]));
        const double abs_response = fabs(response);
        gradients[i] = static_cast<score_t>(response * label_weight  * weights_[i]);
        hessians[i] = static_cast<score_t>(abs_response * (sigmoid_ - abs_response) * label_weight * weights_[i]);
      }
    }
  }

  const char* GetName() const override {
    return "binary";
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief True if using unbalance training */
  bool is_unbalance_;
  /*! \brief Sigmoid parameter */
  double sigmoid_;
  /*! \brief Values for positive and negative labels */
  int label_val_[2];
  /*! \brief Weights for positive and negative labels */
  double label_weights_[2];
  /*! \brief Weights for data */
  const float* weights_;
  double scale_pos_weight_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_BINARY_OBJECTIVE_HPP_
