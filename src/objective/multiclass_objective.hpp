#ifndef LIGHTGBM_OBJECTIVE_MULTICLASS_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_MULTICLASS_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>

#include <cstring>
#include <cmath>

namespace LightGBM {
/*!
* \brief Objective function for multiclass classification
*/
class MulticlassLogloss: public ObjectiveFunction {
public:
  explicit MulticlassLogloss(const ObjectiveConfig& config) {
    num_class_ = config.num_class;
    is_unbalance_ = config.is_unbalance;
  }

  ~MulticlassLogloss() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
    label_int_.resize(num_data_);
    for (int i = 0; i < num_data_; ++i) {
      label_int_[i] = static_cast<int>(label_[i]);
      if (label_int_[i] < 0 || label_int_[i] >= num_class_) {
        Log::Fatal("Label must be in [0, %d), but found %d in label", num_class_, label_int_[i]);
      }
    }
    label_pos_weights_ = std::vector<float>(num_class_, 1);
    label_neg_weights_ = std::vector<float>(num_class_, 1);
    if (is_unbalance_) {
      std::vector<int> cnts(num_class_, 0);
      for (int i = 0; i < num_data_; ++i) {
        ++cnts[label_int_[i]];
      }
      int average_cnt = num_data_ / num_class_;
      for (int i = 0; i < num_class_; ++i) {
        int cnt_cur = cnts[i];
        int cnt_other = (average_cnt - cnts[i]);
        if (cnt_cur > cnt_other) {
          label_pos_weights_[i] = 1.0f;
          label_neg_weights_[i] = static_cast<float>(cnt_cur) / cnt_other;
        } else {
          label_pos_weights_[i] = static_cast<float>(cnt_other) / cnt_cur;
          label_neg_weights_[i] = 1.0f;
        }
      }
    } 
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        std::vector<double> rec(num_class_);
        for (int k = 0; k < num_class_; ++k){
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          rec[k] = static_cast<double>(score[idx]);
        }
        Common::Softmax(&rec);
        for (int k = 0; k < num_class_; ++k) {
          auto p = rec[k];
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          if (label_int_[i] == k) {
            gradients[idx] = static_cast<score_t>(p - 1.0f) * label_pos_weights_[k];
            hessians[idx] = static_cast<score_t>(2.0f * p * (1.0f - p))* label_pos_weights_[k];
          } else {
            gradients[idx] = static_cast<score_t>(p) * label_neg_weights_[k];
            hessians[idx] = static_cast<score_t>(2.0f * p * (1.0f - p)) * label_neg_weights_[k];
          }
        }
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        std::vector<double> rec(num_class_);
        for (int k = 0; k < num_class_; ++k){
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          rec[k] = static_cast<double>(score[idx]);
        }
        Common::Softmax(&rec);
        for (int k = 0; k < num_class_; ++k) {
          auto p = rec[k];
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          if (label_int_[i] == k) {
            gradients[idx] = static_cast<score_t>((p - 1.0f) * weights_[i]) * label_pos_weights_[k];
            hessians[idx] = static_cast<score_t>(2.0f * p * (1.0f - p) * weights_[i]) * label_pos_weights_[k];
          } else {
            gradients[idx] = static_cast<score_t>(p * weights_[i]) * label_neg_weights_[k];
            hessians[idx] = static_cast<score_t>(2.0f * p * (1.0f - p) * weights_[i]) * label_neg_weights_[k];
          }
          
        }
      }
    }
  }

  const char* GetName() const override {
    return "multiclass";
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Number of classes */
  int num_class_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Corresponding integers of label_ */
  std::vector<int> label_int_;
  /*! \brief Weights for data */
  const float* weights_;
  /*! \brief Weights for label */
  std::vector<float> label_pos_weights_;
  std::vector<float> label_neg_weights_;
  bool is_unbalance_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_MULTICLASS_OBJECTIVE_HPP_
