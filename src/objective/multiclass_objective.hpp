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
  explicit MulticlassLogloss(const ObjectiveConfig& config)
        :label_int_(nullptr) {
    num_class_ = config.num_class;
  }
  
  ~MulticlassLogloss() {
    if (label_int_ != nullptr) { delete[] label_int_; }    
  }
  
  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
    label_int_ = new int[num_data_];
    for (int i = 0; i < num_data_; ++i){
        label_int_[i] = static_cast<int>(label_[i]); 
        if (label_int_[i] < 0 || label_int_[i] >= num_class_) {
            Log::Fatal("Label must be in [0, num_class)");
        }
    }
  }

  void GetGradients(const score_t* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        std::vector<score_t> rec(num_class_);
        for (int k = 0; k < num_class_; ++k){
            rec[k] = score[k * num_data_ + i];
        }
        Common::Softmax(&rec);  
        for (int k = 0; k < num_class_; ++k) {
          score_t p = rec[k];
          if (label_int_[i] == k) {
            gradients[k * num_data_ + i] = p - 1.0f;
          } else {
            gradients[k * num_data_ + i] = p;
          }
          hessians[k * num_data_ + i] = 2.0f * p * (1.0f - p);
        }  
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        std::vector<score_t> rec(num_class_);
        for (int k = 0; k < num_class_; ++k){
            rec[k] = score[k * num_data_ + i];
        }  
        Common::Softmax(&rec);
        for (int k = 0; k < num_class_; ++k) {
          float p = rec[k];
          if (label_int_[i] == k) {
            gradients[k * num_data_ + i] = (p - 1.0f) * weights_[i];
          } else {
            gradients[k * num_data_ + i] = p * weights_[i];
          }
          hessians[k * num_data_ + i] = 2.0f * p * (1.0f - p) * weights_[i];
        }
      }
    }
  }

  float GetSigmoid() const override {
    return -1.0f;
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Number of classes */
  int num_class_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Corresponding integers of label_ */
  int* label_int_;
  /*! \brief Weights for data */
  const float* weights_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_MULTICLASS_OBJECTIVE_HPP_
