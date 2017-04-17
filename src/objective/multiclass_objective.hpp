#ifndef LIGHTGBM_OBJECTIVE_MULTICLASS_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_MULTICLASS_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>

#include <cstring>
#include <cmath>
#include <vector>

#include "binary_objective.hpp"

namespace LightGBM {
/*!
* \brief Objective function for multiclass classification, use softmax as objective functions
*/
class MulticlassSoftmax: public ObjectiveFunction {
public:
  explicit MulticlassSoftmax(const ObjectiveConfig& config) {
    num_class_ = config.num_class;
  }

  ~MulticlassSoftmax() {

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
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      std::vector<double> rec;
      #pragma omp parallel for schedule(static) private(rec)
      for (data_size_t i = 0; i < num_data_; ++i) {
        rec.resize(num_class_);
        for (int k = 0; k < num_class_; ++k) {
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          rec[k] = static_cast<double>(score[idx]);
        }
        Common::Softmax(&rec);
        for (int k = 0; k < num_class_; ++k) {
          auto p = rec[k];
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          if (label_int_[i] == k) {
            gradients[idx] = static_cast<score_t>(p - 1.0f);
          } else {
            gradients[idx] = static_cast<score_t>(p);
          }
          hessians[idx] = static_cast<score_t>(2.0f * p * (1.0f - p));
        }
      }
    } else {
      std::vector<double> rec;
      #pragma omp parallel for schedule(static) private(rec)
      for (data_size_t i = 0; i < num_data_; ++i) {
        rec.resize(num_class_);
        for (int k = 0; k < num_class_; ++k) {
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          rec[k] = static_cast<double>(score[idx]);
        }
        Common::Softmax(&rec);
        for (int k = 0; k < num_class_; ++k) {
          auto p = rec[k];
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          if (label_int_[i] == k) {
            gradients[idx] = static_cast<score_t>((p - 1.0f) * weights_[i]);
          } else {
            gradients[idx] = static_cast<score_t>((p) * weights_[i]);
          }
          hessians[idx] = static_cast<score_t>((2.0f * p * (1.0f - p))* weights_[i]);
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
};

/*!
* \brief Objective function for multiclass classification, use one-vs-all binary objective function
*/
class MulticlassOVA: public ObjectiveFunction {
public:
  explicit MulticlassOVA(const ObjectiveConfig& config) {
    num_class_ = config.num_class;
    for (int i = 0; i < num_class_; ++i) {
      binary_loss_.emplace_back(
        new BinaryLogloss(config, [i](float label) { return static_cast<int>(label) == i; }));
    }
  }

  ~MulticlassOVA() {

  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    for (int i = 0; i < num_class_; ++i) {
      binary_loss_[i]->Init(metadata, num_data);
    }
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    for (int i = 0; i < num_class_; ++i) {
      int64_t bias = static_cast<int64_t>(num_data_) * i;
      binary_loss_[i]->GetGradients(score + bias, gradients + bias, hessians + bias);
    }
  }

  const char* GetName() const override {
    return "multiclassova";
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Number of classes */
  int num_class_;
  std::vector<std::unique_ptr<BinaryLogloss>> binary_loss_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_MULTICLASS_OBJECTIVE_HPP_
