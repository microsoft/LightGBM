/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_OBJECTIVE_CUSTOM_OBJECTIVE_HPP_
#define LIGHTGBM_SRC_OBJECTIVE_CUSTOM_OBJECTIVE_HPP_

#include "multiclass_objective.hpp"

namespace LightGBM {
/*!
* \brief Focal Loss objective function for multiclass classification
*/
class FocalLossSoftmax: public MulticlassSoftmax {
 public:
  explicit FocalLossSoftmax(const Config& config) : MulticlassSoftmax(config) {
    gamma_ = config.focal_gamma;
  }

  explicit FocalLossSoftmax(const std::vector<std::string>& strs) : MulticlassSoftmax(strs) {
    gamma_ = 1.0;  // default value for model loading
    for (auto str : strs) {
      auto tokens = Common::Split(str.c_str(), ':');
      if (tokens.size() == 2) {
        if (tokens[0] == std::string("focal_gamma")) {
          Common::Atof(tokens[1].c_str(), &gamma_);
        }
      }
    }
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      std::vector<double> rec;
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) private(rec)
      for (data_size_t i = 0; i < num_data_; ++i) {
        rec.resize(num_class_);
        for (int k = 0; k < num_class_; ++k) {
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          rec[k] = static_cast<double>(score[idx]);
        }
        Common::Softmax(&rec);
        double Z = 0.0;
        double D = 0.0;
        double p_label = 0.0;
        for (int k = 0; k < num_class_; ++k) {
          if (label_int_[i] == k) {
            auto p = rec[k];
            auto l = std::log(p);
            auto A = gamma_ * p * l + p - 1.0;
            auto B = std::pow(1.0 - p, gamma_ - 1.0);
            p_label = p;
            Z = A * B;
            D = gamma_ / (1.0 - p) * B * (-1.0 * A + l - p + 1.0);
          }
        }
        for (int k = 0; k < num_class_; ++k) {
          auto p = rec[k];
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          if (label_int_[i] == k) {
            gradients[idx] = static_cast<score_t>((1.0 - p) * Z);
            hessians[idx] = static_cast<score_t>(-1.0 * Z * p * (1.0 - p) + p * (1.0 - p) * (1.0 - p) * D);
          } else {
            gradients[idx] = static_cast<score_t>(-1.0 * p * Z);
            hessians[idx] = static_cast<score_t>(-1.0 * Z * p * (1.0 - p) + p * p * p_label * D);
          }
        }
      }
    } else {
      std::vector<double> rec;
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) private(rec)
      for (data_size_t i = 0; i < num_data_; ++i) {
        rec.resize(num_class_);
        for (int k = 0; k < num_class_; ++k) {
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          rec[k] = static_cast<double>(score[idx]);
        }
        Common::Softmax(&rec);
        double Z = 0.0;
        double D = 0.0;
        double p_label = 0.0;
        for (int k = 0; k < num_class_; ++k) {
          if (label_int_[i] == k) {
            auto p = rec[k];
            auto l = std::log(p);
            auto A = gamma_ * p * l + p - 1.0;
            auto B = std::pow(1.0 - p, gamma_ - 1.0);
            p_label = p;
            Z = A * B;
            D = gamma_ / (1.0 - p) * B * (-1.0 * A + l - p + 1.0);
          }
        }
        for (int k = 0; k < num_class_; ++k) {
          auto p = rec[k];
          size_t idx = static_cast<size_t>(num_data_) * k + i;
          if (label_int_[i] == k) {
            gradients[idx] = static_cast<score_t>((1.0 - p) * Z * weights_[i]);
            hessians[idx] = static_cast<score_t>((-1.0 * Z * p * (1.0 - p) + p * (1.0 - p) * (1.0 - p) * D) * weights_[i]);
          } else {
            gradients[idx] = static_cast<score_t>(-1.0 * p * Z * weights_[i]);
            hessians[idx] = static_cast<score_t>((-1.0 * Z * p * (1.0 - p) + p * p * p_label * D) * weights_[i]);
          }
        }
      }
    }
  }

  const char* GetName() const override {
    return "focalloss";
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName() << " ";
    str_buf << "num_class:" << num_class_ << " ";
    str_buf << "focal_gamma:" << gamma_;
    return str_buf.str();
  }

 private:
  double gamma_;
};

}  // namespace LightGBM
#endif   // LIGHTGBM_SRC_OBJECTIVE_CUSTOM_OBJECTIVE_HPP_
