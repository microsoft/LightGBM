#include "gbdt.h"

#include <LightGBM/utils/openmp_wrapper.h>

#include <LightGBM/utils/common.h>

#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>

#include <ctime>

#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <utility>

namespace LightGBM {

void GBDT::PredictRaw(const double* features, double* output) const {
  if (num_threads_ <= num_tree_per_iteration_) {
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      for (int i = 0; i < num_iteration_for_pred_; ++i) {
        output[k] += models_[i * num_tree_per_iteration_ + k]->Predict(features);
      }
    }
  } else {
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      double t = 0.0f;
      #pragma omp parallel for schedule(static) reduction(+:t)
      for (int i = 0; i < num_iteration_for_pred_; ++i) {
        t += models_[i * num_tree_per_iteration_ + k]->Predict(features);
      }
      output[k] = t;
    }
  }
}

void GBDT::Predict(const double* features, double* output) const {
  if (num_threads_ <= num_tree_per_iteration_) {
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      for (int i = 0; i < num_iteration_for_pred_; ++i) {
        output[k] += models_[i * num_tree_per_iteration_ + k]->Predict(features);
      }
    }
  } else {
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      double t = 0.0f;
      #pragma omp parallel for schedule(static) reduction(+:t)
      for (int i = 0; i < num_iteration_for_pred_; ++i) {
        t += models_[i * num_tree_per_iteration_ + k]->Predict(features);
      }
      output[k] = t;
    }
  }
  if (objective_function_ != nullptr) {
    objective_function_->ConvertOutput(output, output);
  }
}

void GBDT::PredictLeafIndex(const double* features, double* output) const {
  int total_tree = num_iteration_for_pred_ * num_tree_per_iteration_;
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < total_tree; ++i) {
    output[i] = models_[i]->PredictLeafIndex(features);
  }
}

}  // namespace LightGBM