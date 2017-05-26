#include "gbdt.h"

#include <LightGBM/utils/openmp_wrapper.h>

#include <LightGBM/utils/common.h>

#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/prediction_early_stop.h>

#include <ctime>

#include <sstream>
#include <chrono>
#include <string>
#include <vector>
#include <utility>

namespace LightGBM {

void GBDT::PredictRaw(const double* features, double* output, const PredictionEarlyStopInstance* earlyStop) const {
  const auto noEarlyStop = createPredictionEarlyStopInstance("none", PredictionEarlyStopConfig());
  if (earlyStop == nullptr)
  {
    earlyStop = &noEarlyStop;
  }

  int earlyStopRoundCounter = 0;
  for (int i = 0; i < num_iteration_for_pred_; ++i) {
    // predict all the trees for one iteration
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      output[k] += models_[i * num_tree_per_iteration_ + k]->Predict(features);
    }

    // check early stopping
    ++earlyStopRoundCounter;
    if (earlyStop->roundPeriod == earlyStopRoundCounter) {
      if (earlyStop->callbackFunction(output, num_tree_per_iteration_))
        return;
      earlyStopRoundCounter = 0;
    }
  }
}

void GBDT::Predict(const double* features, double* output, const PredictionEarlyStopInstance* earlyStop) const {
  PredictRaw(features, output, earlyStop);

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
