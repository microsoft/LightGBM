#ifndef LIGHTGBM_PREDICTION_EARLY_STOP_H_
#define LIGHTGBM_PREDICTION_EARLY_STOP_H_

#include <functional>
#include <string>

#include <LightGBM/export.h>

namespace LightGBM {

struct PredictionEarlyStopInstance {
  /// Callback function type for early stopping.
  /// Takes current prediction and number of elements in prediction
  /// @returns true if prediction should stop according to criterion
  using FunctionType = std::function<bool(const double*, int)>;

  FunctionType callbackFunction;  // callback function itself
  int          roundPeriod;       // call callbackFunction every `runPeriod` iterations
};

struct PredictionEarlyStopConfig {
  int roundPeriod;
  double marginThreshold;
};

/// Create an early stopping algorithm of type `type`, with given roundPeriod and margin threshold
LIGHTGBM_EXPORT PredictionEarlyStopInstance createPredictionEarlyStopInstance(const std::string& type,
                                                                              const PredictionEarlyStopConfig& config);

}   // namespace LightGBM

#endif // LIGHTGBM_PREDICTION_EARLY_STOP_H_
