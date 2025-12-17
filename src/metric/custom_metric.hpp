/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_METRIC_CUSTOM_METRIC_HPP_
#define LIGHTGBM_SRC_METRIC_CUSTOM_METRIC_HPP_

#include "multiclass_metric.hpp"

namespace LightGBM {

/*! \brief Focal Loss metric for multiclass task */
class FocalLossMetric: public MulticlassMetric<FocalLossMetric> {
 public:
  explicit FocalLossMetric(const Config& config) : MulticlassMetric<FocalLossMetric>(config) {}

  inline static double LossOnPoint(label_t label, std::vector<double>* score, const Config&) {
    constexpr double gamma = 1.0;  // TODO: make this configurable
    size_t k = static_cast<size_t>(label);
    auto& ref_score = *score;
    double p_k = ref_score[k];
    if (p_k > kEpsilon) {
      // Focal loss: -((1 - p_k)^gamma) * log(p_k)
      return -std::pow(1.0 - p_k, gamma) * std::log(p_k);
    } else {
      return -std::pow(1.0 - kEpsilon, gamma) * std::log(kEpsilon);
    }
  }

  inline static const std::string Name(const Config&) {
    return "focalloss";
  }
};

}  // namespace LightGBM
#endif   // LIGHTGBM_SRC_METRIC_CUSTOM_METRIC_HPP_
