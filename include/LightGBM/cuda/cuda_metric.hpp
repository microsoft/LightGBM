/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_CUDA_CUDA_METRIC_HPP_
#define LIGHTGBM_CUDA_CUDA_METRIC_HPP_

#ifdef USE_CUDA_EXP

#include <LightGBM/metric.h>

namespace LightGBM {

template <typename HOST_METRIC>
class CUDAMetricInterface: public HOST_METRIC {
 public:
  explicit CUDAMetricInterface(const Config& config): HOST_METRIC(config) {}

  bool IsCUDAMetric() const { return true; }

 protected:
  const label_t* cuda_labels_;
  const label_t* cuda_weights_;
};

}  // namespace LightGBM

#endif  // USE_CUDA_EXP

#endif  // LIGHTGBM_CUDA_CUDA_METRIC_HPP_
