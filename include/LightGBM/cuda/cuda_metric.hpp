/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_CUDA_CUDA_METRIC_HPP_
#define LIGHTGBM_CUDA_CUDA_METRIC_HPP_

#ifdef USE_CUDA

#include <LightGBM/metric.h>

namespace LightGBM {

template <typename HOST_METRIC>
class CUDAMetricInterface: public HOST_METRIC {
 public:
  explicit CUDAMetricInterface(const Config& config): HOST_METRIC(config) {
    cuda_labels_ = nullptr;
    cuda_weights_ = nullptr;
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    HOST_METRIC::Init(metadata, num_data);
    cuda_labels_ = metadata.cuda_metadata()->cuda_label();
    cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  }

  bool IsCUDAMetric() const { return true; }

 protected:
  const label_t* cuda_labels_;
  const label_t* cuda_weights_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_CUDA_CUDA_METRIC_HPP_
