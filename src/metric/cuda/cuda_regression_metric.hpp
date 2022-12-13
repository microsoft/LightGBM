/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_
#define LIGHTGBM_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_

#ifdef USE_CUDA_EXP

#include <LightGBM/cuda/cuda_metric.hpp>
#include <LightGBM/cuda/cuda_utils.h>

#include <vector>

#include "../regression_metric.hpp"

#define NUM_DATA_PER_EVAL_THREAD (1024)

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
class CUDARegressionMetricInterface: public CUDAMetricInterface<HOST_METRIC> {
 public:
  explicit CUDARegressionMetricInterface(const Config& config): CUDAMetricInterface<HOST_METRIC>(config) {}

  virtual ~CUDARegressionMetricInterface() {}

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;

 protected:
  double LaunchEvalKernel(const double* score_convert) const;

  CUDAVector<double> score_convert_buffer_;
  CUDAVector<double> reduce_block_buffer_;
  CUDAVector<double> reduce_block_buffer_inner_;
};

class CUDARMSEMetric: public CUDARegressionMetricInterface<RMSEMetric, CUDARMSEMetric> {
 public:
  explicit CUDARMSEMetric(const Config& config);

  virtual ~CUDARMSEMetric() {}

  __device__ static double MetricOnPointCUDA(label_t label, double score) {
    return (score - label) * (score - label);
  }
};

class CUDAL2Metric : public CUDARegressionMetricInterface<L2Metric, CUDAL2Metric> {
 public:
  explicit CUDAL2Metric(const Config& config);

  virtual ~CUDAL2Metric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score) {
    return (score - label) * (score - label);
  }
};

}  // namespace LightGBM

#endif  // USE_CUDA_EXP

#endif  // LIGHTGBM_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_
