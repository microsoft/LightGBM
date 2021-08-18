/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_
#define LIGHTGBM_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_

#include "cuda_metric.hpp"
#include "../binary_metric.hpp"

#define EVAL_BLOCK_SIZE_BINARY_METRIC (1024)

namespace LightGBM {

template <typename CUDAPointWiseLossCalculator>
class CUDABinaryMetric : public CUDAMetricInterface, public BinaryMetric<CUDAPointWiseLossCalculator> {
 public:
  explicit CUDABinaryMetric(const Config& config);

  ~CUDABinaryMetric();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;

 protected:
  void LaunchEvalKernel(const double* score) const;

  void LaunchEvalKernelInner(const double* score) const;

  const label_t* cuda_label_;
  const label_t* cuda_weights_;
  double* cuda_sum_loss_buffer_;
  double* cuda_score_convert_buffer_;
  double* cuda_sum_loss_;
};

class CUDABinaryLoglossMetric : public CUDABinaryMetric<CUDABinaryLoglossMetric> {
 public:
  explicit CUDABinaryLoglossMetric(const Config& config);

  inline static double LossOnPoint(label_t label, double prob) {
    if (label <= 0) {
      if (1.0f - prob > kEpsilon) {
        return -std::log(1.0f - prob);
      }
    } else {
      if (prob > kEpsilon) {
        return -std::log(prob);
      }
    }
    return -std::log(kEpsilon);
  }

  __device__ inline static double LossOnPointCUDA(label_t label, double prob) {
    if (label <= 0) {
      if (1.0f - prob > kEpsilon) {
        return -log(1.0f - prob);
      }
    } else {
      if (prob > kEpsilon) {
        return -log(prob);
      }
    }
    return -log(kEpsilon);
  }

  inline static const char* Name() {
    return "binary_logloss";
  }
};

class CUDABinaryErrorMetric: public CUDABinaryMetric<CUDABinaryErrorMetric> {
 public:
  explicit CUDABinaryErrorMetric(const Config& config);

  inline static double LossOnPoint(label_t label, double prob) {
    if (prob <= 0.5f) {
      return label > 0;
    } else {
      return label <= 0;
    }
  }

  __device__ inline static double LossOnPointCUDA(label_t label, double prob) {
    if (prob <= 0.5f) {
      return label > 0;
    } else {
      return label <= 0;
    }
  }

  inline static const char* Name() {
    return "binary_error";
  }
};

class CUDAAUCMetric : public AUCMetric {
 public:
  CUDAAUCMetric(const Config& config);

  ~CUDAAUCMetric();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override;

 private:
  void LaunchEvalKernel(const double* score) const;

  void TestCUDABitonicSortForQueryItems() const;

  data_size_t* cuda_indices_buffer_;
  double* cuda_sum_pos_buffer_;
  double* cuda_block_sum_pos_buffer_;
  data_size_t* cuda_threshold_mark_;
  data_size_t* cuda_block_threshold_mark_buffer_;
  uint16_t* cuda_block_mark_first_zero_;
  const label_t* cuda_label_;
  const label_t* cuda_weights_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_
