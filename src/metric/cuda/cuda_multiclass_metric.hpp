/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_METRIC_CUDA_CUDA_MULTICLASS_METRIC_HPP_
#define LIGHTGBM_METRIC_CUDA_CUDA_MULTICLASS_METRIC_HPP_

#include "cuda_metric.hpp"
#include "../multiclass_metric.hpp"

#define EVAL_BLOCK_SIZE_MULTICLASS_METRIC (1024)

namespace LightGBM {

template <typename CUDAPointWiseLossCalculator>
class CUDAMulticlassMetric : public CUDAMetricInterface, public MulticlassMetric<CUDAPointWiseLossCalculator> {
 public:
  explicit CUDAMulticlassMetric(const Config& config);

  ~CUDAMulticlassMetric();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;

  inline static double AverageLoss(double sum_loss, double sum_weights) {
    // need sqrt the result for RMSE loss
    return (sum_loss / sum_weights);
  }

  inline static double LossOnPoint(label_t /*label*/, std::vector<double>* /*score*/, const Config& /*config*/) {
    Log::Fatal("Calling host LossOnPoint for a CUDA metric.");
    return 0.0f;
  }

 protected:
  void LaunchEvalKernel(const double* score) const;

  void LaunchEvalKernelInner(const double* score) const;

  const label_t* cuda_label_;
  const label_t* cuda_weights_;
  double* cuda_score_convert_buffer_;
  double* cuda_sum_loss_buffer_;
  double* cuda_sum_loss_;
};

class CUDAMultiErrorMetric : public CUDAMulticlassMetric<CUDAMultiErrorMetric> {
 public:
  explicit CUDAMultiErrorMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(
    label_t label,
    const double* score,
    const data_size_t data_index,
    const data_size_t num_data,
    const int num_classes,
    const int multi_error_top_k) {
    const size_t k = static_cast<size_t>(label);
    const double true_class_score = score[k * num_data + data_index];
    int num_larger = 0;
    for (int i = 0; i < num_classes; ++i) {
      const double this_class_score = score[i * num_data + data_index];
      if (this_class_score >= true_class_score) ++num_larger;
      if (num_larger > multi_error_top_k) return 1.0f;
    }
    return 0.0f;
  }

  inline static const std::string Name(const Config& config) {
    if (config.multi_error_top_k == 1) {
      return "multi_error";
    } else {
      return "multi_error@" + std::to_string(config.multi_error_top_k);
    }
  }
};

class CUDAMultiSoftmaxLoglossMetric : public CUDAMulticlassMetric<CUDAMultiSoftmaxLoglossMetric> {
 public:
  explicit CUDAMultiSoftmaxLoglossMetric(const Config& config);

  __device__ inline static double LossOnPointCUDA(label_t label, const double* score,
    const data_size_t data_index,
    const data_size_t num_data,
    const int /*num_classes*/, const int /*multi_error_top_k*/) {
    size_t k = static_cast<size_t>(label);
    const double point_score = score[k * num_data + data_index];
    if (point_score > kEpsilon) {
      return static_cast<double>(-log(point_score));
    } else {
      return -log(kEpsilon);
    }
  }

  inline static const std::string Name(const Config& /*config*/) {
    return "multi_logloss";
  }
};

class CUDAAucMuMetric : public CUDAMetricInterface, public AucMuMetric {
 public:
  explicit CUDAAucMuMetric(const Config& config);

  ~CUDAAucMuMetric();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override;

 private:
  void LaunchEvalKernel(const double* score) const;

  const label_t* cuda_label_;
  const label_t* cuda_weights_;

  int num_class_pair_;
  data_size_t max_pair_buffer_size_;

  data_size_t* cuda_class_start_;
  data_size_t* cuda_class_size_;
  data_size_t* cuda_sorted_indices_;
  double* cuda_dist_;
  double* cuda_class_data_weights_;
  double* cuda_class_weights_;
  data_size_t* cuda_sorted_indices_by_dist_;
  double* cuda_curr_v_;

  double* cuda_sum_pos_buffer_;
  data_size_t* cuda_threshold_mark_;
  data_size_t* cuda_block_mark_buffer_;
  uint16_t* cuda_block_mark_first_zero_;

  double* cuda_reduce_block_buffer_;
  double* cuda_reduce_ans_buffer_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_METRIC_CUDA_CUDA_MULTICLASS_METRIC_HPP_
