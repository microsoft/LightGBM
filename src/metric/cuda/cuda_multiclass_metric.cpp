/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_multiclass_metric.hpp"

namespace LightGBM {

template <typename CUDAPointWiseLossCalculator>
CUDAMulticlassMetric<CUDAPointWiseLossCalculator>::CUDAMulticlassMetric(const Config& config): MulticlassMetric<CUDAPointWiseLossCalculator>(config) {}

template <typename CUDAPointWiseLossCalculator>
CUDAMulticlassMetric<CUDAPointWiseLossCalculator>::~CUDAMulticlassMetric() {}

template <typename CUDAPointWiseLossCalculator>
void CUDAMulticlassMetric<CUDAPointWiseLossCalculator>::Init(const Metadata& metadata, data_size_t num_data) {
  MulticlassMetric<CUDAPointWiseLossCalculator>::Init(metadata, num_data);
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();

  const data_size_t num_blocks = (num_data + EVAL_BLOCK_SIZE_MULTICLASS_METRIC - 1) / EVAL_BLOCK_SIZE_MULTICLASS_METRIC;
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_buffer_, static_cast<size_t>(num_blocks), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_score_convert_buffer_, static_cast<size_t>(num_data * this->num_class_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_sum_loss_, 1, __FILE__, __LINE__);
}

template <typename CUDAPointWiseLossCalculator>
std::vector<double> CUDAMulticlassMetric<CUDAPointWiseLossCalculator>::Eval(const double* score, const ObjectiveFunction* objective) const {
  double sum_loss = 0.0f;
  objective->GetCUDAConvertOutputFunc()(this->num_data_, score, cuda_score_convert_buffer_);
  LaunchEvalKernel(cuda_score_convert_buffer_);
  CopyFromCUDADeviceToHostOuter<double>(&sum_loss, cuda_sum_loss_, 1, __FILE__, __LINE__);
  return std::vector<double>(1, CUDAPointWiseLossCalculator::AverageLoss(sum_loss, this->sum_weights_));
}

CUDAMultiErrorMetric::CUDAMultiErrorMetric(const Config& config): CUDAMulticlassMetric<CUDAMultiErrorMetric>(config) {}

CUDAMultiSoftmaxLoglossMetric::CUDAMultiSoftmaxLoglossMetric(const Config& config): CUDAMulticlassMetric<CUDAMultiSoftmaxLoglossMetric>(config) {}

CUDAAucMuMetric::CUDAAucMuMetric(const Config& config): AucMuMetric(config) {}

CUDAAucMuMetric::~CUDAAucMuMetric() {}

void CUDAAucMuMetric::Init(const Metadata& metadata, data_size_t num_data) {
  AucMuMetric::Init(metadata, num_data);
  std::vector<data_size_t> class_start(num_class_, 0);
  data_size_t max_class_size = 0;
  int max_class_size_class = -1;
  for (int i = 0; i < num_class_; ++i) {
    const data_size_t this_class_size = class_sizes_[i];
    if (this_class_size > max_class_size) {
      max_class_size = this_class_size;
      max_class_size_class = i;
    }
  }
  data_size_t second_max_class_size = 0;
  for (int i = 0; i < num_class_; ++i) {
    if (i != max_class_size_class) {
      const data_size_t this_class_size = class_sizes_[i];
      if (this_class_size > second_max_class_size) {
        second_max_class_size = this_class_size;
      }
    }
  }
  for (int i = 1; i < num_class_; ++i) {
    class_start[i] += class_start[i - 1] + class_sizes_[i - 1];
  }
  InitCUDAMemoryFromHostMemoryOuter<data_size_t>(&cuda_class_start_, class_start.data(), class_start.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<data_size_t>(&cuda_class_size_, class_sizes_.data(), class_sizes_.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<data_size_t>(&cuda_sorted_indices_, sorted_data_idx_.data(), sorted_data_idx_.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_class_data_weights_, class_data_weights_.data(), class_data_weights_.size(), __FILE__, __LINE__);
  const int num_class_pair = (num_class_ - 1) * num_class_ / 2;
  max_pair_buffer_size_ = max_class_size + second_max_class_size;
  const size_t total_pair_buffer_size = static_cast<size_t>(max_pair_buffer_size_ * num_class_pair);
  AllocateCUDAMemoryOuter<double>(&cuda_dist_, total_pair_buffer_size, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_sorted_indices_by_dist_, total_pair_buffer_size, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_sum_pos_buffer_, total_pair_buffer_size, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_threshold_mark_, total_pair_buffer_size, __FILE__, __LINE__);

  const int num_blocks = (max_pair_buffer_size_ + EVAL_BLOCK_SIZE_MULTICLASS_METRIC - 1) / EVAL_BLOCK_SIZE_MULTICLASS_METRIC;
  const size_t class_pair_block_buffer = static_cast<size_t>(num_class_pair * (num_blocks + 1));
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_block_mark_buffer_, class_pair_block_buffer, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<uint16_t>(&cuda_block_mark_first_zero_, class_pair_block_buffer, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_reduce_block_buffer_, class_pair_block_buffer, __FILE__, __LINE__);
  SetCUDAMemoryOuter<double>(cuda_reduce_block_buffer_, 0, class_pair_block_buffer, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_reduce_ans_buffer_, static_cast<size_t>(num_class_pair), __FILE__, __LINE__);
  const size_t curr_v_size = static_cast<size_t>(num_class_pair * num_class_);
  std::vector<double> all_curr_v(curr_v_size, 0.0f);
  for (int i = 0; i < num_class_ - 1; ++i) {
    for (int j = i + 1; j < num_class_; ++j) {
      const int i_p = num_class_ - 2 - i;
      const int pair_index = i_p * (i_p + 1) / 2 + j - i - 1;
      for (int k = 0; k < num_class_; ++k) {
        all_curr_v[pair_index * num_class_ + k] = class_weights_[i][k] - class_weights_[j][k];
      }
    }
  }
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_curr_v_, all_curr_v.data(), all_curr_v.size(), __FILE__, __LINE__);

  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  cuda_label_ = metadata.cuda_metadata()->cuda_label();
}

std::vector<double> CUDAAucMuMetric::Eval(const double* score, const ObjectiveFunction*) const {
  LaunchEvalKernel(score);
  double ans = 0.0f;
  const int num_class_pair = (num_class_ - 1) * num_class_ / 2;
  CopyFromCUDADeviceToHostOuter<double>(&ans, cuda_reduce_ans_buffer_, static_cast<size_t>(num_class_pair), __FILE__, __LINE__);
  return std::vector<double>(1, ans / static_cast<double>(num_class_pair));
}

}  // namespace LightGBM
