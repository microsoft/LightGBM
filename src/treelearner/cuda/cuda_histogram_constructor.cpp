/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

namespace LightGBM {

CUDAHistogramConstructor::CUDAHistogramConstructor(const Dataset* train_data,
  const int num_leaves, const int num_threads,
  const score_t* cuda_gradients, const score_t* cuda_hessians,
  const std::vector<uint32_t>& feature_hist_offsets,
  const int min_data_in_leaf, const double min_sum_hessian_in_leaf): num_data_(train_data->num_data()),
  num_features_(train_data->num_features()), num_leaves_(num_leaves), num_threads_(num_threads),
  num_feature_groups_(train_data->num_feature_groups()),
  cuda_gradients_(cuda_gradients), cuda_hessians_(cuda_hessians),
  min_data_in_leaf_(min_data_in_leaf), min_sum_hessian_in_leaf_(min_sum_hessian_in_leaf) {
  int offset = 0;
  for (int group_id = 0; group_id < train_data->num_feature_groups(); ++group_id) {
    feature_group_bin_offsets_.emplace_back(offset);
    offset += train_data->FeatureGroupNumBin(group_id);
  }
  for (int feature_index = 0; feature_index < train_data->num_features(); ++feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(feature_index);
    const uint32_t most_freq_bin = bin_mapper->GetMostFreqBin();
    if (most_freq_bin == 0) {
      feature_mfb_offsets_.emplace_back(1);
    } else {
      feature_mfb_offsets_.emplace_back(0);
    }
    feature_num_bins_.emplace_back(static_cast<uint32_t>(bin_mapper->num_bin()));
    feature_most_freq_bins_.emplace_back(most_freq_bin);
  }
  feature_group_bin_offsets_.emplace_back(offset);
  feature_hist_offsets_.clear();
  for (size_t i = 0; i < feature_hist_offsets.size(); ++i) {
    feature_hist_offsets_.emplace_back(feature_hist_offsets[i]);
  }
  num_total_bin_ = offset;
}

void CUDAHistogramConstructor::BeforeTrain() {
  SetCUDAMemory<hist_t>(cuda_hist_, 0, num_total_bin_ * 2 * num_leaves_);
}

void CUDAHistogramConstructor::Init(const Dataset* train_data, TrainingShareStates* share_state) {
  AllocateCUDAMemory<hist_t>(num_total_bin_ * 2 * num_leaves_, &cuda_hist_);
  SetCUDAMemory<hist_t>(cuda_hist_, 0, num_total_bin_ * 2 * num_leaves_);

  AllocateCUDAMemory<score_t>(num_data_, &cuda_ordered_gradients_);
  AllocateCUDAMemory<score_t>(num_data_, &cuda_ordered_hessians_);

  InitCUDAMemoryFromHostMemory<int>(&cuda_num_total_bin_, &num_total_bin_, 1);

  InitCUDAMemoryFromHostMemory<int>(&cuda_num_feature_groups_, &num_feature_groups_, 1);

  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_group_bin_offsets_,
    feature_group_bin_offsets_.data(), feature_group_bin_offsets_.size());

  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_mfb_offsets_,
    feature_mfb_offsets_.data(), feature_mfb_offsets_.size());

  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_num_bins_,
    feature_num_bins_.data(), feature_num_bins_.size());

  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_hist_offsets_,
    feature_hist_offsets_.data(), feature_hist_offsets_.size());

  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_most_freq_bins_,
    feature_most_freq_bins_.data(), feature_most_freq_bins_.size());

  InitCUDAValueFromConstant<int>(&cuda_num_features_, num_features_);

  DivideCUDAFeatureGroups(train_data, share_state);

  InitCUDAData(train_data, share_state);
}

void CUDAHistogramConstructor::InitCUDAData(const Dataset* train_data, TrainingShareStates* share_state) {
  uint8_t bit_type = 0;
  size_t total_size = 0;
  const uint8_t* cpu_data_ptr = share_state->GetRowWiseData(&bit_type, &total_size, &is_sparse_);
  CHECK_EQ(bit_type, 8);
  std::vector<uint8_t> partitioned_data;
  GetDenseDataPartitioned<uint8_t>(cpu_data_ptr, &partitioned_data);
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_data_uint8_t_, partitioned_data.data(), total_size);
  SynchronizeCUDADevice();
}

void CUDAHistogramConstructor::PushOneData(const uint32_t feature_bin_value,
  const int feature_group_id,
  const data_size_t data_index) {
  const uint8_t feature_bin_value_uint8 = static_cast<uint8_t>(feature_bin_value);
  const size_t index = static_cast<size_t>(data_index) * static_cast<size_t>(num_feature_groups_) +
    static_cast<size_t>(feature_group_id);
  data_[index] = feature_bin_value_uint8;
}

void CUDAHistogramConstructor::ConstructHistogramForLeaf(const int* cuda_smaller_leaf_index, const data_size_t* cuda_num_data_in_smaller_leaf,
  const int* cuda_larger_leaf_index, const data_size_t** cuda_data_indices_in_smaller_leaf, const data_size_t** cuda_data_indices_in_larger_leaf,
  const double* cuda_smaller_leaf_sum_gradients, const double* cuda_smaller_leaf_sum_hessians, hist_t** cuda_smaller_leaf_hist,
  const double* cuda_larger_leaf_sum_gradients, const double* cuda_larger_leaf_sum_hessians, hist_t** cuda_larger_leaf_hist,
  const data_size_t* cuda_leaf_num_data, const data_size_t num_data_in_smaller_leaf, const data_size_t num_data_in_larger_leaf,
  const double sum_hessians_in_smaller_leaf, const double sum_hessians_in_larger_leaf) {
  //auto start = std::chrono::steady_clock::now();
  if ((num_data_in_smaller_leaf <= min_data_in_leaf_ || sum_hessians_in_smaller_leaf <= min_sum_hessian_in_leaf_) &&
    (num_data_in_larger_leaf <= min_data_in_leaf_ || sum_hessians_in_larger_leaf <= min_sum_hessian_in_leaf_)) {
    return;
  }
  LaunchConstructHistogramKernel(cuda_smaller_leaf_index, cuda_num_data_in_smaller_leaf,
    cuda_data_indices_in_smaller_leaf, cuda_leaf_num_data, cuda_smaller_leaf_hist, num_data_in_smaller_leaf);
  SynchronizeCUDADevice();
  //auto end = std::chrono::steady_clock::now();
  //double duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  //Log::Warning("LaunchConstructHistogramKernel time %f", duration);
  global_timer.Start("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
  //start = std::chrono::steady_clock::now();
  LaunchSubtractHistogramKernel(cuda_smaller_leaf_index,
    cuda_larger_leaf_index, cuda_smaller_leaf_sum_gradients, cuda_smaller_leaf_sum_hessians,
    cuda_larger_leaf_sum_gradients, cuda_larger_leaf_sum_hessians, cuda_smaller_leaf_hist, cuda_larger_leaf_hist);
  //end = std::chrono::steady_clock::now();
  //duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  global_timer.Stop("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
  //Log::Warning("LaunchSubtractHistogramKernel time %f", duration);
  /*PrintLastCUDAError();
  std::vector<hist_t> cpu_hist(6143 * 2, 0.0f);
  CopyFromCUDADeviceToHost<hist_t>(cpu_hist.data(), cuda_hist_, 6143 * 2);*/
}

void CUDAHistogramConstructor::CalcConstructHistogramKernelDim(
  int* grid_dim_x, int* grid_dim_y, int* block_dim_x, int* block_dim_y,
  const data_size_t num_data_in_smaller_leaf) {
  *block_dim_x = max_num_column_per_partition_;
  *block_dim_y = NUM_THRADS_PER_BLOCK / max_num_column_per_partition_;
  *grid_dim_x = num_feature_partitions_;
  const int min_grid_dim_y = 160;
  *grid_dim_y = std::max(min_grid_dim_y,
    ((num_data_in_smaller_leaf + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + (*block_dim_y) - 1) / (*block_dim_y));
}

void CUDAHistogramConstructor::DivideCUDAFeatureGroups(const Dataset* train_data, TrainingShareStates* share_state) {
  const uint32_t max_num_bin_per_partition = SHRAE_HIST_SIZE / 2;
  const std::vector<uint32_t>& column_hist_offsets = share_state->column_hist_offsets();
  std::vector<int> feature_group_num_feature_offsets;
  int offsets = 0;
  int prev_group_index = -1;
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    const int feature_group_index = train_data->Feature2Group(feature_index);
    if (prev_group_index == -1 || feature_group_index != prev_group_index) {
      feature_group_num_feature_offsets.emplace_back(offsets);
    }
    ++offsets;
  }
  CHECK_EQ(offsets, num_features_);
  feature_group_num_feature_offsets.emplace_back(offsets);

  uint32_t start_hist_offset = 0;
  feature_partition_column_index_offsets_.clear();
  column_hist_offsets_.clear();
  column_hist_offsets_full_.clear();
  feature_partition_column_index_offsets_.emplace_back(0);
  column_hist_offsets_full_.emplace_back(0);
  const int num_feature_groups = train_data->num_feature_groups();
  int column_index = 0;
  num_feature_partitions_ = 0;
  for (int feature_group_index = 0; feature_group_index < num_feature_groups; ++feature_group_index) {
    if (!train_data->IsMultiGroup(feature_group_index)) {
      const uint32_t column_feature_hist_start = column_hist_offsets[column_index];
      const uint32_t column_feature_hist_end = column_hist_offsets[column_index + 1];
      const uint32_t num_bin_in_dense_group = column_feature_hist_end - column_feature_hist_start;
      if (num_bin_in_dense_group > max_num_bin_per_partition) {
        Log::Fatal("Too many bins in a dense feature group.");
      }
      const uint32_t cur_hist_num_bin = column_feature_hist_end - start_hist_offset;
      if (cur_hist_num_bin > max_num_bin_per_partition) {
        feature_partition_column_index_offsets_.emplace_back(column_index);
        start_hist_offset = column_feature_hist_start;
        column_hist_offsets_full_.emplace_back(start_hist_offset);
        ++num_feature_partitions_;
      }
      column_hist_offsets_.emplace_back(column_hist_offsets[column_index] - start_hist_offset);
      if (feature_group_index == num_feature_groups - 1) {
        feature_partition_column_index_offsets_.emplace_back(column_index + 1);
        column_hist_offsets_full_.emplace_back(column_hist_offsets.back());
        ++num_feature_partitions_;
      }
      ++column_index;
    } else {
      const int group_feature_index_start = feature_group_num_feature_offsets[feature_group_index];
      const int num_features_in_group = feature_group_num_feature_offsets[feature_group_index + 1] - group_feature_index_start;
      for (int sub_feature_index = 0; sub_feature_index < num_features_in_group; ++sub_feature_index) {
        const int feature_index = group_feature_index_start + sub_feature_index;
        const uint32_t column_feature_hist_start = column_hist_offsets[column_index];
        const uint32_t column_feature_hist_end = column_hist_offsets[column_index + 1];
        const uint32_t cur_hist_num_bin = column_feature_hist_end - start_hist_offset;
        if (cur_hist_num_bin > max_num_bin_per_partition) {
          feature_partition_column_index_offsets_.emplace_back(column_index);
          start_hist_offset = column_feature_hist_start;
          column_hist_offsets_full_.emplace_back(start_hist_offset);
          ++num_feature_partitions_;
        }
        column_hist_offsets_.emplace_back(column_hist_offsets[column_index] - start_hist_offset);
        if (feature_group_index == num_feature_groups - 1 && sub_feature_index == num_features_in_group - 1) {
          CHECK_EQ(feature_index, num_features_ - 1);
          feature_partition_column_index_offsets_.emplace_back(column_index + 1);
          column_hist_offsets_full_.emplace_back(column_hist_offsets.back());
          ++num_feature_partitions_;
        }
        ++column_index;
      }
    }
  }
  max_num_column_per_partition_ = 0;
  for (size_t i = 0; i < feature_partition_column_index_offsets_.size() - 1; ++i) {
    const int num_column = feature_partition_column_index_offsets_[i + 1] - feature_partition_column_index_offsets_[i];
    if (num_column > max_num_column_per_partition_) {
      max_num_column_per_partition_ = num_column;
    }
  }

  for (size_t i = 0; i < feature_partition_column_index_offsets_.size(); ++i) {
    Log::Warning("feature_partition_column_index_offsets_[%d] = %d", i, feature_partition_column_index_offsets_[i]);
  }

  for (size_t i = 0; i < column_hist_offsets_.size(); ++i) {
    Log::Warning("column_hist_offsets_[%d] = %d", i, column_hist_offsets_[i]);
  }

  for (size_t i = 0; i < column_hist_offsets_full_.size(); ++i) {
    Log::Warning("column_hist_offsets_full_[%d] = %d", i, column_hist_offsets_full_[i]);
  }

  InitCUDAMemoryFromHostMemory<int>(&cuda_feature_partition_column_index_offsets_,
    feature_partition_column_index_offsets_.data(),
    feature_partition_column_index_offsets_.size());

  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_column_hist_offsets_,
    column_hist_offsets_.data(),
    column_hist_offsets_.size());

  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_column_hist_offsets_full_,
    column_hist_offsets_full_.data(),
    column_hist_offsets_full_.size());
}

template <typename BIN_TYPE>
void CUDAHistogramConstructor::GetDenseDataPartitioned(const BIN_TYPE* row_wise_data, std::vector<BIN_TYPE>* partitioned_data) {
  Log::Warning("feature_partition_column_index_offsets_.size() = %d", feature_partition_column_index_offsets_.size());
  const int num_total_columns = feature_partition_column_index_offsets_.back();
  partitioned_data->resize(static_cast<size_t>(num_total_columns) * static_cast<size_t>(num_data_), 0);
  BIN_TYPE* out_data = partitioned_data->data();
  Threading::For<data_size_t>(0, num_data_, 512,
    [this, num_total_columns, row_wise_data, out_data] (int thread_index, data_size_t start, data_size_t end) {
      for (size_t i = 0; i < feature_partition_column_index_offsets_.size() - 1; ++i) {
        const int num_prev_columns = static_cast<int>(feature_partition_column_index_offsets_[i]);
        const data_size_t offset = num_data_ * num_prev_columns;
        const int partition_column_start = feature_partition_column_index_offsets_[i];
        const int partition_column_end = feature_partition_column_index_offsets_[i + 1];
        const int num_columns_in_cur_partition = partition_column_end - partition_column_start;
        for (data_size_t data_index = start; data_index < end; ++data_index) {
          const data_size_t data_offset = offset + data_index * num_columns_in_cur_partition;
          const data_size_t read_data_offset = data_index * num_total_columns;
          for (int column_index = 0; column_index < num_columns_in_cur_partition; ++column_index) {
            const int true_column_index = read_data_offset + column_index + partition_column_start;
            const BIN_TYPE bin = row_wise_data[true_column_index];
            out_data[data_offset + column_index] = bin;
          }
        }
      }
    });
}

}  // namespace LightGBM

#endif  // USE_CUDA
