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
  min_data_in_leaf_(min_data_in_leaf), min_sum_hessian_in_leaf_(min_sum_hessian_in_leaf),
  cuda_gradients_(cuda_gradients), cuda_hessians_(cuda_hessians) {
  train_data_ = train_data;
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

  InitCUDAData(share_state);
}

void CUDAHistogramConstructor::InitCUDAData(TrainingShareStates* share_state) {
  bit_type_ = 0;
  size_t total_size = 0;
  const uint8_t* data_ptr = nullptr;
  data_ptr_bit_type_ = 0;
  const uint8_t* cpu_data_ptr = share_state->GetRowWiseData(&bit_type_, &total_size, &is_sparse_, &data_ptr, &data_ptr_bit_type_);
  Log::Warning("bit_type_ = %d, is_sparse_ = %d, data_ptr_bit_type_ = %d", bit_type_, static_cast<int>(is_sparse_), data_ptr_bit_type_);
  if (bit_type_ == 8) {
    if (!is_sparse_) {
      std::vector<uint8_t> partitioned_data;
      GetDenseDataPartitioned<uint8_t>(cpu_data_ptr, &partitioned_data);
      InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_data_uint8_t_, partitioned_data.data(), total_size);
    } else {
      std::vector<std::vector<uint8_t>> partitioned_data;
      if (data_ptr_bit_type_ == 16) {
        std::vector<std::vector<uint16_t>> partitioned_data_ptr;
        std::vector<uint16_t> partition_ptr;
        const uint16_t* data_ptr_uint16_t = reinterpret_cast<const uint16_t*>(data_ptr);
        GetSparseDataPartitioned<uint8_t, uint16_t>(cpu_data_ptr, data_ptr_uint16_t, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
        InitCUDAMemoryFromHostMemory<uint16_t>(&cuda_partition_ptr_uint16_t_, partition_ptr.data(), partition_ptr.size());
        AllocateCUDAMemory<uint8_t>(partition_ptr.back(), &cuda_data_uint8_t_);
        AllocateCUDAMemory<uint16_t>((num_data_ + 1) * partitioned_data_ptr.size(), &cuda_row_ptr_uint16_t_);
        for (size_t i = 0; i < partitioned_data.size(); ++i) {
          const std::vector<uint16_t>& data_ptr_for_this_partition = partitioned_data_ptr[i];
          const std::vector<uint8_t>& data_for_this_partition = partitioned_data[i];
          CopyFromHostToCUDADevice<uint8_t>(cuda_data_uint8_t_ + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size());
          CopyFromHostToCUDADevice<uint16_t>(cuda_row_ptr_uint16_t_ + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size());
        }
      } else if (data_ptr_bit_type_ == 32) {
        const uint32_t* data_ptr_uint32_t = reinterpret_cast<const uint32_t*>(data_ptr);
        std::vector<std::vector<uint32_t>> partitioned_data_ptr;
        std::vector<uint32_t> partition_ptr;
        GetSparseDataPartitioned<uint8_t, uint32_t>(cpu_data_ptr, data_ptr_uint32_t, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
        InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_partition_ptr_uint32_t_, partition_ptr.data(), partition_ptr.size());
        AllocateCUDAMemory<uint8_t>(partition_ptr.back(), &cuda_data_uint8_t_);
        AllocateCUDAMemory<uint32_t>((num_data_ + 1) * partitioned_data_ptr.size(), &cuda_row_ptr_uint32_t_); 
        for (size_t i = 0; i < partitioned_data.size(); ++i) {
          const std::vector<uint32_t>& data_ptr_for_this_partition = partitioned_data_ptr[i];
          const std::vector<uint8_t>& data_for_this_partition = partitioned_data[i];
          CopyFromHostToCUDADevice<uint8_t>(cuda_data_uint8_t_ + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size());
          CopyFromHostToCUDADevice<uint32_t>(cuda_row_ptr_uint32_t_ + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size());
        }
      } else if (data_ptr_bit_type_ == 64) {
        const uint64_t* data_ptr_uint64_t = reinterpret_cast<const uint64_t*>(data_ptr);
        std::vector<std::vector<uint64_t>> partitioned_data_ptr;
        std::vector<uint64_t> partition_ptr;
        GetSparseDataPartitioned<uint8_t, uint64_t>(cpu_data_ptr, data_ptr_uint64_t, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
        InitCUDAMemoryFromHostMemory<uint64_t>(&cuda_partition_ptr_uint64_t_, partition_ptr.data(), partition_ptr.size());
        AllocateCUDAMemory<uint8_t>(partition_ptr.back(), &cuda_data_uint8_t_);
        AllocateCUDAMemory<uint64_t>((num_data_ + 1) * partitioned_data_ptr.size(), &cuda_row_ptr_uint64_t_); 
        for (size_t i = 0; i < partitioned_data.size(); ++i) {
          const std::vector<uint64_t>& data_ptr_for_this_partition = partitioned_data_ptr[i];
          const std::vector<uint8_t>& data_for_this_partition = partitioned_data[i];
          CopyFromHostToCUDADevice<uint8_t>(cuda_data_uint8_t_ + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size());
          CopyFromHostToCUDADevice<uint64_t>(cuda_row_ptr_uint64_t_ + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size());
        }
      } else {
        Log::Fatal("Unknow data ptr bit type %d", data_ptr_bit_type_);
      }
    }
  } else if (bit_type_ == 16) {
    if (!is_sparse_) {
      std::vector<uint16_t> partitioned_data;
      GetDenseDataPartitioned<uint16_t>(reinterpret_cast<const uint16_t*>(cpu_data_ptr), &partitioned_data);
      InitCUDAMemoryFromHostMemory<uint16_t>(&cuda_data_uint16_t_, partitioned_data.data(), total_size);
    } else {
      std::vector<std::vector<uint16_t>> partitioned_data;
      if (data_ptr_bit_type_ == 16) {
        std::vector<std::vector<uint16_t>> partitioned_data_ptr;
        std::vector<uint16_t> partition_ptr;
        const uint16_t* data_ptr_uint16_t = reinterpret_cast<const uint16_t*>(data_ptr);
        GetSparseDataPartitioned<uint16_t, uint16_t>(reinterpret_cast<const uint16_t*>(cpu_data_ptr), data_ptr_uint16_t, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
        InitCUDAMemoryFromHostMemory<uint16_t>(&cuda_partition_ptr_uint16_t_, partition_ptr.data(), partition_ptr.size());
        AllocateCUDAMemory<uint16_t>(partition_ptr.back(), &cuda_data_uint16_t_);
        AllocateCUDAMemory<uint16_t>((num_data_ + 1) * partitioned_data_ptr.size(), &cuda_row_ptr_uint16_t_); 
        for (size_t i = 0; i < partitioned_data.size(); ++i) {
          const std::vector<uint16_t>& data_ptr_for_this_partition = partitioned_data_ptr[i];
          const std::vector<uint16_t>& data_for_this_partition = partitioned_data[i];
          CopyFromHostToCUDADevice<uint16_t>(cuda_data_uint16_t_ + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size());
          CopyFromHostToCUDADevice<uint16_t>(cuda_row_ptr_uint16_t_ + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size());
        }
      } else if (data_ptr_bit_type_ == 32) {
        std::vector<std::vector<uint32_t>> partitioned_data_ptr;
        std::vector<uint32_t> partition_ptr;
        const uint32_t* data_ptr_uint32_t = reinterpret_cast<const uint32_t*>(data_ptr);
        GetSparseDataPartitioned<uint16_t, uint32_t>(reinterpret_cast<const uint16_t*>(cpu_data_ptr), data_ptr_uint32_t, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
        InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_partition_ptr_uint32_t_, partition_ptr.data(), partition_ptr.size());
        AllocateCUDAMemory<uint16_t>(partition_ptr.back(), &cuda_data_uint16_t_);
        AllocateCUDAMemory<uint32_t>((num_data_ + 1) * partitioned_data_ptr.size(), &cuda_row_ptr_uint32_t_);
        for (size_t i = 0; i < partitioned_data.size(); ++i) {
          const std::vector<uint32_t>& data_ptr_for_this_partition = partitioned_data_ptr[i];
          const std::vector<uint16_t>& data_for_this_partition = partitioned_data[i];
          CopyFromHostToCUDADevice<uint16_t>(cuda_data_uint16_t_ + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size());
          CopyFromHostToCUDADevice<uint32_t>(cuda_row_ptr_uint32_t_ + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size());
        }
      } else if (data_ptr_bit_type_ == 64) {
        std::vector<std::vector<uint64_t>> partitioned_data_ptr;
        std::vector<uint64_t> partition_ptr;
        const uint64_t* data_ptr_uint64_t = reinterpret_cast<const uint64_t*>(data_ptr);
        GetSparseDataPartitioned<uint16_t, uint64_t>(reinterpret_cast<const uint16_t*>(cpu_data_ptr), data_ptr_uint64_t, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
        InitCUDAMemoryFromHostMemory<uint64_t>(&cuda_partition_ptr_uint64_t_, partition_ptr.data(), partition_ptr.size());
        AllocateCUDAMemory<uint16_t>(partition_ptr.back(), &cuda_data_uint16_t_);
        AllocateCUDAMemory<uint64_t>((num_data_ + 1) * partitioned_data_ptr.size(), &cuda_row_ptr_uint64_t_); 
        for (size_t i = 0; i < partitioned_data.size(); ++i) {
          const std::vector<uint64_t>& data_ptr_for_this_partition = partitioned_data_ptr[i];
          const std::vector<uint16_t>& data_for_this_partition = partitioned_data[i];
          CopyFromHostToCUDADevice<uint16_t>(cuda_data_uint16_t_ + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size());
          CopyFromHostToCUDADevice<uint64_t>(cuda_row_ptr_uint64_t_ + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size());
        }
      } else {
        Log::Fatal("Unknow data ptr bit type %d", data_ptr_bit_type_);
      }
    }
  } else if (bit_type_ == 32) {
    if (!is_sparse_) {
      std::vector<uint32_t> partitioned_data;
      GetDenseDataPartitioned<uint32_t>(reinterpret_cast<const uint32_t*>(cpu_data_ptr), &partitioned_data);
      InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_data_uint32_t_, partitioned_data.data(), total_size);
    } else {
      std::vector<std::vector<uint32_t>> partitioned_data;
      if (data_ptr_bit_type_ == 16) {
        const uint16_t* data_ptr_uint16_t = reinterpret_cast<const uint16_t*>(data_ptr);
        std::vector<std::vector<uint16_t>> partitioned_data_ptr;
        std::vector<uint16_t> partition_ptr;
        GetSparseDataPartitioned<uint32_t, uint16_t>(reinterpret_cast<const uint32_t*>(cpu_data_ptr), data_ptr_uint16_t, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
        InitCUDAMemoryFromHostMemory<uint16_t>(&cuda_partition_ptr_uint16_t_, partition_ptr.data(), partition_ptr.size());
        AllocateCUDAMemory<uint32_t>(partition_ptr.back(), &cuda_data_uint32_t_);
        AllocateCUDAMemory<uint16_t>((num_data_ + 1) * partitioned_data_ptr.size(), &cuda_row_ptr_uint16_t_); 
        for (size_t i = 0; i < partitioned_data.size(); ++i) {
          const std::vector<uint16_t>& data_ptr_for_this_partition = partitioned_data_ptr[i];
          const std::vector<uint32_t>& data_for_this_partition = partitioned_data[i];
          CopyFromHostToCUDADevice<uint32_t>(cuda_data_uint32_t_ + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size());
          CopyFromHostToCUDADevice<uint16_t>(cuda_row_ptr_uint16_t_ + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size());
        }
      } else if (data_ptr_bit_type_ == 32) {
        const uint32_t* data_ptr_uint32_t = reinterpret_cast<const uint32_t*>(data_ptr);
        std::vector<std::vector<uint32_t>> partitioned_data_ptr;
        std::vector<uint32_t> partition_ptr;
        GetSparseDataPartitioned<uint32_t, uint32_t>(reinterpret_cast<const uint32_t*>(cpu_data_ptr), data_ptr_uint32_t, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
        InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_partition_ptr_uint32_t_, partition_ptr.data(), partition_ptr.size());
        AllocateCUDAMemory<uint32_t>(partition_ptr.back(), &cuda_data_uint32_t_);
        AllocateCUDAMemory<uint32_t>((num_data_ + 1) * partitioned_data_ptr.size(), &cuda_row_ptr_uint32_t_); 
        for (size_t i = 0; i < partitioned_data.size(); ++i) {
          const std::vector<uint32_t>& data_ptr_for_this_partition = partitioned_data_ptr[i];
          const std::vector<uint32_t>& data_for_this_partition = partitioned_data[i];
          CopyFromHostToCUDADevice<uint32_t>(cuda_data_uint32_t_ + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size());
          CopyFromHostToCUDADevice<uint32_t>(cuda_row_ptr_uint32_t_ + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size());
        }
      } else if (data_ptr_bit_type_ == 64) {
        const uint64_t* data_ptr_uint64_t = reinterpret_cast<const uint64_t*>(data_ptr);
        std::vector<std::vector<uint64_t>> partitioned_data_ptr;
        std::vector<uint64_t> partition_ptr;
        GetSparseDataPartitioned<uint32_t, uint64_t>(reinterpret_cast<const uint32_t*>(cpu_data_ptr), data_ptr_uint64_t, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
        InitCUDAMemoryFromHostMemory<uint64_t>(&cuda_partition_ptr_uint64_t_, partition_ptr.data(), partition_ptr.size());
        AllocateCUDAMemory<uint32_t>(partition_ptr.back(), &cuda_data_uint32_t_);
        AllocateCUDAMemory<uint64_t>((num_data_ + 1) * partitioned_data_ptr.size(), &cuda_row_ptr_uint64_t_); 
        for (size_t i = 0; i < partitioned_data.size(); ++i) {
          const std::vector<uint64_t>& data_ptr_for_this_partition = partitioned_data_ptr[i];
          const std::vector<uint32_t>& data_for_this_partition = partitioned_data[i];
          CopyFromHostToCUDADevice<uint32_t>(cuda_data_uint32_t_ + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size());
          CopyFromHostToCUDADevice<uint64_t>(cuda_row_ptr_uint64_t_ + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size());
        }
      } else {
        Log::Fatal("Unknow data ptr bit type %d", data_ptr_bit_type_);
      }
    }
  } else {
    Log::Fatal("Unknow bit type = %d", bit_type_);
  }
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
  const int* cuda_larger_leaf_index, const data_size_t** cuda_data_indices_in_smaller_leaf, const data_size_t** /*cuda_data_indices_in_larger_leaf*/,
  const double* cuda_smaller_leaf_sum_gradients, const double* cuda_smaller_leaf_sum_hessians, hist_t** cuda_smaller_leaf_hist,
  const double* cuda_larger_leaf_sum_gradients, const double* cuda_larger_leaf_sum_hessians, hist_t** cuda_larger_leaf_hist,
  const data_size_t* cuda_leaf_num_data, const data_size_t num_data_in_smaller_leaf, const data_size_t num_data_in_larger_leaf,
  const double sum_hessians_in_smaller_leaf, const double sum_hessians_in_larger_leaf) {
  if ((num_data_in_smaller_leaf <= min_data_in_leaf_ || sum_hessians_in_smaller_leaf <= min_sum_hessian_in_leaf_) &&
    (num_data_in_larger_leaf <= min_data_in_leaf_ || sum_hessians_in_larger_leaf <= min_sum_hessian_in_leaf_)) {
    return;
  }
  LaunchConstructHistogramKernel(cuda_smaller_leaf_index, cuda_num_data_in_smaller_leaf,
    cuda_data_indices_in_smaller_leaf, cuda_leaf_num_data, cuda_smaller_leaf_hist, num_data_in_smaller_leaf);
  SynchronizeCUDADevice();
  /*std::vector<hist_t> root_hist(20000);
  CopyFromCUDADeviceToHost<hist_t>(root_hist.data(), cuda_hist_, 20000);
  for (int real_feature_index = 0; real_feature_index < train_data_->num_total_features(); ++real_feature_index) {
    const int inner_feature_index = train_data_->InnerFeatureIndex(real_feature_index);
    if (inner_feature_index >= 0) {
      const uint32_t feature_hist_start = feature_hist_offsets_[inner_feature_index];
      const uint32_t feature_hist_end = feature_hist_offsets_[inner_feature_index + 1];
      Log::Warning("real_feature_index = %d, inner_feature_index = %d", real_feature_index, inner_feature_index);
      for (uint32_t hist_position = feature_hist_start; hist_position < feature_hist_end; ++hist_position) {
        Log::Warning("hist_position = %d, bin_in_feature = %d, grad = %f, hess = %f",
          hist_position, hist_position - feature_hist_start, root_hist[hist_position * 2], root_hist[hist_position * 2 + 1]);
      }
    }
  }*/
  global_timer.Start("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
  LaunchSubtractHistogramKernel(cuda_smaller_leaf_index,
    cuda_larger_leaf_index, cuda_smaller_leaf_sum_gradients, cuda_smaller_leaf_sum_hessians,
    cuda_larger_leaf_sum_gradients, cuda_larger_leaf_sum_hessians, cuda_smaller_leaf_hist, cuda_larger_leaf_hist);
  global_timer.Stop("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
}

void CUDAHistogramConstructor::CalcConstructHistogramKernelDim(
  int* grid_dim_x, int* grid_dim_y, int* block_dim_x, int* block_dim_y,
  const data_size_t num_data_in_smaller_leaf) {
  *block_dim_x = max_num_column_per_partition_;
  *block_dim_y = NUM_THRADS_PER_BLOCK / max_num_column_per_partition_;
  *grid_dim_x = num_feature_partitions_;
  const int min_grid_dim_y = 10;
  *grid_dim_y = std::max(min_grid_dim_y,
    ((num_data_in_smaller_leaf + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + (*block_dim_y) - 1) / (*block_dim_y));
  //Log::Warning("block_dim_x = %d, block_dim_y = %d, grid_dim_x = %d, grid_dim_y = %d", *block_dim_x, *block_dim_y, *grid_dim_x, *grid_dim_y);
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

  /*Log::Warning("max_num_column_per_partition_ = %d", max_num_column_per_partition_);
  for (size_t i = 0; i < feature_partition_column_index_offsets_.size(); ++i) {
    Log::Warning("feature_partition_column_index_offsets_[%d] = %d", i, feature_partition_column_index_offsets_[i]);
  }
  for (size_t i = 0; i < column_hist_offsets_full_.size(); ++i) {
    Log::Warning("column_hist_offsets_full_[%d] = %d", i, column_hist_offsets_full_[i]);
  }
  for (size_t i = 0; i < column_hist_offsets_.size(); ++i) {
    Log::Warning("column_hist_offsets_[%d] = %d", i, column_hist_offsets_[i]);
  }*/

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
  const int num_total_columns = feature_partition_column_index_offsets_.back();
  partitioned_data->resize(static_cast<size_t>(num_total_columns) * static_cast<size_t>(num_data_), 0);
  BIN_TYPE* out_data = partitioned_data->data();
  Threading::For<data_size_t>(0, num_data_, 512,
    [this, num_total_columns, row_wise_data, out_data] (int /*thread_index*/, data_size_t start, data_size_t end) {
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

template <typename BIN_TYPE, typename DATA_PTR_TYPE>
void CUDAHistogramConstructor::GetSparseDataPartitioned(
  const BIN_TYPE* row_wise_data,
  const DATA_PTR_TYPE* row_ptr,
  std::vector<std::vector<BIN_TYPE>>* partitioned_data,
  std::vector<std::vector<DATA_PTR_TYPE>>* partitioned_row_ptr,
  std::vector<DATA_PTR_TYPE>* partition_ptr) {
  const int num_partitions = static_cast<int>(feature_partition_column_index_offsets_.size()) - 1;
  partitioned_data->resize(num_partitions);
  partitioned_row_ptr->resize(num_partitions);
  std::vector<int> thread_max_elements_per_row(num_threads_, 0);
  Threading::For<int>(0, num_partitions, 1,
    [partitioned_data, partitioned_row_ptr, row_ptr, row_wise_data, &thread_max_elements_per_row, this] (int thread_index, int start, int end) {
      for (int partition_index = start; partition_index < end; ++partition_index) {
        std::vector<BIN_TYPE>& data_for_this_partition = partitioned_data->at(partition_index);
        std::vector<DATA_PTR_TYPE>& row_ptr_for_this_partition = partitioned_row_ptr->at(partition_index);
        const int partition_hist_start = column_hist_offsets_full_[partition_index];
        const int partition_hist_end = column_hist_offsets_full_[partition_index + 1];
        DATA_PTR_TYPE offset = 0;
        row_ptr_for_this_partition.clear();
        data_for_this_partition.clear();
        row_ptr_for_this_partition.emplace_back(offset);
        for (data_size_t data_index = 0; data_index < num_data_; ++data_index) {
          const DATA_PTR_TYPE row_start = row_ptr[data_index];
          const DATA_PTR_TYPE row_end = row_ptr[data_index + 1];
          const BIN_TYPE* row_data_start = row_wise_data + row_start;
          const BIN_TYPE* row_data_end = row_wise_data + row_end;
          const size_t partition_start_in_row = std::lower_bound(row_data_start, row_data_end, partition_hist_start) - row_data_start;
          const size_t partition_end_in_row = std::lower_bound(row_data_start, row_data_end, partition_hist_end) - row_data_start;
          for (size_t pos = partition_start_in_row; pos < partition_end_in_row; ++pos) {
            const BIN_TYPE bin = row_data_start[pos];
            CHECK_GE(bin, static_cast<BIN_TYPE>(partition_hist_start));
            data_for_this_partition.emplace_back(bin - partition_hist_start);
          }
          CHECK_GE(partition_end_in_row, partition_start_in_row);
          const data_size_t num_elements_in_row = partition_end_in_row - partition_start_in_row;
          offset += static_cast<DATA_PTR_TYPE>(num_elements_in_row);
          row_ptr_for_this_partition.emplace_back(offset);
          if (num_elements_in_row > thread_max_elements_per_row[thread_index]) {
            thread_max_elements_per_row[thread_index] = num_elements_in_row;
          }
        }
      }
    });
  partition_ptr->clear();
  DATA_PTR_TYPE offset = 0;
  partition_ptr->emplace_back(offset);
  for (size_t i = 0; i < partitioned_row_ptr->size(); ++i) {
    offset += partitioned_row_ptr->at(i).back();
    partition_ptr->emplace_back(offset);
  }
  max_num_column_per_partition_ = 0;
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    if (thread_max_elements_per_row[thread_index] > max_num_column_per_partition_) {
      max_num_column_per_partition_ = thread_max_elements_per_row[thread_index];
    }
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
