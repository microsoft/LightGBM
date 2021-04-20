/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "new_cuda_tree_learner.hpp"

#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/feature_group.h>

namespace LightGBM {

NewCUDATreeLearner::NewCUDATreeLearner(const Config* config): SerialTreeLearner(config) {

}

NewCUDATreeLearner::~NewCUDATreeLearner() {}

void NewCUDATreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  int num_total_gpus = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDeviceCount(&num_total_gpus));
  num_gpus_ = config_->num_gpu > num_total_gpus ? num_total_gpus : config_->num_gpu;
  num_threads_ = OMP_NUM_THREADS();

  AllocateFeatureTasks();
  AllocateCUDAMemory();

  CreateCUDAHistogramConstructor();
}

void NewCUDATreeLearner::BeforeTrain() {
  SerialTreeLearner::BeforeTrain();
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int device_id = 0; device_id < num_gpus_; ++device_id) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
    device_leaf_splits_initializer_[device_id]->Init();
  }
}

void NewCUDATreeLearner::AllocateFeatureTasks() {
  device_feature_groups_.resize(num_gpus_);
  device_num_total_bins_.resize(num_gpus_, 0);
  const int num_feature_groups = train_data_->num_feature_groups();
  const int num_feature_groups_per_device = (num_feature_groups + num_gpus_ - 1) / num_gpus_;
  for (int device_id = 0; device_id < num_gpus_; ++device_id) {
    device_feature_groups_[device_id].clear();
    const int device_feature_group_start = device_id * num_feature_groups_per_device;
    const int device_feature_group_end = std::min(device_feature_group_start + num_feature_groups_per_device, num_feature_groups);
    int& num_total_bin = device_num_total_bins_[device_id];
    num_total_bin = 0;
    for (int group_id = device_feature_group_start; group_id < device_feature_group_end; ++group_id) {
      device_feature_groups_.emplace_back(group_id);
      num_total_bin += train_data_->FeatureGroupNumBin(group_id);
    }
  }
}

void NewCUDATreeLearner::AllocateCUDAMemory() {
  device_data_indices_.resize(num_gpus_, nullptr);
  device_gradients_.resize(num_gpus_, nullptr);
  if (config_->is_constant_hessian) {
    device_hessians_.resize(num_gpus_, nullptr);
  }
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int device_id = 0; device_id < num_gpus_; ++device_id) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
    if (device_data_indices_[device_id] != nullptr) {
      CUDASUCCESS_OR_FATAL(cudaFree(device_data_indices_[device_id]));
    }
    CUDASUCESS_OR_FATAL(cudaMalloc(&(device_data_indices_[device_id]), num_data_));
    if (device_gradients_[device_id] != nullptr) {
      CUDASUCCESS_OR_FATAL(cudaFree(device_gradients_[device_id]));
    }
    CUDASUCESS_OR_FATAL(cudaMalloc(&(device_gradients_[device_id]), num_data_));
    if (config_->is_constant_hessian) {
      if (device_hessians_[device_id] != nullptr) {
        CUDASUCCESS_OR_FATAL(cudaFree(device_hessians_[device_id]));
      }
      CUDASUCESS_OR_FATAL(cudaMalloc(&(device_hessians_[device_id]), num_data_));
    }
  }
}

void NewCUDATreeLearner::CreateCUDAHistogramConstructors() {
  device_histogram_constructors_.resize(num_gpus_);
  device_leaf_splits_initializers_.resize(num_gpus_);
  device_best_split_finders_.resize(num_gpus_);
  device_splitters_.ressize(num_gpus_);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int device_id = 0; device_id < num_gpus_; ++device_id) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
    device_leaf_splits_initializers_[device_id].reset(
      new CUDALeafSplitsInit(device_gradients_[device_id], device_hessians_[device_id]));
    device_histogram_constructors_[device_id].reset(
      new CUDAHistogramConstructor(device_feature_groups_[device_id],
        train_data_, config_->num_leaves, device_histograms_[device_id])));
    device_best_split_finders_[device_id].reset(
      new CUDABestSplitFinder(device_histogram_constructors_[device_id]->cuda_hist(),
        train_data_, device_feature_groups_[device_id], config_->num_leaves));
    device_splitters_[device_id].reset(
      new CUDADataSplitter(device_data_indices_[device_id], num_data_));
  }
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int device_id = 0; device_id < num_gpus_; ++device_id) {
    device_leaf_splits_initializers_[device_id]->Init();
    device_histogram_constructors_[device_id]->Init();
  }
  PushDataIntoDeviceHistogramConstructors();
}

void NewCUDATreeLearner::PushDataIntoDeviceHistogramConstructors() {
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int device_id = 0; device_id < num_gpus_; ++device_id) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
    CUDAHistogramConstructor* cuda_histogram_constructor = device_histogram_constructors_[device_id].get();
    for (int group_id : device_feature_groups_[device_id]) {
      BinIterator* iter = train_data_->FeatureGroupIterator(group_id);
      iter->Reset(0);
      for (const data_size_t data_index = 0; data_index < num_data_; ++data_index) {
        const uint32_t bin = static_cast<uint32_t>(iter->RawGet(data_index));
        cuda_histogram_constructor->PushOneData(bin, group_id, data_index);
      }
    }
    // call finish load to tranfer data from CPU to GPU
    cuda_histogram_constructor->FinishLoad();
  }
}

void NewCUDATreeLearner::FindBestSplits(const Tree* tree) {
  std::vector<int8_t> is_feature_used(num_features_, 1);
  ConstructHistograms(is_feature_used, true);
  FindBestSplitsFromHistograms(is_feature_used, true, tree);
}

void NewCUDATreeLearner::ConstructHistograms(const std::vector<int8_t>& /*is_feature_used*/,
  bool /*use_subtract*/) {
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int device_id = 0; device_id < num_gpus_; ++device_id) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
    device_histogram_constructors_[device_id]->ConstructHistogramForLeaf(
      device_leaf_splits_initializers_[device_id]->smaller_leaf_index(),
      device_leaf_splits_initializers_[device_id]->larger_leaf_index());
  }
}

void NewCUDATreeLearner::FindBestSplitsFromHistograms(const std::vector<int8_t>& /*is_feature_used*/,
  bool /*use_subtract*/, const Tree* /*tree*/) {
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int device_id = 0; device_id < num_gpus_; ++device_id) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
    device_best_split_finders_[device_id]->FindBestSplitsForLeaf(
      device_leaf_splits_initializers_[device_id]->smaller_leaf_index());
    device_best_split_finders_[device_id]->FindBestSplitsForLeaf(
      device_leaf_splits_initializers_[device_id]->larger_leaf_index());
    device_best_split_finders_[device_id]->FindBestFromAllSplits();
  }
}

void NewCUDATreeLearner::Split(Tree* /*tree*/, int /*best_leaf*/,
  int* /*left_leaf*/, int* /*right_leaf*/) {
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int device_id = 0; device_id < num_gpus_; ++device_id) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
    device_splitters_[device_id]->Split(
      device_best_split_finders_[device_id]->best_leaf(),
      device_best_split_finders_[device_id]->best_split_feature_index(),
      device_best_split_finders_[device_id]->best_split_threshold());
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
