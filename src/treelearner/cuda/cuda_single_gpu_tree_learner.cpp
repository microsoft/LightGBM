/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_single_gpu_tree_learner.hpp"

#include <LightGBM/cuda/cuda_tree.hpp>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/feature_group.h>
#include <LightGBM/objective_function.h>

#include <memory>

namespace LightGBM {

CUDASingleGPUTreeLearner::CUDASingleGPUTreeLearner(const Config* config): SerialTreeLearner(config) {
  cuda_gradients_ = nullptr;
  cuda_hessians_ = nullptr;
}

CUDASingleGPUTreeLearner::~CUDASingleGPUTreeLearner() {
  DeallocateCUDAMemory<score_t>(&cuda_gradients_, __FILE__, __LINE__);
  DeallocateCUDAMemory<score_t>(&cuda_hessians_, __FILE__, __LINE__);
}

void CUDASingleGPUTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_threads_ = OMP_NUM_THREADS();
  // use the first gpu by default
  gpu_device_id_ = config_->gpu_device_id >= 0 ? config_->gpu_device_id : 0;
  CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_device_id_));

  cuda_smaller_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_smaller_leaf_splits_->Init();
  cuda_larger_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_larger_leaf_splits_->Init();

  cuda_histogram_constructor_.reset(new CUDAHistogramConstructor(train_data_, config_->num_leaves, num_threads_,
    share_state_->feature_hist_offsets(),
    config_->min_data_in_leaf, config_->min_sum_hessian_in_leaf, gpu_device_id_));
  cuda_histogram_constructor_->Init(train_data_, share_state_.get());

  cuda_data_partition_.reset(new CUDADataPartition(
    train_data_, share_state_->feature_hist_offsets().back(), config_->num_leaves, num_threads_,
    cuda_histogram_constructor_->cuda_hist_pointer()));
  cuda_data_partition_->Init();

  cuda_best_split_finder_.reset(new CUDABestSplitFinder(cuda_histogram_constructor_->cuda_hist(),
    train_data_, this->share_state_->feature_hist_offsets(), config_));
  cuda_best_split_finder_->Init();

  leaf_best_split_feature_.resize(config_->num_leaves, -1);
  leaf_best_split_threshold_.resize(config_->num_leaves, 0);
  leaf_best_split_default_left_.resize(config_->num_leaves, 0);
  leaf_num_data_.resize(config_->num_leaves, 0);
  leaf_data_start_.resize(config_->num_leaves, 0);
  leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);

  AllocateCUDAMemory<score_t>(&cuda_gradients_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemory<score_t>(&cuda_hessians_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
}

void CUDASingleGPUTreeLearner::BeforeTrain() {
  const data_size_t root_num_data = cuda_data_partition_->root_num_data();
  CopyFromHostToCUDADevice<score_t>(cuda_gradients_, gradients_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  CopyFromHostToCUDADevice<score_t>(cuda_hessians_, hessians_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  const data_size_t* leaf_splits_init_indices =
    cuda_data_partition_->use_bagging() ? cuda_data_partition_->cuda_data_indices() : nullptr;
  cuda_data_partition_->BeforeTrain();
  cuda_smaller_leaf_splits_->InitValues(
    cuda_gradients_,
    cuda_hessians_,
    leaf_splits_init_indices,
    cuda_data_partition_->cuda_data_indices(),
    root_num_data,
    cuda_histogram_constructor_->cuda_hist_pointer(),
    &leaf_sum_hessians_[0]);
  leaf_num_data_[0] = root_num_data;
  cuda_larger_leaf_splits_->InitValues();
  cuda_histogram_constructor_->BeforeTrain(cuda_gradients_, cuda_hessians_);
  col_sampler_.ResetByTree();
  cuda_best_split_finder_->BeforeTrain(col_sampler_.is_feature_used_bytree());
  leaf_data_start_[0] = 0;
  smaller_leaf_index_ = 0;
  larger_leaf_index_ = -1;
}

void CUDASingleGPUTreeLearner::AddPredictionToScore(const Tree* tree, double* out_score) const {
  cuda_data_partition_->UpdateTrainScore(tree, out_score);
}

Tree* CUDASingleGPUTreeLearner::Train(const score_t* gradients,
  const score_t* hessians, bool /*is_first_tree*/) {
  gradients_ = gradients;
  hessians_ = hessians;
  global_timer.Start("CUDASingleGPUTreeLearner::BeforeTrain");
  BeforeTrain();
  global_timer.Stop("CUDASingleGPUTreeLearner::BeforeTrain");
  const bool track_branch_features = !(config_->interaction_constraints_vector.empty());
  std::unique_ptr<CUDATree> tree(new CUDATree(config_->num_leaves, track_branch_features, config_->linear_tree, config_->gpu_device_id));
  for (int i = 0; i < config_->num_leaves - 1; ++i) {
    global_timer.Start("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf");
    const data_size_t num_data_in_smaller_leaf = leaf_num_data_[smaller_leaf_index_];
    const data_size_t num_data_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_num_data_[larger_leaf_index_];
    const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[smaller_leaf_index_];
    const double sum_hessians_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_sum_hessians_[larger_leaf_index_];
    cuda_histogram_constructor_->ConstructHistogramForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf,
      sum_hessians_in_larger_leaf);
    global_timer.Stop("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf");
    global_timer.Start("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf");
    cuda_best_split_finder_->FindBestSplitsForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      smaller_leaf_index_, larger_leaf_index_,
      num_data_in_smaller_leaf, num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf);
    global_timer.Stop("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf");
    global_timer.Start("CUDASingleGPUTreeLearner::FindBestFromAllSplits");
    const CUDASplitInfo* best_split_info = nullptr;
    if (larger_leaf_index_ >= 0) {
      best_split_info = cuda_best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(),
        smaller_leaf_index_,
        larger_leaf_index_,
        &leaf_best_split_feature_[smaller_leaf_index_],
        &leaf_best_split_threshold_[smaller_leaf_index_],
        &leaf_best_split_default_left_[smaller_leaf_index_],
        &leaf_best_split_feature_[larger_leaf_index_],
        &leaf_best_split_threshold_[larger_leaf_index_],
        &leaf_best_split_default_left_[larger_leaf_index_],
        &best_leaf_index_);
    } else {
      best_split_info = cuda_best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(),
        smaller_leaf_index_,
        larger_leaf_index_,
        &leaf_best_split_feature_[smaller_leaf_index_],
        &leaf_best_split_threshold_[smaller_leaf_index_],
        &leaf_best_split_default_left_[smaller_leaf_index_],
        nullptr,
        nullptr,
        nullptr,
        &best_leaf_index_);
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::FindBestFromAllSplits");

    if (best_leaf_index_ == -1) {
      Log::Warning("No further splits with positive gain, training stopped with %d leaves.", (i + 1));
      break;
    }

    global_timer.Start("CUDASingleGPUTreeLearner::Split");
    int right_leaf_index = tree->Split(best_leaf_index_,
                                       train_data_->RealFeatureIndex(leaf_best_split_feature_[best_leaf_index_]),
                                       train_data_->RealThreshold(leaf_best_split_feature_[best_leaf_index_],
                                       leaf_best_split_threshold_[best_leaf_index_]),
                                       train_data_->FeatureBinMapper(leaf_best_split_feature_[best_leaf_index_])->missing_type(),
                                       best_split_info);

    cuda_data_partition_->Split(best_split_info,
                                best_leaf_index_,
                                right_leaf_index,
                                leaf_best_split_feature_[best_leaf_index_],
                                leaf_best_split_threshold_[best_leaf_index_],
                                leaf_best_split_default_left_[best_leaf_index_],
                                leaf_num_data_[best_leaf_index_],
                                leaf_data_start_[best_leaf_index_],
                                cuda_smaller_leaf_splits_->GetCUDAStructRef(),
                                cuda_larger_leaf_splits_->GetCUDAStructRef(),
                                &leaf_num_data_[best_leaf_index_],
                                &leaf_num_data_[right_leaf_index],
                                &leaf_data_start_[best_leaf_index_],
                                &leaf_data_start_[right_leaf_index],
                                &leaf_sum_hessians_[best_leaf_index_],
                                &leaf_sum_hessians_[right_leaf_index]);
    smaller_leaf_index_ = (leaf_num_data_[best_leaf_index_] < leaf_num_data_[right_leaf_index] ? best_leaf_index_ : right_leaf_index);
    larger_leaf_index_ = (smaller_leaf_index_ == best_leaf_index_ ? right_leaf_index : best_leaf_index_);
    global_timer.Stop("CUDASingleGPUTreeLearner::Split");
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  tree->ToHost();
  return tree.release();
}

void CUDASingleGPUTreeLearner::ResetTrainingData(
  const Dataset* train_data,
  bool is_constant_hessian) {
  SerialTreeLearner::ResetTrainingData(train_data, is_constant_hessian);
  CHECK_EQ(num_features_, train_data_->num_features());
  cuda_histogram_constructor_->ResetTrainingData(train_data, share_state_.get());
  cuda_data_partition_->ResetTrainingData(train_data,
    static_cast<int>(share_state_->feature_hist_offsets().back()),
    cuda_histogram_constructor_->cuda_hist_pointer());
  cuda_best_split_finder_->ResetTrainingData(
    cuda_histogram_constructor_->cuda_hist(),
    train_data,
    share_state_->feature_hist_offsets());
  cuda_smaller_leaf_splits_->Resize(num_data_);
  cuda_larger_leaf_splits_->Resize(num_data_);
  CHECK_EQ(is_constant_hessian, share_state_->is_constant_hessian);
  DeallocateCUDAMemory<score_t>(&cuda_gradients_, __FILE__, __LINE__);
  DeallocateCUDAMemory<score_t>(&cuda_hessians_, __FILE__, __LINE__);
  AllocateCUDAMemory<score_t>(&cuda_gradients_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  AllocateCUDAMemory<score_t>(&cuda_hessians_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
}

void CUDASingleGPUTreeLearner::ResetConfig(const Config* config) {
  const int old_num_leaves = config_->num_leaves;
  SerialTreeLearner::ResetConfig(config);
  if (config_->gpu_device_id >= 0 && config_->gpu_device_id != gpu_device_id_) {
    Log::Fatal("Changing gpu device ID by resetting configuration parameter is not allowed for CUDA tree learner.");
  }
  num_threads_ = OMP_NUM_THREADS();
  if (config_->num_leaves != old_num_leaves) {
    leaf_best_split_feature_.resize(config_->num_leaves, -1);
    leaf_best_split_threshold_.resize(config_->num_leaves, 0);
    leaf_best_split_default_left_.resize(config_->num_leaves, 0);
    leaf_num_data_.resize(config_->num_leaves, 0);
    leaf_data_start_.resize(config_->num_leaves, 0);
    leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);
  }
  cuda_histogram_constructor_->ResetConfig(config);
  cuda_best_split_finder_->ResetConfig(config);
  cuda_data_partition_->ResetConfig(config);
}

void CUDASingleGPUTreeLearner::SetBaggingData(const Dataset* /*subset*/,
  const data_size_t* used_indices, data_size_t num_data) {
  cuda_data_partition_->SetUsedDataIndices(used_indices, num_data);
}

void CUDASingleGPUTreeLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                                         data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const {
  CHECK(tree->is_cuda_tree());
  CUDATree* cuda_tree = reinterpret_cast<CUDATree*>(tree);
  SerialTreeLearner::RenewTreeOutput(tree, obj, residual_getter, total_num_data, bag_indices, bag_cnt);
  cuda_tree->SyncLeafOutputFromHostToCUDA();
}

}  // namespace LightGBM

#endif  // USE_CUDA
