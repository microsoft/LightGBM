/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_NEW_CUDA_TREE_LEARNER_HPP_
#define LIGHTGBM_NEW_CUDA_TREE_LEARNER_HPP_

#ifdef USE_CUDA

#include "../serial_tree_learner.h"
#include "cuda_leaf_splits_init.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_data_splitter.hpp"

namespace LightGBM {

class NewCUDATreeLearner: public SerialTreeLearner {
 public:
  explicit NewCUDATreeLearner(const Config* config);

  ~NewCUDATreeLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ResetTrainingData(const Dataset* train_data,
                         bool is_constant_hessian) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;
  
  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override;

 protected:
  void AllocateFeatureTasks();

  void AllocateCUDAMemory(const bool is_constant_hessian);

  void CreateCUDAHistogramConstructors();

  void PushDataIntoDeviceHistogramConstructors();

  void FindBestSplits(const Tree* tree) override;

  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;

  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract, const Tree* tree) override;

  void Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf) override;

  void BeforeTrain() override;

  // number of GPUs
  int num_gpus_;
  // number of threads on CPU
  int num_threads_;

  // feature groups allocated to each device
  std::vector<std::vector<int>> device_feature_groups_;
  // number of total bins of feature groups allocated to each device
  std::vector<int> device_num_total_bins_;
  // number of maximum work groups per device
  std::vector<int> device_num_workgroups_;

  // full data indices on CUDA devices, as the data indices of data_partition_ in CPU version
  std::vector<data_size_t*> device_data_indices_;
  // gradient values on CUDA devices
  std::vector<score_t*> device_gradients_;
  // hessian values on CUDA devices
  std::vector<score_t*> device_hessians_;
  // histogram storage on CUDA devices
  std::vector<hist_t*> device_histograms_;

  // device leaf splits initializer
  std::vector<std::unique_ptr<CUDALeafSplitsInit>> device_leaf_splits_initializers_;
  // device histogram constructors
  std::vector<std::unique_ptr<CUDAHistogramConstructor>> device_histogram_constructors_;
  // device best split finder
  std::vector<std::unique_ptr<CUDABestSplitFinder>> device_best_split_finders_;
  // device splitter
  std::vector<std::unique_ptr<CUDADataSplitter>> device_splitters_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_TREE_LEARNER_HPP_
