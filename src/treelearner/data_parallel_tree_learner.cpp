/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cstring>
#include <tuple>
#include <vector>

#include "parallel_tree_learner.h"

namespace LightGBM {

template <typename TREELEARNER_T>
DataParallelTreeLearner<TREELEARNER_T>::DataParallelTreeLearner(const Config* config)
  :TREELEARNER_T(config) {
}

template <typename TREELEARNER_T>
DataParallelTreeLearner<TREELEARNER_T>::~DataParallelTreeLearner() {
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::Init(const Dataset* train_data, bool is_constant_hessian) {
  // initialize SerialTreeLearner
  TREELEARNER_T::Init(train_data, is_constant_hessian);
  // Get local rank and global machine size
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();

  auto max_cat_threshold = this->config_->max_cat_threshold;
  // need to be able to hold smaller and larger best splits in SyncUpGlobalBestSplit
  size_t split_info_size = static_cast<size_t>(SplitInfo::Size(max_cat_threshold) * 2);
  size_t histogram_size = static_cast<size_t>(this->share_state_->num_hist_total_bin() * kHistEntrySize);

  // allocate buffer for communication
  size_t buffer_size = std::max(histogram_size, split_info_size);

  input_buffer_.resize(buffer_size);
  output_buffer_.resize(buffer_size);

  is_feature_aggregated_.resize(this->num_features_);

  block_start_.resize(num_machines_);
  block_len_.resize(num_machines_);

  buffer_write_start_pos_.resize(this->num_features_);
  buffer_read_start_pos_.resize(this->num_features_);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::ResetConfig(const Config* config) {
  TREELEARNER_T::ResetConfig(config);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::BeforeTrain() {
  TREELEARNER_T::BeforeTrain();
  // generate feature partition for current tree
  std::vector<std::vector<int>> feature_distribution(num_machines_, std::vector<int>());
  std::vector<int> num_bins_distributed(num_machines_, 0);
  for (int i = 0; i < this->train_data_->num_total_features(); ++i) {
    int inner_feature_index = this->train_data_->InnerFeatureIndex(i);
    if (inner_feature_index == -1) { continue; }
    if (this->col_sampler_.is_feature_used_bytree()[inner_feature_index]) {
      int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));
      feature_distribution[cur_min_machine].push_back(inner_feature_index);
      auto num_bin = this->train_data_->FeatureNumBin(inner_feature_index);
      if (this->train_data_->FeatureBinMapper(inner_feature_index)->GetMostFreqBin() == 0) {
        num_bin -= 1;
      }
      num_bins_distributed[cur_min_machine] += num_bin;
    }
    is_feature_aggregated_[inner_feature_index] = false;
  }
  // get local used feature
  for (auto fid : feature_distribution[rank_]) {
    is_feature_aggregated_[fid] = true;
  }

  // get block start and block len for reduce scatter
  reduce_scatter_size_ = 0;
  for (int i = 0; i < num_machines_; ++i) {
    block_len_[i] = 0;
    for (auto fid : feature_distribution[i]) {
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetMostFreqBin() == 0) {
        num_bin -= 1;
      }
      block_len_[i] += num_bin * kHistEntrySize;
    }
    reduce_scatter_size_ += block_len_[i];
  }

  block_start_[0] = 0;
  for (int i = 1; i < num_machines_; ++i) {
    block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
  }

  // get buffer_write_start_pos_
  int bin_size = 0;
  for (int i = 0; i < num_machines_; ++i) {
    for (auto fid : feature_distribution[i]) {
      buffer_write_start_pos_[fid] = bin_size;
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetMostFreqBin() == 0) {
        num_bin -= 1;
      }
      bin_size += num_bin * kHistEntrySize;
    }
  }

  // get buffer_read_start_pos_
  bin_size = 0;
  for (auto fid : feature_distribution[rank_]) {
    buffer_read_start_pos_[fid] = bin_size;
    auto num_bin = this->train_data_->FeatureNumBin(fid);
    if (this->train_data_->FeatureBinMapper(fid)->GetMostFreqBin() == 0) {
      num_bin -= 1;
    }
    bin_size += num_bin * kHistEntrySize;
  }

  // sync global data sumup info
  std::tuple<data_size_t, double, double> data(this->smaller_leaf_splits_->num_data_in_leaf(),
                                               this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians());
  int size = sizeof(data);
  std::memcpy(input_buffer_.data(), &data, size);
  // global sumup reduce
  Network::Allreduce(input_buffer_.data(), size, sizeof(std::tuple<data_size_t, double, double>), output_buffer_.data(), [](const char *src, char *dst, int type_size, comm_size_t len) {
    comm_size_t used_size = 0;
    const std::tuple<data_size_t, double, double> *p1;
    std::tuple<data_size_t, double, double> *p2;
    while (used_size < len) {
      p1 = reinterpret_cast<const std::tuple<data_size_t, double, double> *>(src);
      p2 = reinterpret_cast<std::tuple<data_size_t, double, double> *>(dst);
      std::get<0>(*p2) = std::get<0>(*p2) + std::get<0>(*p1);
      std::get<1>(*p2) = std::get<1>(*p2) + std::get<1>(*p1);
      std::get<2>(*p2) = std::get<2>(*p2) + std::get<2>(*p1);
      src += type_size;
      dst += type_size;
      used_size += type_size;
    }
  });
  // copy back
  std::memcpy(reinterpret_cast<void*>(&data), output_buffer_.data(), size);
  // set global sumup info
  this->smaller_leaf_splits_->Init(std::get<1>(data), std::get<2>(data));
  // init global data count in leaf
  global_data_count_in_leaf_[0] = std::get<0>(data);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplits(const Tree* tree) {
  TREELEARNER_T::ConstructHistograms(
      this->col_sampler_.is_feature_used_bytree(), true);
  const int smaller_leaf_index = this->smaller_leaf_splits_->leaf_index();
  const data_size_t local_data_on_smaller_leaf = this->data_partition_->leaf_count(smaller_leaf_index);
  if (local_data_on_smaller_leaf <= 0) {
    // clear histogram buffer before synchronizing
    // otherwise histogram contents from the previous iteration will be sent
    #pragma omp parallel for schedule(static)
    for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
      if (this->col_sampler_.is_feature_used_bytree()[feature_index] == false)
        continue;
      const BinMapper* feature_bin_mapper = this->train_data_->FeatureBinMapper(feature_index);
      const int offset = static_cast<int>(feature_bin_mapper->GetMostFreqBin() == 0);
      const int num_bin = feature_bin_mapper->num_bin();
      hist_t* hist_ptr = this->smaller_leaf_histogram_array_[feature_index].RawData();
      std::memset(reinterpret_cast<void*>(hist_ptr), 0, (num_bin - offset) * kHistEntrySize);
    }
  }
  // construct local histograms
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    if (this->col_sampler_.is_feature_used_bytree()[feature_index] == false)
      continue;
    // copy to buffer
    std::memcpy(input_buffer_.data() + buffer_write_start_pos_[feature_index],
                this->smaller_leaf_histogram_array_[feature_index].RawData(),
                this->smaller_leaf_histogram_array_[feature_index].SizeOfHistgram());
  }
  // Reduce scatter for histogram
  Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, sizeof(hist_t), block_start_.data(),
                         block_len_.data(), output_buffer_.data(), static_cast<comm_size_t>(output_buffer_.size()), &HistogramSumReducer);
  this->FindBestSplitsFromHistograms(
      this->col_sampler_.is_feature_used_bytree(), true, tree);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplitsFromHistograms(const std::vector<int8_t>&, bool, const Tree* tree) {
  std::vector<SplitInfo> smaller_bests_per_thread(this->share_state_->num_threads);
  std::vector<SplitInfo> larger_bests_per_thread(this->share_state_->num_threads);
  std::vector<int8_t> smaller_node_used_features =
      this->col_sampler_.GetByNode(tree, this->smaller_leaf_splits_->leaf_index());
  std::vector<int8_t> larger_node_used_features =
      this->col_sampler_.GetByNode(tree, this->larger_leaf_splits_->leaf_index());
  double smaller_leaf_parent_output = this->GetParentOutput(tree, this->smaller_leaf_splits_.get());
  double larger_leaf_parent_output = this->GetParentOutput(tree, this->larger_leaf_splits_.get());
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_aggregated_[feature_index]) continue;
    const int tid = omp_get_thread_num();
    const int real_feature_index = this->train_data_->RealFeatureIndex(feature_index);
    // restore global histograms from buffer
    this->smaller_leaf_histogram_array_[feature_index].FromMemory(
      output_buffer_.data() + buffer_read_start_pos_[feature_index]);

    this->train_data_->FixHistogram(feature_index,
                                    this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians(),
                                    this->smaller_leaf_histogram_array_[feature_index].RawData());

    this->ComputeBestSplitForFeature(
        this->smaller_leaf_histogram_array_, feature_index, real_feature_index,
        smaller_node_used_features[feature_index],
        GetGlobalDataCountInLeaf(this->smaller_leaf_splits_->leaf_index()),
        this->smaller_leaf_splits_.get(),
        &smaller_bests_per_thread[tid],
        smaller_leaf_parent_output);

    // only root leaf
    if (this->larger_leaf_splits_ == nullptr || this->larger_leaf_splits_->leaf_index() < 0) continue;

    // construct histgroms for large leaf, we init larger leaf as the parent, so we can just subtract the smaller leaf's histograms
    this->larger_leaf_histogram_array_[feature_index].Subtract(
      this->smaller_leaf_histogram_array_[feature_index]);

    this->ComputeBestSplitForFeature(
        this->larger_leaf_histogram_array_, feature_index, real_feature_index,
        larger_node_used_features[feature_index],
        GetGlobalDataCountInLeaf(this->larger_leaf_splits_->leaf_index()),
        this->larger_leaf_splits_.get(),
        &larger_bests_per_thread[tid],
        larger_leaf_parent_output);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_bests_per_thread);
  int leaf = this->smaller_leaf_splits_->leaf_index();
  this->best_split_per_leaf_[leaf] = smaller_bests_per_thread[smaller_best_idx];

  if (this->larger_leaf_splits_ != nullptr &&  this->larger_leaf_splits_->leaf_index() >= 0) {
    leaf = this->larger_leaf_splits_->leaf_index();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_bests_per_thread);
    this->best_split_per_leaf_[leaf] = larger_bests_per_thread[larger_best_idx];
  }

  SplitInfo smaller_best_split, larger_best_split;
  smaller_best_split = this->best_split_per_leaf_[this->smaller_leaf_splits_->leaf_index()];
  // find local best split for larger leaf
  if (this->larger_leaf_splits_->leaf_index() >= 0) {
    larger_best_split = this->best_split_per_leaf_[this->larger_leaf_splits_->leaf_index()];
  }

  // sync global best info
  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold);

  // set best split
  this->best_split_per_leaf_[this->smaller_leaf_splits_->leaf_index()] = smaller_best_split;
  if (this->larger_leaf_splits_->leaf_index() >= 0) {
    this->best_split_per_leaf_[this->larger_leaf_splits_->leaf_index()] = larger_best_split;
  }
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  TREELEARNER_T::SplitInner(tree, best_Leaf, left_leaf, right_leaf, false);
  const SplitInfo& best_split_info = this->best_split_per_leaf_[best_Leaf];
  // need update global number of data in leaf
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
}

// instantiate template classes, otherwise linker cannot find the code
template class DataParallelTreeLearner<CUDATreeLearner>;
template class DataParallelTreeLearner<GPUTreeLearner>;
template class DataParallelTreeLearner<SerialTreeLearner>;

}  // namespace LightGBM
