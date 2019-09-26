/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/utils/common.h>

#include <cstring>
#include <tuple>
#include <vector>

#include "parallel_tree_learner.h"

namespace LightGBM {

template <typename TREELEARNER_T>
VotingParallelTreeLearner<TREELEARNER_T>::VotingParallelTreeLearner(const Config* config)
  :TREELEARNER_T(config) {
  top_k_ = this->config_->top_k;
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::Init(const Dataset* train_data, bool is_constant_hessian) {
  TREELEARNER_T::Init(train_data, is_constant_hessian);
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();

  // limit top k
  if (top_k_ > this->num_features_) {
    top_k_ = this->num_features_;
  }
  // get max bin
  int max_bin = 0;
  for (int i = 0; i < this->num_features_; ++i) {
    if (max_bin < this->train_data_->FeatureNumBin(i)) {
      max_bin = this->train_data_->FeatureNumBin(i);
    }
  }
  // calculate buffer size
  size_t buffer_size = 2 * top_k_ * std::max(max_bin * sizeof(HistogramBinEntry), sizeof(LightSplitInfo) * num_machines_);
  // left and right on same time, so need double size
  input_buffer_.resize(buffer_size);
  output_buffer_.resize(buffer_size);

  smaller_is_feature_aggregated_.resize(this->num_features_);
  larger_is_feature_aggregated_.resize(this->num_features_);

  block_start_.resize(num_machines_);
  block_len_.resize(num_machines_);

  smaller_buffer_read_start_pos_.resize(this->num_features_);
  larger_buffer_read_start_pos_.resize(this->num_features_);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);

  smaller_leaf_splits_global_.reset(new LeafSplits(this->train_data_->num_data()));
  larger_leaf_splits_global_.reset(new LeafSplits(this->train_data_->num_data()));

  local_config_ = *this->config_;
  local_config_.min_data_in_leaf /= num_machines_;
  local_config_.min_sum_hessian_in_leaf /= num_machines_;

  this->histogram_pool_.ResetConfig(&local_config_);

  // initialize histograms for global
  smaller_leaf_histogram_array_global_.reset(new FeatureHistogram[this->num_features_]);
  larger_leaf_histogram_array_global_.reset(new FeatureHistogram[this->num_features_]);
  auto num_total_bin = this->train_data_->NumTotalBin();
  smaller_leaf_histogram_data_.resize(num_total_bin);
  larger_leaf_histogram_data_.resize(num_total_bin);
  feature_metas_.resize(train_data->num_features());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < train_data->num_features(); ++i) {
    feature_metas_[i].num_bin = train_data->FeatureNumBin(i);
    feature_metas_[i].default_bin = train_data->FeatureBinMapper(i)->GetDefaultBin();
    feature_metas_[i].missing_type = train_data->FeatureBinMapper(i)->missing_type();
    feature_metas_[i].monotone_type = train_data->FeatureMonotone(i);
    feature_metas_[i].penalty = train_data->FeaturePenalte(i);
    if (train_data->FeatureBinMapper(i)->GetDefaultBin() == 0) {
      feature_metas_[i].bias = 1;
    } else {
      feature_metas_[i].bias = 0;
    }
    feature_metas_[i].config = this->config_;
    feature_metas_[i].bin_type = train_data->FeatureBinMapper(i)->bin_type();
  }
  uint64_t offset = 0;
  for (int j = 0; j < train_data->num_features(); ++j) {
    offset += static_cast<uint64_t>(train_data->SubFeatureBinOffset(j));
    smaller_leaf_histogram_array_global_[j].Init(smaller_leaf_histogram_data_.data() + offset, &feature_metas_[j]);
    larger_leaf_histogram_array_global_[j].Init(larger_leaf_histogram_data_.data() + offset, &feature_metas_[j]);
    auto num_bin = train_data->FeatureNumBin(j);
    if (train_data->FeatureBinMapper(j)->GetDefaultBin() == 0) {
      num_bin -= 1;
    }
    offset += static_cast<uint64_t>(num_bin);
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::ResetConfig(const Config* config) {
  TREELEARNER_T::ResetConfig(config);

  local_config_ = *this->config_;
  local_config_.min_data_in_leaf /= num_machines_;
  local_config_.min_sum_hessian_in_leaf /= num_machines_;

  this->histogram_pool_.ResetConfig(&local_config_);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);

  for (size_t i = 0; i < feature_metas_.size(); ++i) {
    feature_metas_[i].config = this->config_;
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::BeforeTrain() {
  TREELEARNER_T::BeforeTrain();
  // sync global data sumup info
  std::tuple<data_size_t, double, double> data(this->smaller_leaf_splits_->num_data_in_leaf(), this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians());
  int size = sizeof(std::tuple<data_size_t, double, double>);
  std::memcpy(input_buffer_.data(), &data, size);

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

  std::memcpy(reinterpret_cast<void*>(&data), output_buffer_.data(), size);

  // set global sumup info
  smaller_leaf_splits_global_->Init(std::get<1>(data), std::get<2>(data));
  larger_leaf_splits_global_->Init();
  // init global data count in leaf
  global_data_count_in_leaf_[0] = std::get<0>(data);
}

template <typename TREELEARNER_T>
bool VotingParallelTreeLearner<TREELEARNER_T>::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  if (TREELEARNER_T::BeforeFindBestSplit(tree, left_leaf, right_leaf)) {
    data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
    data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
    if (right_leaf < 0) {
      return true;
    } else if (num_data_in_left_child < num_data_in_right_child) {
      // get local sumup
      this->smaller_leaf_splits_->Init(left_leaf, this->data_partition_.get(), this->gradients_, this->hessians_);
      this->larger_leaf_splits_->Init(right_leaf, this->data_partition_.get(), this->gradients_, this->hessians_);
    } else {
      // get local sumup
      this->smaller_leaf_splits_->Init(right_leaf, this->data_partition_.get(), this->gradients_, this->hessians_);
      this->larger_leaf_splits_->Init(left_leaf, this->data_partition_.get(), this->gradients_, this->hessians_);
    }
    return true;
  } else {
    return false;
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::GlobalVoting(int leaf_idx, const std::vector<LightSplitInfo>& splits, std::vector<int>* out) {
  out->clear();
  if (leaf_idx < 0) {
    return;
  }
  // get mean number on machines
  score_t mean_num_data = GetGlobalDataCountInLeaf(leaf_idx) / static_cast<score_t>(num_machines_);
  std::vector<LightSplitInfo> feature_best_split(this->train_data_->num_total_features() , LightSplitInfo());
  for (auto & split : splits) {
    int fid = split.feature;
    if (fid < 0) {
      continue;
    }
    // weighted gain
    double gain = split.gain * (split.left_count + split.right_count) / mean_num_data;
    if (gain > feature_best_split[fid].gain) {
      feature_best_split[fid] = split;
      feature_best_split[fid].gain = gain;
    }
  }
  // get top k
  std::vector<LightSplitInfo> top_k_splits;
  ArrayArgs<LightSplitInfo>::MaxK(feature_best_split, top_k_, &top_k_splits);
  std::stable_sort(top_k_splits.begin(), top_k_splits.end(), std::greater<LightSplitInfo>());
  for (auto& split : top_k_splits) {
    if (split.gain == kMinScore || split.feature == -1) {
      continue;
    }
    out->push_back(split.feature);
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::CopyLocalHistogram(const std::vector<int>& smaller_top_features, const std::vector<int>& larger_top_features) {
  for (int i = 0; i < this->num_features_; ++i) {
    smaller_is_feature_aggregated_[i] = false;
    larger_is_feature_aggregated_[i] = false;
  }
  size_t total_num_features = smaller_top_features.size() + larger_top_features.size();
  size_t average_feature = (total_num_features + num_machines_ - 1) / num_machines_;
  size_t used_num_features = 0, smaller_idx = 0, larger_idx = 0;
  block_start_[0] = 0;
  reduce_scatter_size_ = 0;
  // Copy histogram to buffer, and Get local aggregate features
  for (int i = 0; i < num_machines_; ++i) {
    size_t cur_size = 0, cur_used_features = 0;
    size_t cur_total_feature = std::min(average_feature, total_num_features - used_num_features);
    // copy histograms.
    while (cur_used_features < cur_total_feature) {
      // copy smaller leaf histograms first
      if (smaller_idx < smaller_top_features.size()) {
        int inner_feature_index = this->train_data_->InnerFeatureIndex(smaller_top_features[smaller_idx]);
        ++cur_used_features;
        // mark local aggregated feature
        if (i == rank_) {
          smaller_is_feature_aggregated_[inner_feature_index] = true;
          smaller_buffer_read_start_pos_[inner_feature_index] = static_cast<int>(cur_size);
        }
        // copy
        std::memcpy(input_buffer_.data() + reduce_scatter_size_, this->smaller_leaf_histogram_array_[inner_feature_index].RawData(), this->smaller_leaf_histogram_array_[inner_feature_index].SizeOfHistgram());
        cur_size += this->smaller_leaf_histogram_array_[inner_feature_index].SizeOfHistgram();
        reduce_scatter_size_ += this->smaller_leaf_histogram_array_[inner_feature_index].SizeOfHistgram();
        ++smaller_idx;
      }
      if (cur_used_features >= cur_total_feature) {
        break;
      }
      // then copy larger leaf histograms
      if (larger_idx < larger_top_features.size()) {
        int inner_feature_index = this->train_data_->InnerFeatureIndex(larger_top_features[larger_idx]);
        ++cur_used_features;
        // mark local aggregated feature
        if (i == rank_) {
          larger_is_feature_aggregated_[inner_feature_index] = true;
          larger_buffer_read_start_pos_[inner_feature_index] = static_cast<int>(cur_size);
        }
        // copy
        std::memcpy(input_buffer_.data() + reduce_scatter_size_, this->larger_leaf_histogram_array_[inner_feature_index].RawData(), this->larger_leaf_histogram_array_[inner_feature_index].SizeOfHistgram());
        cur_size += this->larger_leaf_histogram_array_[inner_feature_index].SizeOfHistgram();
        reduce_scatter_size_ += this->larger_leaf_histogram_array_[inner_feature_index].SizeOfHistgram();
        ++larger_idx;
      }
    }
    used_num_features += cur_used_features;
    block_len_[i] = static_cast<int>(cur_size);
    if (i < num_machines_ - 1) {
      block_start_[i + 1] = block_start_[i] + block_len_[i];
    }
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::FindBestSplits() {
  // use local data to find local best splits
  std::vector<int8_t> is_feature_used(this->num_features_, 0);
#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    if (!this->is_feature_used_[feature_index]) continue;
    if (this->parent_leaf_histogram_array_ != nullptr
      && !this->parent_leaf_histogram_array_[feature_index].is_splittable()) {
      this->smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    is_feature_used[feature_index] = 1;
  }
  bool use_subtract = true;
  if (this->parent_leaf_histogram_array_ == nullptr) {
    use_subtract = false;
  }
  TREELEARNER_T::ConstructHistograms(is_feature_used, use_subtract);

  std::vector<SplitInfo> smaller_bestsplit_per_features(this->num_features_);
  std::vector<SplitInfo> larger_bestsplit_per_features(this->num_features_);

  OMP_INIT_EX();
  // find splits
#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_used[feature_index]) { continue; }
    const int real_feature_index = this->train_data_->RealFeatureIndex(feature_index);
    this->train_data_->FixHistogram(feature_index,
      this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians(),
      this->smaller_leaf_splits_->num_data_in_leaf(),
      this->smaller_leaf_histogram_array_[feature_index].RawData());

    this->smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
      this->smaller_leaf_splits_->sum_gradients(),
      this->smaller_leaf_splits_->sum_hessians(),
      this->smaller_leaf_splits_->num_data_in_leaf(),
      this->smaller_leaf_splits_->min_constraint(),
      this->smaller_leaf_splits_->max_constraint(),
      &smaller_bestsplit_per_features[feature_index]);
    smaller_bestsplit_per_features[feature_index].feature = real_feature_index;
    // only has root leaf
    if (this->larger_leaf_splits_ == nullptr || this->larger_leaf_splits_->LeafIndex() < 0) { continue; }

    if (use_subtract) {
      this->larger_leaf_histogram_array_[feature_index].Subtract(this->smaller_leaf_histogram_array_[feature_index]);
    } else {
      this->train_data_->FixHistogram(feature_index, this->larger_leaf_splits_->sum_gradients(), this->larger_leaf_splits_->sum_hessians(),
        this->larger_leaf_splits_->num_data_in_leaf(),
        this->larger_leaf_histogram_array_[feature_index].RawData());
    }
    // find best threshold for larger child
    this->larger_leaf_histogram_array_[feature_index].FindBestThreshold(
      this->larger_leaf_splits_->sum_gradients(),
      this->larger_leaf_splits_->sum_hessians(),
      this->larger_leaf_splits_->num_data_in_leaf(),
      this->larger_leaf_splits_->min_constraint(),
      this->larger_leaf_splits_->max_constraint(),
      &larger_bestsplit_per_features[feature_index]);
    larger_bestsplit_per_features[feature_index].feature = real_feature_index;
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  std::vector<SplitInfo> smaller_top_k_splits, larger_top_k_splits;
  // local voting
  ArrayArgs<SplitInfo>::MaxK(smaller_bestsplit_per_features, top_k_, &smaller_top_k_splits);
  ArrayArgs<SplitInfo>::MaxK(larger_bestsplit_per_features, top_k_, &larger_top_k_splits);

  std::vector<LightSplitInfo> smaller_top_k_light_splits(top_k_);
  std::vector<LightSplitInfo> larger_top_k_light_splits(top_k_);
  for (int i = 0; i < top_k_; ++i) {
    smaller_top_k_light_splits[i].CopyFrom(smaller_top_k_splits[i]);
    larger_top_k_light_splits[i].CopyFrom(larger_top_k_splits[i]);
  }

  // gather
  int offset = 0;
  for (int i = 0; i < top_k_; ++i) {
    std::memcpy(input_buffer_.data() + offset, &smaller_top_k_light_splits[i], sizeof(LightSplitInfo));
    offset += sizeof(LightSplitInfo);
    std::memcpy(input_buffer_.data() + offset, &larger_top_k_light_splits[i], sizeof(LightSplitInfo));
    offset += sizeof(LightSplitInfo);
  }
  Network::Allgather(input_buffer_.data(), offset, output_buffer_.data());
  // get all top-k from all machines
  std::vector<LightSplitInfo> smaller_top_k_splits_global;
  std::vector<LightSplitInfo> larger_top_k_splits_global;
  offset = 0;
  for (int i = 0; i < num_machines_; ++i) {
    for (int j = 0; j < top_k_; ++j) {
      smaller_top_k_splits_global.push_back(LightSplitInfo());
      std::memcpy(&smaller_top_k_splits_global.back(), output_buffer_.data() + offset, sizeof(LightSplitInfo));
      offset += sizeof(LightSplitInfo);
      larger_top_k_splits_global.push_back(LightSplitInfo());
      std::memcpy(&larger_top_k_splits_global.back(), output_buffer_.data() + offset, sizeof(LightSplitInfo));
      offset += sizeof(LightSplitInfo);
    }
  }
  // global voting
  std::vector<int> smaller_top_features, larger_top_features;
  GlobalVoting(this->smaller_leaf_splits_->LeafIndex(), smaller_top_k_splits_global, &smaller_top_features);
  GlobalVoting(this->larger_leaf_splits_->LeafIndex(), larger_top_k_splits_global, &larger_top_features);
  // copy local histgrams to buffer
  CopyLocalHistogram(smaller_top_features, larger_top_features);

  // Reduce scatter for histogram
  Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, sizeof(HistogramBinEntry), block_start_.data(), block_len_.data(),
                         output_buffer_.data(), static_cast<comm_size_t>(output_buffer_.size()), &HistogramBinEntry::SumReducer);

  this->FindBestSplitsFromHistograms(is_feature_used, false);
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::FindBestSplitsFromHistograms(const std::vector<int8_t>&, bool) {
  std::vector<SplitInfo> smaller_bests_per_thread(this->num_threads_);
  std::vector<SplitInfo> larger_best_per_thread(this->num_threads_);
  std::vector<int8_t> smaller_node_used_features(this->num_features_, 1);
  std::vector<int8_t> larger_node_used_features(this->num_features_, 1);
  if (this->config_->feature_fraction_bynode < 1.0f) {
    smaller_node_used_features = this->GetUsedFeatures(false);
    larger_node_used_features = this->GetUsedFeatures(false);
  }
  // find best split from local aggregated histograms

  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    const int real_feature_index = this->train_data_->RealFeatureIndex(feature_index);
    if (smaller_is_feature_aggregated_[feature_index]) {
      SplitInfo smaller_split;
      // restore from buffer
      smaller_leaf_histogram_array_global_[feature_index].FromMemory(
        output_buffer_.data() + smaller_buffer_read_start_pos_[feature_index]);

      this->train_data_->FixHistogram(feature_index,
                                      smaller_leaf_splits_global_->sum_gradients(), smaller_leaf_splits_global_->sum_hessians(),
                                      GetGlobalDataCountInLeaf(smaller_leaf_splits_global_->LeafIndex()),
                                      smaller_leaf_histogram_array_global_[feature_index].RawData());

      // find best threshold
      smaller_leaf_histogram_array_global_[feature_index].FindBestThreshold(
        smaller_leaf_splits_global_->sum_gradients(),
        smaller_leaf_splits_global_->sum_hessians(),
        GetGlobalDataCountInLeaf(smaller_leaf_splits_global_->LeafIndex()),
        smaller_leaf_splits_global_->min_constraint(),
        smaller_leaf_splits_global_->max_constraint(),
        &smaller_split);
      smaller_split.feature = real_feature_index;
      if (smaller_split > smaller_bests_per_thread[tid] && smaller_node_used_features[feature_index]) {
        smaller_bests_per_thread[tid] = smaller_split;
      }
    }

    if (larger_is_feature_aggregated_[feature_index]) {
      SplitInfo larger_split;
      // restore from buffer
      larger_leaf_histogram_array_global_[feature_index].FromMemory(output_buffer_.data() + larger_buffer_read_start_pos_[feature_index]);

      this->train_data_->FixHistogram(feature_index,
                                      larger_leaf_splits_global_->sum_gradients(), larger_leaf_splits_global_->sum_hessians(),
                                      GetGlobalDataCountInLeaf(larger_leaf_splits_global_->LeafIndex()),
                                      larger_leaf_histogram_array_global_[feature_index].RawData());

      // find best threshold
      larger_leaf_histogram_array_global_[feature_index].FindBestThreshold(
        larger_leaf_splits_global_->sum_gradients(),
        larger_leaf_splits_global_->sum_hessians(),
        GetGlobalDataCountInLeaf(larger_leaf_splits_global_->LeafIndex()),
        larger_leaf_splits_global_->min_constraint(),
        larger_leaf_splits_global_->max_constraint(),
        &larger_split);
      larger_split.feature = real_feature_index;
      if (larger_split > larger_best_per_thread[tid] && larger_node_used_features[feature_index]) {
        larger_best_per_thread[tid] = larger_split;
      }
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_bests_per_thread);
  int leaf = this->smaller_leaf_splits_->LeafIndex();
  this->best_split_per_leaf_[leaf] = smaller_bests_per_thread[smaller_best_idx];

  if (this->larger_leaf_splits_ != nullptr && this->larger_leaf_splits_->LeafIndex() >= 0) {
    leaf = this->larger_leaf_splits_->LeafIndex();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_best_per_thread);
    this->best_split_per_leaf_[leaf] = larger_best_per_thread[larger_best_idx];
  }

  // find local best
  SplitInfo smaller_best_split, larger_best_split;
  smaller_best_split = this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()];
  // find local best split for larger leaf
  if (this->larger_leaf_splits_->LeafIndex() >= 0) {
    larger_best_split = this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()];
  }
  // sync global best info
  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold);

  // copy back
  this->best_split_per_leaf_[smaller_leaf_splits_global_->LeafIndex()] = smaller_best_split;
  if (larger_best_split.feature >= 0 && larger_leaf_splits_global_->LeafIndex() >= 0) {
    this->best_split_per_leaf_[larger_leaf_splits_global_->LeafIndex()] = larger_best_split;
  }
}

template <typename TREELEARNER_T>
void VotingParallelTreeLearner<TREELEARNER_T>::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  TREELEARNER_T::Split(tree, best_Leaf, left_leaf, right_leaf);
  const SplitInfo& best_split_info = this->best_split_per_leaf_[best_Leaf];
  // set the global number of data for leaves
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
  auto p_left = smaller_leaf_splits_global_.get();
  auto p_right = larger_leaf_splits_global_.get();
  // init the global sumup info
  if (best_split_info.left_count < best_split_info.right_count) {
    smaller_leaf_splits_global_->Init(*left_leaf, this->data_partition_.get(),
      best_split_info.left_sum_gradient,
      best_split_info.left_sum_hessian);
    larger_leaf_splits_global_->Init(*right_leaf, this->data_partition_.get(),
      best_split_info.right_sum_gradient,
      best_split_info.right_sum_hessian);
  } else {
    smaller_leaf_splits_global_->Init(*right_leaf, this->data_partition_.get(),
      best_split_info.right_sum_gradient,
      best_split_info.right_sum_hessian);
    larger_leaf_splits_global_->Init(*left_leaf, this->data_partition_.get(),
      best_split_info.left_sum_gradient,
      best_split_info.left_sum_hessian);
    p_left = larger_leaf_splits_global_.get();
    p_right = smaller_leaf_splits_global_.get();
  }
  const int inner_feature_index = this->train_data_->InnerFeatureIndex(best_split_info.feature);
  bool is_numerical_split = this->train_data_->FeatureBinMapper(inner_feature_index)->bin_type() == BinType::NumericalBin;
  p_left->SetValueConstraint(best_split_info.min_constraint, best_split_info.max_constraint);
  p_right->SetValueConstraint(best_split_info.min_constraint, best_split_info.max_constraint);
  if (is_numerical_split) {
    double mid = (best_split_info.left_output + best_split_info.right_output) / 2.0f;
    if (best_split_info.monotone_type < 0) {
      p_left->SetValueConstraint(mid, best_split_info.max_constraint);
      p_right->SetValueConstraint(best_split_info.min_constraint, mid);
    } else if (best_split_info.monotone_type > 0) {
      p_left->SetValueConstraint(best_split_info.min_constraint, mid);
      p_right->SetValueConstraint(mid, best_split_info.max_constraint);
    }
  }
}

// instantiate template classes, otherwise linker cannot find the code
template class VotingParallelTreeLearner<GPUTreeLearner>;
template class VotingParallelTreeLearner<SerialTreeLearner>;
}  // namespace LightGBM
