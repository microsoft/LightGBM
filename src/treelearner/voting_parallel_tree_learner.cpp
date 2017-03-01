#include "parallel_tree_learner.h"

#include <LightGBM/utils/common.h>

#include <cstring>

#include <tuple>
#include <vector>

namespace LightGBM {

VotingParallelTreeLearner::VotingParallelTreeLearner(const TreeConfig* tree_config)
  :SerialTreeLearner(tree_config) {
  top_k_ = tree_config_->top_k;
}

void VotingParallelTreeLearner::Init(const Dataset* train_data) {
  SerialTreeLearner::Init(train_data);
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();

  // limit top k
  if (top_k_ > num_features_) {
    top_k_ = num_features_;
  }
  // get max bin
  int max_bin = 0;
  for (int i = 0; i < num_features_; ++i) {
    if (max_bin < train_data_->FeatureNumBin(i)) {
      max_bin = train_data_->FeatureNumBin(i);
    }
  }
  // calculate buffer size
  size_t buffer_size = 2 * top_k_ * std::max(max_bin * sizeof(HistogramBinEntry), sizeof(SplitInfo) * num_machines_);
  // left and right on same time, so need double size
  input_buffer_.resize(buffer_size);
  output_buffer_.resize(buffer_size);

  smaller_is_feature_aggregated_.resize(num_features_);
  larger_is_feature_aggregated_.resize(num_features_);

  block_start_.resize(num_machines_);
  block_len_.resize(num_machines_);

  smaller_buffer_read_start_pos_.resize(num_features_);
  larger_buffer_read_start_pos_.resize(num_features_);
  global_data_count_in_leaf_.resize(tree_config_->num_leaves);

  smaller_leaf_splits_global_.reset(new LeafSplits(train_data_->num_features(), train_data_->num_data()));
  larger_leaf_splits_global_.reset(new LeafSplits(train_data_->num_features(), train_data_->num_data()));

  local_tree_config_ = *tree_config_;
  local_tree_config_.min_data_in_leaf /= num_machines_;
  local_tree_config_.min_sum_hessian_in_leaf /= num_machines_;

  histogram_pool_.ResetConfig(&local_tree_config_);

  // initialize histograms for global
  smaller_leaf_histogram_array_global_.reset(new FeatureHistogram[num_features_]);
  larger_leaf_histogram_array_global_.reset(new FeatureHistogram[num_features_]);
  int num_total_bin = 0;
  for (int i = 0; i < num_features_; ++i) {
    num_total_bin += train_data_->FeatureNumBin(i);
  }
  smaller_leaf_histogram_data_.resize(num_total_bin);
  larger_leaf_histogram_data_.resize(num_total_bin);
  feature_metas_.resize(train_data->num_features());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < train_data->num_features(); ++i) {
    feature_metas_[i].feature_idx = i;
    feature_metas_[i].num_bin = train_data->FeatureNumBin(i);
    if (train_data->FeatureBinMapper(i)->GetDefaultBin() == 0) {
      feature_metas_[i].bias = 1;
    } else {
      feature_metas_[i].bias = 0;
    }
    feature_metas_[i].tree_config = tree_config_;
  }
  uint64_t offset = 0;
  for (int j = 0; j < train_data->num_features(); ++j) {
    offset += static_cast<uint64_t>(train_data->SubFeatureBinOffset(j));
    smaller_leaf_histogram_array_global_[j].Init(smaller_leaf_histogram_data_.data() + offset, &feature_metas_[j], train_data->FeatureBinMapper(j)->bin_type());
    larger_leaf_histogram_array_global_[j].Init(larger_leaf_histogram_data_.data() + offset, &feature_metas_[j], train_data->FeatureBinMapper(j)->bin_type());
    auto num_bin = train_data->FeatureNumBin(j);
    if (train_data->FeatureBinMapper(j)->GetDefaultBin() == 0) {
      num_bin -= 1;
    }
    offset += static_cast<uint64_t>(num_bin);
  }
}

void VotingParallelTreeLearner::ResetConfig(const TreeConfig* tree_config) {
  SerialTreeLearner::ResetConfig(tree_config);

  local_tree_config_ = *tree_config_;
  local_tree_config_.min_data_in_leaf /= num_machines_;
  local_tree_config_.min_sum_hessian_in_leaf /= num_machines_;

  histogram_pool_.ResetConfig(&local_tree_config_);
  global_data_count_in_leaf_.resize(tree_config_->num_leaves);

  for (size_t i = 0; i < feature_metas_.size(); ++i) {
    feature_metas_[i].tree_config = tree_config_;
  }
}

void VotingParallelTreeLearner::BeforeTrain() {
  SerialTreeLearner::BeforeTrain();
  // sync global data sumup info
  std::tuple<data_size_t, double, double> data(smaller_leaf_splits_->num_data_in_leaf(), smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians());
  int size = sizeof(std::tuple<data_size_t, double, double>);
  std::memcpy(input_buffer_.data(), &data, size);

  Network::Allreduce(input_buffer_.data(), size, size, output_buffer_.data(), [](const char *src, char *dst, int len) {
    int used_size = 0;
    int type_size = sizeof(std::tuple<data_size_t, double, double>);
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

  std::memcpy(&data, output_buffer_.data(), size);

  // set global sumup info
  smaller_leaf_splits_global_->Init(std::get<1>(data), std::get<2>(data));
  larger_leaf_splits_global_->Init();
  // init global data count in leaf
  global_data_count_in_leaf_[0] = std::get<0>(data);
}

bool VotingParallelTreeLearner::BeforeFindBestSplit(int left_leaf, int right_leaf) {
  if (SerialTreeLearner::BeforeFindBestSplit(left_leaf, right_leaf)) {
    data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
    data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
    if (right_leaf < 0) {
      return true;
    } else if (num_data_in_left_child < num_data_in_right_child) {
      // get local sumup
      smaller_leaf_splits_->Init(left_leaf, data_partition_.get(), gradients_, hessians_);
      larger_leaf_splits_->Init(right_leaf, data_partition_.get(), gradients_, hessians_);
    } else {
      // get local sumup
      smaller_leaf_splits_->Init(right_leaf, data_partition_.get(), gradients_, hessians_);
      larger_leaf_splits_->Init(left_leaf, data_partition_.get(), gradients_, hessians_);
    }
    return true;
  } else {
    return false;
  }
}

void VotingParallelTreeLearner::GlobalVoting(int leaf_idx, const std::vector<SplitInfo>& splits, std::vector<int>* out) {
  out->clear();
  if (leaf_idx < 0) {
    return;
  }
  // get mean number on machines
  score_t mean_num_data = GetGlobalDataCountInLeaf(leaf_idx) / static_cast<score_t>(num_machines_);
  std::vector<SplitInfo> feature_best_split(num_features_, SplitInfo());
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
  std::vector<SplitInfo> top_k_splits;
  ArrayArgs<SplitInfo>::MaxK(feature_best_split, top_k_, &top_k_splits);
  for (auto& split : top_k_splits) {
    if (split.gain == kMinScore || split.feature == -1) {
      continue;
    }
    out->push_back(split.feature);
  }
}

void VotingParallelTreeLearner::CopyLocalHistogram(const std::vector<int>& smaller_top_features, const std::vector<int>& larger_top_features) {
  for (int i = 0; i < num_features_; ++i) {
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
        int fid = smaller_top_features[smaller_idx];
        ++cur_used_features;
        // mark local aggregated feature
        if (i == rank_) {
          smaller_is_feature_aggregated_[fid] = true;
          smaller_buffer_read_start_pos_[fid] = static_cast<int>(cur_size);
        }
        // copy
        std::memcpy(input_buffer_.data() + reduce_scatter_size_, smaller_leaf_histogram_array_[fid].RawData(), smaller_leaf_histogram_array_[fid].SizeOfHistgram());
        cur_size += smaller_leaf_histogram_array_[fid].SizeOfHistgram();
        reduce_scatter_size_ += smaller_leaf_histogram_array_[fid].SizeOfHistgram();
        ++smaller_idx;
      }
      if (cur_used_features >= cur_total_feature) {
        break;
      }
      // then copy larger leaf histograms
      if (larger_idx < larger_top_features.size()) {
        int fid = larger_top_features[larger_idx];
        ++cur_used_features;
        // mark local aggregated feature
        if (i == rank_) {
          larger_is_feature_aggregated_[fid] = true;
          larger_buffer_read_start_pos_[fid] = static_cast<int>(cur_size);
        }
        // copy
        std::memcpy(input_buffer_.data() + reduce_scatter_size_, larger_leaf_histogram_array_[fid].RawData(), larger_leaf_histogram_array_[fid].SizeOfHistgram());
        cur_size += larger_leaf_histogram_array_[fid].SizeOfHistgram();
        reduce_scatter_size_ += larger_leaf_histogram_array_[fid].SizeOfHistgram();
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

void VotingParallelTreeLearner::FindBestThresholds() {
  // use local data to find local best splits
  std::vector<int8_t> is_feature_used(num_features_, 0);
#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used_[feature_index]) continue;
    if (parent_leaf_histogram_array_ != nullptr
      && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    is_feature_used[feature_index] = 1;
  }
  bool use_subtract = true;
  if (parent_leaf_histogram_array_ == nullptr) {
    use_subtract = false;
  }
  // construct smaller leaf
  HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
  train_data_->ConstructHistograms(is_feature_used,
    smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
    smaller_leaf_splits_->LeafIndex(),
    ordered_bins_, gradients_, hessians_,
    ordered_gradients_.data(), ordered_hessians_.data(),
    ptr_smaller_leaf_hist_data);

  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    // construct larger leaf
    HistogramBinEntry* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - 1;
    train_data_->ConstructHistograms(is_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
      larger_leaf_splits_->LeafIndex(),
      ordered_bins_, gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data(),
      ptr_larger_leaf_hist_data);
  }

  std::vector<SplitInfo> smaller_bestsplit_per_features(num_features_);
  std::vector<SplitInfo> larger_bestsplit_per_features(num_features_);

  // find splits
#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used[feature_index]) { continue; }
    train_data_->FixHistogram(feature_index,
      smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
      smaller_leaf_splits_->num_data_in_leaf(),
      smaller_leaf_histogram_array_[feature_index].RawData());

    smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
      smaller_leaf_splits_->sum_gradients(),
      smaller_leaf_splits_->sum_hessians(),
      smaller_leaf_splits_->num_data_in_leaf(),
      &smaller_bestsplit_per_features[feature_index]);
    // only has root leaf
    if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->LeafIndex() < 0) { continue; }

    if (use_subtract) {
      larger_leaf_histogram_array_[feature_index].Subtract(smaller_leaf_histogram_array_[feature_index]);
    } else {
      train_data_->FixHistogram(feature_index, larger_leaf_splits_->sum_gradients(), larger_leaf_splits_->sum_hessians(),
        larger_leaf_splits_->num_data_in_leaf(),
        larger_leaf_histogram_array_[feature_index].RawData());
    }
    // find best threshold for larger child
    larger_leaf_histogram_array_[feature_index].FindBestThreshold(
      larger_leaf_splits_->sum_gradients(),
      larger_leaf_splits_->sum_hessians(),
      larger_leaf_splits_->num_data_in_leaf(),
      &larger_bestsplit_per_features[feature_index]);
  }

  std::vector<SplitInfo> smaller_top_k_splits, larger_top_k_splits;
  // local voting
  ArrayArgs<SplitInfo>::MaxK(smaller_bestsplit_per_features, top_k_, &smaller_top_k_splits);
  ArrayArgs<SplitInfo>::MaxK(larger_bestsplit_per_features, top_k_, &larger_top_k_splits);
  // gather
  int offset = 0;
  for (int i = 0; i < top_k_; ++i) {
    std::memcpy(input_buffer_.data() + offset, &smaller_top_k_splits[i], sizeof(SplitInfo));
    offset += sizeof(SplitInfo);
    std::memcpy(input_buffer_.data() + offset, &larger_top_k_splits[i], sizeof(SplitInfo));
    offset += sizeof(SplitInfo);
  }
  Network::Allgather(input_buffer_.data(), offset, output_buffer_.data());
  // get all top-k from all machines
  std::vector<SplitInfo> smaller_top_k_splits_global;
  std::vector<SplitInfo> larger_top_k_splits_global;
  offset = 0;
  for (int i = 0; i < num_machines_; ++i) {
    for (int j = 0; j < top_k_; ++j) {
      smaller_top_k_splits_global.push_back(SplitInfo());
      std::memcpy(&smaller_top_k_splits_global.back(), output_buffer_.data() + offset, sizeof(SplitInfo));
      offset += sizeof(SplitInfo);
      larger_top_k_splits_global.push_back(SplitInfo());
      std::memcpy(&larger_top_k_splits_global.back(), output_buffer_.data() + offset, sizeof(SplitInfo));
      offset += sizeof(SplitInfo);
    }
  }
  // global voting
  std::vector<int> smaller_top_features, larger_top_features;
  GlobalVoting(smaller_leaf_splits_->LeafIndex(), smaller_top_k_splits_global, &smaller_top_features);
  GlobalVoting(larger_leaf_splits_->LeafIndex(), larger_top_k_splits_global, &larger_top_features);
  // copy local histgrams to buffer
  CopyLocalHistogram(smaller_top_features, larger_top_features);

  // Reduce scatter for histogram
  Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, block_start_.data(), block_len_.data(),
    output_buffer_.data(), &HistogramBinEntry::SumReducer);

  std::vector<SplitInfo> smaller_best(num_threads_);
  std::vector<SplitInfo> larger_best(num_threads_);
  // find best split from local aggregated histograms
#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    const int tid = omp_get_thread_num();
    if (smaller_is_feature_aggregated_[feature_index]) {
      SplitInfo smaller_split;
      // restore from buffer
      smaller_leaf_histogram_array_global_[feature_index].FromMemory(
        output_buffer_.data() + smaller_buffer_read_start_pos_[feature_index]);
      // find best threshold
      smaller_leaf_histogram_array_global_[feature_index].FindBestThreshold(
        smaller_leaf_splits_global_->sum_gradients(),
        smaller_leaf_splits_global_->sum_hessians(),
        GetGlobalDataCountInLeaf(smaller_leaf_splits_global_->LeafIndex()),
        &smaller_split);
      if (smaller_split.gain > smaller_best[tid].gain) {
        smaller_best[tid] = smaller_split;
      }
    }

    if (larger_is_feature_aggregated_[feature_index]) {
      SplitInfo larger_split;
      // restore from buffer
      larger_leaf_histogram_array_global_[feature_index].FromMemory(output_buffer_.data() + larger_buffer_read_start_pos_[feature_index]);
      // find best threshold
      larger_leaf_histogram_array_global_[feature_index].FindBestThreshold(
        larger_leaf_splits_global_->sum_gradients(),
        larger_leaf_splits_global_->sum_hessians(),
        GetGlobalDataCountInLeaf(larger_leaf_splits_global_->LeafIndex()),
        &larger_split);
      if (larger_split.gain > larger_best[tid].gain) {
        larger_best[tid] = larger_split;
      }
    }
  }
  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_best);
  int leaf = smaller_leaf_splits_->LeafIndex();
  best_split_per_leaf_[leaf] = smaller_best[smaller_best_idx];

  if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->LeafIndex() >= 0) {
    leaf = larger_leaf_splits_->LeafIndex();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_best);
    best_split_per_leaf_[leaf] = larger_best[larger_best_idx];
  }

}

void VotingParallelTreeLearner::FindBestSplitsForLeaves() {
  // find local best
  SplitInfo smaller_best, larger_best;
  smaller_best = best_split_per_leaf_[smaller_leaf_splits_->LeafIndex()];
  // find local best split for larger leaf
  if (larger_leaf_splits_->LeafIndex() >= 0) {
    larger_best = best_split_per_leaf_[larger_leaf_splits_->LeafIndex()];
  }
  // sync global best info
  std::memcpy(input_buffer_.data(), &smaller_best, sizeof(SplitInfo));
  std::memcpy(input_buffer_.data() + sizeof(SplitInfo), &larger_best, sizeof(SplitInfo));

  Network::Allreduce(input_buffer_.data(), sizeof(SplitInfo) * 2, sizeof(SplitInfo), output_buffer_.data(), &SplitInfo::MaxReducer);

  std::memcpy(&smaller_best, output_buffer_.data(), sizeof(SplitInfo));
  std::memcpy(&larger_best, output_buffer_.data() + sizeof(SplitInfo), sizeof(SplitInfo));

  // copy back
  best_split_per_leaf_[smaller_leaf_splits_global_->LeafIndex()] = smaller_best;
  if (larger_best.feature >= 0 && larger_leaf_splits_global_->LeafIndex() >= 0) {
    best_split_per_leaf_[larger_leaf_splits_global_->LeafIndex()] = larger_best;
  }
}

void VotingParallelTreeLearner::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  SerialTreeLearner::Split(tree, best_Leaf, left_leaf, right_leaf);
  const SplitInfo& best_split_info = best_split_per_leaf_[best_Leaf];
  // set the global number of data for leaves
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
  // init the global sumup info
  if (best_split_info.left_count < best_split_info.right_count) {
    smaller_leaf_splits_global_->Init(*left_leaf, data_partition_.get(),
      best_split_info.left_sum_gradient,
      best_split_info.left_sum_hessian);
    larger_leaf_splits_global_->Init(*right_leaf, data_partition_.get(),
      best_split_info.right_sum_gradient,
      best_split_info.right_sum_hessian);
  } else {
    smaller_leaf_splits_global_->Init(*right_leaf, data_partition_.get(),
      best_split_info.right_sum_gradient,
      best_split_info.right_sum_hessian);
    larger_leaf_splits_global_->Init(*left_leaf, data_partition_.get(),
      best_split_info.left_sum_gradient,
      best_split_info.left_sum_hessian);
  }
}

}  // namespace FTLBoost
