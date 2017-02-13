#include "gpu_tree_learner.h"
#include "../io/dense_bin.hpp"

#include <LightGBM/utils/array_args.h>
#include <LightGBM/bin.h>

#include <algorithm>
#include <vector>

namespace LightGBM {

GPUTreeLearner::GPUTreeLearner(const TreeConfig* tree_config)
  :tree_config_(tree_config){
  random_ = Random(tree_config_->feature_fraction_seed);
  Log::Info("This is the GPU trainer!!");

}

GPUTreeLearner::~GPUTreeLearner() {

}

void GPUTreeLearner::Init(const Dataset* train_data) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  num_features_ = train_data_->num_features();
  int max_cache_size = 0;
  // Get the max size of pool
  if (tree_config_->histogram_pool_size <= 0) {
    max_cache_size = tree_config_->num_leaves;
  } else {
    size_t total_histogram_size = 0;
    for (int i = 0; i < train_data_->num_features(); ++i) {
      total_histogram_size += sizeof(HistogramBinEntry) * train_data_->FeatureAt(i)->num_bin();
    }
    max_cache_size = static_cast<int>(tree_config_->histogram_pool_size * 1024 * 1024 / total_histogram_size);
  }
  // at least need 2 leaves
  max_cache_size = std::max(2, max_cache_size);
  max_cache_size = std::min(max_cache_size, tree_config_->num_leaves);
  histogram_pool_.Reset(max_cache_size, tree_config_->num_leaves);

  auto histogram_create_function = [this]() {
    auto tmp_histogram_array = std::unique_ptr<FeatureHistogram[]>(new FeatureHistogram[train_data_->num_features()]);
    for (int j = 0; j < train_data_->num_features(); ++j) {
      tmp_histogram_array[j].Init(train_data_->FeatureAt(j),
        j, tree_config_);
    }
    return tmp_histogram_array.release();
  };
  histogram_pool_.Fill(histogram_create_function);

  // push split information for all leaves
  best_split_per_leaf_.resize(tree_config_->num_leaves);
  // initialize ordered_bins_ with nullptr
  ordered_bins_.resize(num_features_);

  // get ordered bin
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_features_; ++i) {
    ordered_bins_[i].reset(train_data_->FeatureAt(i)->bin_data()->CreateOrderedBin());
  }

  // check existing for ordered bin
  for (int i = 0; i < num_features_; ++i) {
    if (ordered_bins_[i] != nullptr) {
      has_ordered_bin_ = true;
      break;
    }
  }
  // initialize splits for leaf
  smaller_leaf_splits_.reset(new LeafSplits(train_data_->num_features(), train_data_->num_data()));
  larger_leaf_splits_.reset(new LeafSplits(train_data_->num_features(), train_data_->num_data()));

  // initialize data partition
  data_partition_.reset(new DataPartition(num_data_, tree_config_->num_leaves));

  is_feature_used_.resize(num_features_);

  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);
  // if has ordered bin, need to allocate a buffer to fast split
  if (has_ordered_bin_) {
    is_data_in_leaf_.resize(num_data_);
  }
  Log::Info("Number of data: %d, number of features: %d", num_data_, num_features_);
  InitGPU(-1, -1);
}

void PrintHistograms(HistogramBinEntry* h, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("[%3lu]=%9.3g,%9.3g,%8d\t", i, h[i].sum_gradients, h[i].sum_hessians, h[i].cnt);
    if ((i & 3) == 3)
        printf("\n");
  }
  printf("\n");
}

union Float_t
{
    int32_t i;
    float f;
    static int32_t ulp_diff(Float_t a, Float_t b) {
      return abs(a.i - b.i);
    }
};
  

void CompareHistograms(HistogramBinEntry* h1, HistogramBinEntry* h2, size_t size) {
  size_t i;
  Float_t a, b;
  for (i = 0; i < size; ++i) {
    a.f = h1[i].sum_gradients;
    b.f = h2[i].sum_gradients;
    int32_t ulps = Float_t::ulp_diff(a, b);
    if (ulps > 65536) {
      printf("%g != %g (%d ULPs)\n", h1[i].sum_gradients, h2[i].sum_gradients, ulps);
      goto err;
    }
    a.f = h1[i].sum_hessians;
    b.f = h2[i].sum_hessians;
    ulps = Float_t::ulp_diff(a, b);
    if (ulps > 65536) {
      printf("%g != %g (%d ULPs)\n", h1[i].sum_hessians, h2[i].sum_hessians, ulps);
      goto err;
    }
    if (fabs(h1[i].cnt           - h2[i].cnt != 0)) {
      printf("%d != %d\n", h1[i].cnt, h2[i].cnt);
      goto err;
    }
  }
  return;
err:
  PrintHistograms(h1, size);
  PrintHistograms(h2, size);
  Log::Fatal("Mismatched histograms found at location %lu.", i);
}

int GPUTreeLearner::GetNumWorkgroupsPerFeature(data_size_t leaf_num_data) {
  // we roughly want 256 workgroups per device, and we have num_dense_feature4_ feature tuples.
  // also guarantee that there are at least 2K examples per workgroup
  double x = 256.0 / num_dense_feature4_;
  int exp_workgroups_per_feature = ceil(log(x)/log(2.0));
  double t = leaf_num_data / 1024.0;
  #ifdef DEBUG_GPU
  printf("Computing histogram for %d examples and (4 * %d) features\n", leaf_num_data, num_dense_feature4_);
  printf("We can have at most %d workgroups per feature4 for efficiency reasons.\n"
         "Best workgroup size per feature for full utilization is %d\n", (int)ceil(t), (1 << exp_workgroups_per_feature));
  #endif
  exp_workgroups_per_feature = std::min(exp_workgroups_per_feature, (int)ceil(log((double)t)/log(2.0)));
  if (exp_workgroups_per_feature < 0)
      exp_workgroups_per_feature = 0;
  if (exp_workgroups_per_feature > max_exp_workgroups_per_feature_)
      exp_workgroups_per_feature = max_exp_workgroups_per_feature_;
  return exp_workgroups_per_feature;
}

void GPUTreeLearner::GPUHistogram(data_size_t leaf_num_data, FeatureHistogram* histograms) {
  // we have already copied ordered gradients, ordered hessians and indices to GPU
  // decide the best number of workgroups working on one feature4 tuple
  // set work group size based on feature size
  // each 2^exp_workgroups_per_feature workgroups work on a feature4 tuple
  int exp_workgroups_per_feature = GetNumWorkgroupsPerFeature(leaf_num_data);
  int num_workgroups = (1 << exp_workgroups_per_feature) * num_dense_feature4_;
  if (num_workgroups > max_num_workgroups_) {
    num_workgroups = max_num_workgroups_;
    Log::Warning("BUG detected, num_workgroups too large!");
  }
  #ifdef DEBUG_GPU
  printf("setting exp_workgroups_per_feature to %d, using %u work groups\n", exp_workgroups_per_feature, num_workgroups);
  #endif
  // std::cin.get();
  
  // the kernel will process all features, and each
  // 2^exp_workgroups_per_feature (compile time constant) workgroup will
  // process one feature4 tuple

  histogram_kernels_[exp_workgroups_per_feature].set_arg(2, leaf_num_data);
  // for the root node, indices are not copied
  if (leaf_num_data != num_data_) {
    indices_future_.wait();
  }
  hessians_future_.wait();
  gradients_future_.wait();
  // printf("launching kernel!\n");
  // there will be 2^exp_workgroups_per_feature = num_workgroups / num_dense_feature4 sub-histogram per feature4
  // and we will launch num_feature workgroups for this kernel
  // will launch threads for all features
  if (leaf_num_data == num_data_) {
    // printf("using full data kernel with exp_workgroups_per_feature = %d and %d workgroups\n", exp_workgroups_per_feature, num_workgroups);
    queue_.enqueue_1d_range_kernel(histogram_fulldata_kernel_, 0, num_workgroups * 256, 256);
  }
  else {
    queue_.enqueue_1d_range_kernel(histogram_kernels_[exp_workgroups_per_feature], 0, num_workgroups * 256, 256);
  }
  queue_.finish();
  // all features finished, copy results to out
  // printf("Copying histogram back to host...\n");
  boost::compute::copy(device_histogram_outputs_->begin(), device_histogram_outputs_->end(), (char*)host_histogram_outputs_.get(), queue_);
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < num_dense_features_; ++i) {
    int dense_index = dense_feature_map_[i];
    auto old_histogram_array = histograms[dense_index].GetData();
    int bin_size = histograms[dense_index].SizeOfHistgram() / sizeof(HistogramBinEntry);
    // printf("copying dense feature %d (index %d) size %d\n", i, dense_index, bin_size);
    // histogram size can be smaller than 255 (not a fixed number for each feature)
    // but the GPU code can only handle up to 256
    for (int j = 0; j < bin_size; ++j) {
      // printf("%f\n", host_histogram_outputs_[i * 256 + j].sum_gradients);
      old_histogram_array[j].sum_gradients = host_histogram_outputs_[i * 256 + j].sum_gradients;
      old_histogram_array[j].sum_hessians = host_histogram_outputs_[i * 256 + j].sum_hessians;
      old_histogram_array[j].cnt = host_histogram_outputs_[i * 256 + j].cnt;
    }
    // PrintHistograms(old_histogram_array, bin_size);
  }
}

// Copy data to GPU
void GPUTreeLearner::InitGPU(int platform_id, int device_id) {
  num_dense_features_ = 0;
  for (int i = 0; i < num_features_; ++i) {
    if (ordered_bins_[i] == nullptr) {
      printf("feature %d is dense\n", i);
      num_dense_features_++;
    }
  }
  // how many 4-feature tuples we have
  num_dense_feature4_ = (num_dense_features_ + 3) / 4;
  // leave some safe margin for prefetching
  int allocated_num_data_ = num_data_ + 256 * (1 << max_exp_workgroups_per_feature_);
  // currently we don't use constant memory
  int use_constants = 0;
  // printf("%d %d %d\n", num_dense_features_, num_dense_feature4_, allocated_num_data_);

  // initialize GPU
  dev_ = boost::compute::system::default_device();
  if (platform_id >= 0 && device_id >= 0) {
    const std::vector<boost::compute::platform> platforms = boost::compute::system::platforms();
    if ((int)platforms.size() > platform_id) {
      const std::vector<boost::compute::device> platform_devices = platforms[platform_id].devices();
      if ((int)platform_devices.size() > device_id) {
        std::cout << "Using requested OpenCL platform " << platform_id << " device " << device_id << std::endl;
        dev_ = platform_devices[device_id];
      }   
    }   
  }   
  ctx_ = boost::compute::context(dev_);
  queue_ = boost::compute::command_queue(ctx_, dev_);
  // allocate memory for all features (FIXME: 4 GB barrier)
  device_features_ = std::unique_ptr<boost::compute::vector<Feature4>>(new boost::compute::vector<Feature4>(num_dense_feature4_ * num_data_, ctx_));
  // allocate space for gradients and hessians on device
  // we will copy gradients and hessians in after ordered_gradients_ and ordered_hessians_ are constructed
  device_gradients_ = std::unique_ptr<boost::compute::vector<score_t>>(new boost::compute::vector<score_t>(allocated_num_data_, ctx_));
  device_hessians_ = std::unique_ptr<boost::compute::vector<score_t>>(new boost::compute::vector<score_t>(allocated_num_data_, ctx_));
  // copy indices to the device
  device_data_indices_ = std::unique_ptr<boost::compute::vector<data_size_t>>(new boost::compute::vector<data_size_t>(allocated_num_data_, ctx_));
  boost::compute::fill(device_data_indices_->begin(), device_data_indices_->end(), 0, queue_);
  // create output buffer, each feature has a histogram with 256 bins,
  // each work group generates a sub-histogram of 4 features.
  device_subhistograms_ = std::unique_ptr<boost::compute::vector<char>>(new boost::compute::vector<char>(
                          max_num_workgroups_ * 4 * 256 * sizeof(GPUHistogramBinEntry), ctx_));
  // create atomic counters for inter-group synchronization
  sync_counters_ = std::unique_ptr<boost::compute::vector<int>>(new boost::compute::vector<int>(
                    num_dense_feature4_, ctx_));
  boost::compute::fill(sync_counters_->begin(), sync_counters_->end(), 0, queue_);
  // FIXME: bin size 256 fixed
  device_histogram_outputs_ = std::unique_ptr<boost::compute::vector<char>>(new boost::compute::vector<char>(
                    num_dense_feature4_ * 4 * 256 * sizeof(GPUHistogramBinEntry), ctx_));
  // create OpenCL kernels for different number of workgroups per feature
  Log::Info("Using GPU Device: %s, Vendor: %s", dev_.name().c_str(), dev_.vendor().c_str());
  Log::Info("Compiling OpenCL Kernels...");
  for (int i = 0; i <= max_exp_workgroups_per_feature_; ++i) {
    auto program = boost::compute::program::create_with_source_file("histogram.cl", ctx_);
    std::ostringstream opts;
    // FIXME: sparse data
    opts << "-D FEATURE_SIZE=" << num_data_ << " -D POWER_FEATURE_WORKGROUPS=" << i
         << " -D USE_CONSTANT_BUF=" << use_constants 
         << " -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -save-temps";
    std::cout << "Building options: " << opts.str() << std::endl;
    try {
      program.build(opts.str());
    }
    catch (boost::compute::opencl_error &e) {
      Log::Fatal("GPU program built failure:\n %s", program.build_log().c_str());
    }
    histogram_kernels_.push_back(program.create_kernel("histogram256"));
    // setup kernel arguments
    // The only argument that needs to be changed is num_data_
    histogram_kernels_.back().set_args(*device_features_,
    *device_data_indices_, num_data_, *device_gradients_, *device_hessians_,
    *device_subhistograms_, *sync_counters_, *device_histogram_outputs_);
  }
  // create the OpenCL kernel for the root node (all data)
  int full_exp_workgroups_per_feature = GetNumWorkgroupsPerFeature(num_data_);
  auto program = boost::compute::program::create_with_source_file("histogram.cl", ctx_);
  std::ostringstream opts;
  // FIXME: sparse data
  opts << "-D FEATURE_SIZE=" << num_data_ << " -D POWER_FEATURE_WORKGROUPS=" << full_exp_workgroups_per_feature
       << " -D IGNORE_INDICES=1" 
       << " -D USE_CONSTANT_BUF=" << use_constants 
       << " -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -save-temps";
  std::cout << "Building options: " << opts.str() << std::endl;
  try {
    program.build(opts.str());
  }
  catch (boost::compute::opencl_error &e) {
    Log::Fatal("GPU program built failure:\n %s", program.build_log().c_str());
  }
  histogram_fulldata_kernel_ = program.create_kernel("histogram256");
  // setup kernel arguments
  // The only argument that needs to be changed is num_data_
  histogram_fulldata_kernel_.set_args(*device_features_,
  *device_data_indices_, num_data_, *device_gradients_, *device_hessians_,
  *device_subhistograms_, *sync_counters_, *device_histogram_outputs_);

  // Now generate new data structure feature4, and copy data to the device
  int i, k, copied_feature4 = 0, dense_ind[4];
  for (i = 0, k = 0; i < num_features_; ++i) {
    // looking for 4 non-sparse features
    if (ordered_bins_[i] == nullptr) {
      dense_ind[k++] = i;
    }
    // found 
    if (k == 4) {
      k = 0;
      std::vector<Feature4> host4(num_data_);
      printf("Copying feature %d, %d, %d, %d to device\n", dense_ind[0], dense_ind[1], dense_ind[2], dense_ind[3]);
      printf("feature size: %d, %d, %d, %d, %d\n", 
      static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[0])->bin_data())->num_data(),
      static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[1])->bin_data())->num_data(),
      static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[2])->bin_data())->num_data(),
      static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[3])->bin_data())->num_data(),
      num_data_);
      for (int j = 0; j < num_data_; ++j) {
          host4[j].s0 = static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[0])->bin_data())->Get(j);
          host4[j].s1 = static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[1])->bin_data())->Get(j); 
          host4[j].s2 = static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[2])->bin_data())->Get(j);
          host4[j].s3 = static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[3])->bin_data())->Get(j);
      }
      printf("first example of features are: %d %d %d %d\n", host4[0].s0, host4[0].s1, host4[0].s2, host4[0].s3);
      boost::compute::copy(host4.begin(), host4.end(), device_features_->begin() + copied_feature4 * num_data_, queue_);
      printf("Feature %d, %d, %d, %d copied to device\n", dense_ind[0], dense_ind[1], dense_ind[2], dense_ind[3]);
      dense_feature_map_.push_back(dense_ind[0]);
      dense_feature_map_.push_back(dense_ind[1]);
      dense_feature_map_.push_back(dense_ind[2]);
      dense_feature_map_.push_back(dense_ind[3]);
      copied_feature4++;
    }
  }
  if (k != 0) {
    std::vector<Feature4> host4(num_data_);
    printf("%d features left\n", k);
    for (int j = 0; j < num_data_; ++j) {
      for (i = 0; i < k; ++i) {
        host4[j].s[i] = static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[i])->bin_data())->Get(j);
      }
      for (i = k; i < 4; ++i) {
        // fill this empty feature to some "random" value
        host4[j].s[i] = j;
      }
    }
    // copying the last 1-3 features
    boost::compute::copy(host4.begin(), host4.end(), device_features_->begin() + (num_dense_feature4_ - 1) * num_data_, queue_);
    printf("Last features copied to device\n");
    for (i = 0; i < k; ++i) {
      dense_feature_map_.push_back(dense_ind[i]);
    }
  }
  printf("Dense feature list (size %lu): ", dense_feature_map_.size());
  for (i = 0; i < num_dense_features_; ++i) {
    printf("%d ", dense_feature_map_[i]);
  }
  printf("\n");

  // host memory for transferring histograms
  host_histogram_outputs_ = std::unique_ptr<GPUHistogramBinEntry[]>(new GPUHistogramBinEntry[256 * num_dense_feature4_ * 4]());
}

void GPUTreeLearner::ResetTrainingData(const Dataset* train_data) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  num_features_ = train_data_->num_features();

  // initialize ordered_bins_ with nullptr
  ordered_bins_.resize(num_features_);

  // get ordered bin
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_features_; ++i) {
    ordered_bins_[i].reset(train_data_->FeatureAt(i)->bin_data()->CreateOrderedBin());
  }
  has_ordered_bin_ = false;
  // check existing for ordered bin
  for (int i = 0; i < num_features_; ++i) {
    if (ordered_bins_[i] != nullptr) {
      has_ordered_bin_ = true;
      break;
    }
  }
  // initialize splits for leaf
  smaller_leaf_splits_->ResetNumData(num_data_);
  larger_leaf_splits_->ResetNumData(num_data_);

  // initialize data partition
  data_partition_->ResetNumData(num_data_);

  is_feature_used_.resize(num_features_);

  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);
  // if has ordered bin, need to allocate a buffer to fast split
  if (has_ordered_bin_) {
    is_data_in_leaf_.resize(num_data_);
  }

}

void GPUTreeLearner::ResetConfig(const TreeConfig* tree_config) {
  if (tree_config_->num_leaves != tree_config->num_leaves) {
    tree_config_ = tree_config;
    int max_cache_size = 0;
    // Get the max size of pool
    if (tree_config->histogram_pool_size <= 0) {
      max_cache_size = tree_config_->num_leaves;
    } else {
      size_t total_histogram_size = 0;
      for (int i = 0; i < train_data_->num_features(); ++i) {
        total_histogram_size += sizeof(HistogramBinEntry) * train_data_->FeatureAt(i)->num_bin();
      }
      max_cache_size = static_cast<int>(tree_config_->histogram_pool_size * 1024 * 1024 / total_histogram_size);
    }
    // at least need 2 leaves
    max_cache_size = std::max(2, max_cache_size);
    max_cache_size = std::min(max_cache_size, tree_config_->num_leaves);
    histogram_pool_.DynamicChangeSize(max_cache_size, tree_config_->num_leaves);

    // push split information for all leaves
    best_split_per_leaf_.resize(tree_config_->num_leaves);
    data_partition_->ResetLeaves(tree_config_->num_leaves);
  } else {
    tree_config_ = tree_config;
  }

  histogram_pool_.ResetConfig(tree_config_, train_data_->num_features());
}


// train calls BeforeFindBestSplit()
// then FindThreesholds() for going over all features and building the histogram, and find the split for each feature
// then FindBestSplitsForLeaves() for finding the best feature
Tree* GPUTreeLearner::Train(const score_t* gradients, const score_t *hessians) {
  gradients_ = gradients;
  hessians_ = hessians;
  // some initial works before training
  BeforeTrain();
  auto tree = std::unique_ptr<Tree>(new Tree(tree_config_->num_leaves));
  // save pointer to last trained tree
  last_trained_tree_ = tree.get();
  // root leaf
  int left_leaf = 0;
  // only root leaf can be splitted on first time
  int right_leaf = -1;
  for (int split = 0; split < tree_config_->num_leaves - 1; split++) {
    // some initial works before finding best split
    if (BeforeFindBestSplit(left_leaf, right_leaf)) {
      // find best threshold for every feature
      // Go over all features to compute the histogram and find the best split for each feature
      FindBestThresholds();
      // find best split from all features
      FindBestSplitsForLeaves();
    }
    // Get a leaf with max split gain
    int best_leaf = static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_));
    // Get split information for best leaf
    const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];
    // cannot split, quit
    if (best_leaf_SplitInfo.gain <= 0.0) {
      Log::Info("No further splits with positive gain, best gain: %f, leaves: %d",
                   best_leaf_SplitInfo.gain, split + 1);
      break;
    }
    // split tree with best leaf
    Split(tree.get(), best_leaf, &left_leaf, &right_leaf);
  }
  return tree.release();
}

void GPUTreeLearner::BeforeTrain() {

  // copy indices, gradients and hessians to device, start as early as possible
  #ifdef GPU_DEBUG
  printf("Copying intial full gradients and hessians to device\n");
  #endif
  hessians_future_ = boost::compute::copy_async(hessians_,  hessians_  + num_data_, device_hessians_->begin(),  queue_);
  gradients_future_ = boost::compute::copy_async(gradients_, gradients_ + num_data_, device_gradients_->begin(), queue_);

  // reset histogram pool
  histogram_pool_.ResetMap();
  // initialize used features
  for (int i = 0; i < num_features_; ++i) {
    is_feature_used_[i] = false;
  }
  // Get used feature at current tree
  int used_feature_cnt = static_cast<int>(num_features_*tree_config_->feature_fraction);
  auto used_feature_indices = random_.Sample(num_features_, used_feature_cnt);
  for (auto idx : used_feature_indices) {
    is_feature_used_[idx] = true;
  }

  // initialize data partition
  data_partition_->Init();

  // reset the splits for leaves
  for (int i = 0; i < tree_config_->num_leaves; ++i) {
    best_split_per_leaf_[i].Reset();
  }

  // Sumup for root
  if (data_partition_->leaf_count(0) == num_data_) {
    // No need to copy the index for the root
    // indices_future_ = boost::compute::copy_async(data_partition_->indices(), data_partition_->indices() + num_data_, 
    //                     device_data_indices_->begin(), queue_);
    // use all data
    smaller_leaf_splits_->Init(gradients_, hessians_);
    // point to gradients, avoid copy
    ptr_to_ordered_gradients_smaller_leaf_ = gradients_;
    ptr_to_ordered_hessians_smaller_leaf_  = hessians_;
  } else {
    // use bagging, only use part of data
    smaller_leaf_splits_->Init(0, data_partition_.get(), gradients_, hessians_);
    // copy used gradients and hessians to ordered buffer
    const data_size_t* indices = data_partition_->indices();
    data_size_t cnt = data_partition_->leaf_count(0);
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < cnt; ++i) {
      ordered_gradients_[i] = gradients_[indices[i]];
      ordered_hessians_[i] = hessians_[indices[i]];
    }
    // point to ordered_gradients_ and ordered_hessians_
    ptr_to_ordered_gradients_smaller_leaf_ = ordered_gradients_.data();
    ptr_to_ordered_hessians_smaller_leaf_ = ordered_hessians_.data();
  }

  ptr_to_ordered_gradients_larger_leaf_ = nullptr;
  ptr_to_ordered_hessians_larger_leaf_ = nullptr;

  larger_leaf_splits_->Init();

  // if has ordered bin, need to initialize the ordered bin
  if (has_ordered_bin_) {
    if (data_partition_->leaf_count(0) == num_data_) {
      // use all data, pass nullptr
      #pragma omp parallel for schedule(guided)
      for (int i = 0; i < num_features_; ++i) {
        if (ordered_bins_[i] != nullptr) {
          ordered_bins_[i]->Init(nullptr, tree_config_->num_leaves);
        }
      }
    } else {
      // bagging, only use part of data

      // mark used data
      std::memset(is_data_in_leaf_.data(), 0, sizeof(char)*num_data_);
      const data_size_t* indices = data_partition_->indices();
      data_size_t begin = data_partition_->leaf_begin(0);
      data_size_t end = begin + data_partition_->leaf_count(0);
      #pragma omp parallel for schedule(static)
      for (data_size_t i = begin; i < end; ++i) {
        is_data_in_leaf_[indices[i]] = 1;
      }
      // initialize ordered bin
      #pragma omp parallel for schedule(guided)
      for (int i = 0; i < num_features_; ++i) {
        if (ordered_bins_[i] != nullptr) {
          ordered_bins_[i]->Init(is_data_in_leaf_.data(), tree_config_->num_leaves);
        }
      }
    }
  }
}

bool GPUTreeLearner::BeforeFindBestSplit(int left_leaf, int right_leaf) {
  // check depth of current leaf
  if (tree_config_->max_depth > 0) {
    // only need to check left leaf, since right leaf is in same level of left leaf
    if (last_trained_tree_->leaf_depth(left_leaf) >= tree_config_->max_depth) {
      best_split_per_leaf_[left_leaf].gain = kMinScore;
      if (right_leaf >= 0) {
        best_split_per_leaf_[right_leaf].gain = kMinScore;
      }
      return false;
    }
  }
  data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
  data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
  // no enough data to continue
  if (num_data_in_right_child < static_cast<data_size_t>(tree_config_->min_data_in_leaf * 2)
    && num_data_in_left_child < static_cast<data_size_t>(tree_config_->min_data_in_leaf * 2)) {
    best_split_per_leaf_[left_leaf].gain = kMinScore;
    if (right_leaf >= 0) {
      best_split_per_leaf_[right_leaf].gain = kMinScore;
    }
    return false;
  }
  parent_leaf_histogram_array_ = nullptr;
  // -1 if only has one leaf. else equal the index of smaller leaf
  int smaller_leaf = -1;
  int larger_leaf = -1;
  // only have root
  if (right_leaf < 0) {
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
    larger_leaf_histogram_array_ = nullptr;

  } else if (num_data_in_left_child < num_data_in_right_child) {
    smaller_leaf = left_leaf;
    larger_leaf = right_leaf;
    // put parent(left) leaf's histograms into larger leaf's histograms
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { parent_leaf_histogram_array_ = larger_leaf_histogram_array_; }
    histogram_pool_.Move(left_leaf, right_leaf);
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
  } else {
    smaller_leaf = right_leaf;
    larger_leaf = left_leaf;
    // put parent(left) leaf's histograms to larger leaf's histograms
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { parent_leaf_histogram_array_ = larger_leaf_histogram_array_; }
    histogram_pool_.Get(right_leaf, &smaller_leaf_histogram_array_);
  }

  // init for the ordered gradients, only initialize when have 2 leaves
  if (smaller_leaf >= 0) {
    // only need to initialize for smaller leaf

    // Get leaf boundary
    const data_size_t* indices = data_partition_->indices();
    data_size_t begin = data_partition_->leaf_begin(smaller_leaf);
    data_size_t end = begin + data_partition_->leaf_count(smaller_leaf);

    // copy indices to the GPU:
    #ifdef DEBUG_GPU
    Log::Info("Copying indices, gradients and hessians to GPU...");
    #endif
    indices_future_ = boost::compute::copy_async(indices + begin, indices + end, device_data_indices_->begin(), queue_);

    // This is about 7% of time, to re-order gradient and hessians
    #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      ordered_hessians_[i - begin] = hessians_[indices[i]];
    }
    // copy ordered hessians to the GPU:
    hessians_future_ = boost::compute::copy_async(ordered_hessians_.begin(), ordered_hessians_.begin() + end - begin, device_hessians_->begin(), queue_);
    #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      ordered_gradients_[i - begin] = gradients_[indices[i]];
    }
    // copy ordered gradients to the GPU:
    gradients_future_ = boost::compute::copy_async(ordered_gradients_.begin(), ordered_gradients_.begin() + end - begin, device_gradients_->begin(), queue_);
    // assign pointer
    ptr_to_ordered_gradients_smaller_leaf_ = ordered_gradients_.data();
    ptr_to_ordered_hessians_smaller_leaf_ = ordered_hessians_.data();
    

    #ifdef DEBUG_GPU
    Log::Info("gradients/hessians/indiex copied to device with size %d", end - begin);
    #endif

    // usually we can substract to get the histogram for the larger leaf,
    // but sometimes we don't have that histogram available
    if (parent_leaf_histogram_array_ == nullptr) {
      // need order gradient for larger leaf
      data_size_t smaller_size = end - begin;
      data_size_t larger_begin = data_partition_->leaf_begin(larger_leaf);
      data_size_t larger_end = larger_begin + data_partition_->leaf_count(larger_leaf);
      // copy
      #pragma omp parallel for schedule(static)
      for (data_size_t i = larger_begin; i < larger_end; ++i) {
        ordered_gradients_[smaller_size + i - larger_begin] = gradients_[indices[i]];
        ordered_hessians_[smaller_size + i - larger_begin] = hessians_[indices[i]];
      }
      ptr_to_ordered_gradients_larger_leaf_ = ordered_gradients_.data() + smaller_size;
      ptr_to_ordered_hessians_larger_leaf_ = ordered_hessians_.data() + smaller_size;
    }
  }
  

  // split for the ordered bin
  if (has_ordered_bin_ && right_leaf >= 0) {
    // mark data that at left-leaf
    std::memset(is_data_in_leaf_.data(), 0, sizeof(char)*num_data_);
    const data_size_t* indices = data_partition_->indices();
    data_size_t begin = data_partition_->leaf_begin(left_leaf);
    data_size_t end = begin + data_partition_->leaf_count(left_leaf);
    #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      is_data_in_leaf_[indices[i]] = 1;
    }
    // split the ordered bin
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < num_features_; ++i) {
      if (ordered_bins_[i] != nullptr) {
        ordered_bins_[i]->Split(left_leaf, right_leaf, is_data_in_leaf_.data());
      }
    }
  }
  return true;
}


// this function will go over all features on the smaller leaf,
// and call ConstructHistogram to build histograms for all features
// then call FindBestThreshold for all features to find the best split in each feature
// this is the core function we need to replace~!
void GPUTreeLearner::FindBestThresholds() {

  // Find histograms using GPU
  GPUHistogram(smaller_leaf_splits_->num_data_in_leaf(), smaller_leaf_histogram_array_);

  #pragma omp parallel for schedule(guided)
  // go over all features, find the best split
  // feature parallel here
  for (int feature_index = 0; feature_index < num_features_; feature_index++) {
    // feature is not used
    if ((!is_feature_used_.empty() && is_feature_used_[feature_index] == false)) continue;
    // if parent(larger) leaf cannot split at current feature
    if (parent_leaf_histogram_array_ != nullptr && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }

    // construct histograms for smaller leaf
    // histograms for the larger leaf can thus be computed
    if (ordered_bins_[feature_index] == nullptr) {
      // if not use ordered bin
      // this is the major computation step, for dense data
      #if DEBUG_COMPARE
      printf("Comparing histogram for feature %d\n", feature_index);
      size_t size = smaller_leaf_histogram_array_[feature_index].SizeOfHistgram() / sizeof(HistogramBinEntry);
      HistogramBinEntry* current_histogram = smaller_leaf_histogram_array_[feature_index].GetData(false);
      HistogramBinEntry* gpu_histogram = new HistogramBinEntry[size];
      std::copy(current_histogram, current_histogram + size, gpu_histogram);
      // PrintHistograms(smaller_leaf_histogram_array_[feature_index].GetData(false), 
      //                smaller_leaf_histogram_array_[feature_index].SizeOfHistgram() / sizeof(HistogramBinEntry));
      train_data_->FeatureAt(feature_index)->bin_data()->ConstructHistogram(
        smaller_leaf_splits_->data_indices(),
        smaller_leaf_splits_->num_data_in_leaf(),
        ptr_to_ordered_gradients_smaller_leaf_,
        ptr_to_ordered_hessians_smaller_leaf_,
        smaller_leaf_histogram_array_[feature_index].GetData());
      CompareHistograms(gpu_histogram, current_histogram, size);
      delete [] gpu_histogram;
      #endif
    } else {
      // used ordered bin
      ordered_bins_[feature_index]->ConstructHistogram(smaller_leaf_splits_->LeafIndex(),
        gradients_,
        hessians_,
        smaller_leaf_histogram_array_[feature_index].GetData());
    }
    // find best threshold for smaller child
    smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
      smaller_leaf_splits_->sum_gradients(),
      smaller_leaf_splits_->sum_hessians(),
      smaller_leaf_splits_->num_data_in_leaf(),
      &smaller_leaf_splits_->BestSplitPerFeature()[feature_index]);

    // only has root leaf
    if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->LeafIndex() < 0) continue;

    if (parent_leaf_histogram_array_ != nullptr) {
      // construct histgroms for large leaf, we initialize larger leaf as the parent,
      // so we can just subtract the smaller leaf's histograms
      // this is the most common case
      larger_leaf_histogram_array_[feature_index].Subtract(smaller_leaf_histogram_array_[feature_index]);
    } else {
      // no parent histogram, we have to construct the histogram for the larger leaf
      Log::Info("No parent histogram. Doing more calculation!");
      // this can happen if the histogram pool is not large enough
      if (ordered_bins_[feature_index] == nullptr) {
        // if not use ordered bin
        // call ConstructHistogram in dense_bin to construct histogram for this feature
        train_data_->FeatureAt(feature_index)->bin_data()->ConstructHistogram(
          larger_leaf_splits_->data_indices(),
          larger_leaf_splits_->num_data_in_leaf(),
          ptr_to_ordered_gradients_larger_leaf_,
          ptr_to_ordered_hessians_larger_leaf_,
          larger_leaf_histogram_array_[feature_index].GetData());
      } else {
        // used ordered bin
        ordered_bins_[feature_index]->ConstructHistogram(larger_leaf_splits_->LeafIndex(),
          gradients_,
          hessians_,
          larger_leaf_histogram_array_[feature_index].GetData());
      }
    }

    // Now we have constructed for the larger child as well, by substracting if possible
    // find best threshold for larger child, for each feature
    larger_leaf_histogram_array_[feature_index].FindBestThreshold(
      larger_leaf_splits_->sum_gradients(),
      larger_leaf_splits_->sum_hessians(),
      larger_leaf_splits_->num_data_in_leaf(),
      &larger_leaf_splits_->BestSplitPerFeature()[feature_index]);
  }
}


void GPUTreeLearner::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  const SplitInfo& best_split_info = best_split_per_leaf_[best_Leaf];

  // left = parent
  *left_leaf = best_Leaf;
  // split tree, will return right leaf
  *right_leaf = tree->Split(best_Leaf, best_split_info.feature, 
    train_data_->FeatureAt(best_split_info.feature)->bin_type(),
    best_split_info.threshold,
    train_data_->FeatureAt(best_split_info.feature)->feature_index(),
    train_data_->FeatureAt(best_split_info.feature)->BinToValue(best_split_info.threshold),
    static_cast<double>(best_split_info.left_output),
    static_cast<double>(best_split_info.right_output),
    static_cast<data_size_t>(best_split_info.left_count),
    static_cast<data_size_t>(best_split_info.right_count),
    static_cast<double>(best_split_info.gain));

  // split data partition, at the best feature
  // this is also expensive
  // best_leaf is the index in the tree
  // bin_data() contains the example bin values of the split feature
  // threahold is the split value
  // rightleaf is another leaf left
  data_partition_->Split(best_Leaf, train_data_->FeatureAt(best_split_info.feature)->bin_data(),
                         best_split_info.threshold, *right_leaf);

  // init the leaves that used on next iteration
  // see which leaf is larger
  if (best_split_info.left_count < best_split_info.right_count) {
    smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                               best_split_info.left_sum_gradient,
                               best_split_info.left_sum_hessian);
    larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                               best_split_info.right_sum_gradient,
                               best_split_info.right_sum_hessian);
  } else {
    smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(), best_split_info.right_sum_gradient, best_split_info.right_sum_hessian);
    larger_leaf_splits_->Init(*left_leaf, data_partition_.get(), best_split_info.left_sum_gradient, best_split_info.left_sum_hessian);
  }
}

}  // namespace LightGBM
