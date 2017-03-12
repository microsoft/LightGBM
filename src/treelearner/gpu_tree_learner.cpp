#ifdef USE_GPU
#include "gpu_tree_learner.h"
#include "../io/dense_bin.hpp"

#include <LightGBM/utils/array_args.h>
#include <LightGBM/bin.h>

#include <algorithm>
#include <vector>

#define GPU_DEBUG 0

namespace LightGBM {

GPUTreeLearner::GPUTreeLearner(const TreeConfig* tree_config)
  :tree_config_(tree_config){
  random_ = Random(tree_config_->feature_fraction_seed);
  use_bagging_ = false;
  Log::Info("This is the GPU trainer!!");

}

GPUTreeLearner::~GPUTreeLearner() {
  if (ptr_pinned_gradients_) {
    queue_.enqueue_unmap_buffer(pinned_gradients_, ptr_pinned_gradients_);
  }
  if (ptr_pinned_hessians_) {
    queue_.enqueue_unmap_buffer(pinned_hessians_, ptr_pinned_hessians_);
  }
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
  // Initialize GPU buffers and kernels
  InitGPU(tree_config_->gpu_platform_id, tree_config_->gpu_device_id);
}

#if GPU_DEBUG > 0
// functions used for debugging the GPU histogram construction

void PrintHistograms(HistogramBinEntry* h, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    printf("%03lu=%9.3g,%9.3g,%7d\t", i, h[i].sum_gradients, h[i].sum_hessians, h[i].cnt);
    if ((i & 3) == 3)
        printf("\n");
  }
  printf("\n");
}

union Float_t
{
    int64_t i;
    double f;
    static int64_t ulp_diff(Float_t a, Float_t b) {
      return abs(a.i - b.i);
    }
};
  

void CompareHistograms(HistogramBinEntry* h1, HistogramBinEntry* h2, size_t size, int feature_id) {
  size_t i;
  Float_t a, b;
  for (i = 0; i < size; ++i) {
    a.f = h1[i].sum_gradients;
    b.f = h2[i].sum_gradients;
    int32_t ulps = Float_t::ulp_diff(a, b);
    if (fabs(h1[i].cnt           - h2[i].cnt != 0)) {
      printf("%d != %d\n", h1[i].cnt, h2[i].cnt);
      goto err;
    }
    if (ulps > 0) {
      printf("grad %g != %g (%d ULPs)\n", h1[i].sum_gradients, h2[i].sum_gradients, ulps);
      // goto err;
    }
    a.f = h1[i].sum_hessians;
    b.f = h2[i].sum_hessians;
    ulps = Float_t::ulp_diff(a, b);
    if (ulps > 0) {
      printf("hessian %g != %g (%d ULPs)\n", h1[i].sum_hessians, h2[i].sum_hessians, ulps);
      // goto err;
    }
  }
  return;
err:
  Log::Warning("Mismatched histograms found for feature %d at location %lu.", feature_id, i);
  std::cin.get();
  PrintHistograms(h1, size);
  printf("\n");
  PrintHistograms(h2, size);
  std::cin.get();
}
#endif

int GPUTreeLearner::GetNumWorkgroupsPerFeature(data_size_t leaf_num_data) {
  // we roughly want 256 workgroups per device, and we have num_dense_feature4_ feature tuples.
  // also guarantee that there are at least 2K examples per workgroup
  double x = 256.0 / num_dense_feature4_;
  int exp_workgroups_per_feature = ceil(log2(x));
  double t = leaf_num_data / 1024.0;
  #if GPU_DEBUG >= 2
  printf("Computing histogram for %d examples and (4 * %d) features\n", leaf_num_data, num_dense_feature4_);
  printf("We can have at most %d workgroups per feature4 for efficiency reasons.\n"
         "Best workgroup size per feature for full utilization is %d\n", (int)ceil(t), (1 << exp_workgroups_per_feature));
  #endif
  exp_workgroups_per_feature = std::min(exp_workgroups_per_feature, (int)ceil(log((double)t)/log(2.0)));
  if (exp_workgroups_per_feature < 0)
      exp_workgroups_per_feature = 0;
  if (exp_workgroups_per_feature > max_exp_workgroups_per_feature_)
      exp_workgroups_per_feature = max_exp_workgroups_per_feature_;
  // return 0;
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
    Log::Warning("BUG detected, num_workgroups %d too large!", num_workgroups);
  }
  #if GPU_DEBUG >= 2
  printf("setting exp_workgroups_per_feature to %d, using %u work groups\n", exp_workgroups_per_feature, num_workgroups);
  printf("Constructing histogram with %d examples\n", leaf_num_data);
  #endif
  
  // the GPU kernel will process all features in one call, and each
  // 2^exp_workgroups_per_feature (compile time constant) workgroup will
  // process one feature4 tuple

  histogram_kernels_[exp_workgroups_per_feature].set_arg(3, leaf_num_data);
  // for the root node, indices are not copied
  if (leaf_num_data != num_data_) {
    indices_future_.wait();
  }
  hessians_future_.wait();
  gradients_future_.wait();
  // there will be 2^exp_workgroups_per_feature = num_workgroups / num_dense_feature4 sub-histogram per feature4
  // and we will launch num_feature workgroups for this kernel
  // will launch threads for all features
  // the queue should be asynchrounous, and we will can WaitAndGetHistograms() before we start processing dense features
  if (leaf_num_data == num_data_) {
    kernel_wait_obj_ = boost::compute::wait_list(queue_.enqueue_1d_range_kernel(histogram_fulldata_kernels_[exp_workgroups_per_feature], 0, num_workgroups * 256, 256));
  }
  else {
    kernel_wait_obj_ = boost::compute::wait_list(
                       queue_.enqueue_1d_range_kernel(histogram_kernels_[exp_workgroups_per_feature], 0, num_workgroups * 256, 256));
  }
  // copy the results asynchronously. Size depends on if double precision is used
  size_t hist_bin_entry_sz = tree_config_->gpu_use_dp ? sizeof(HistogramBinEntry) : sizeof(GPUHistogramBinEntry);
  size_t size = num_dense_feature4_ * 4 * device_bin_size_ * hist_bin_entry_sz;
  boost::compute::event histogram_wait_event;
  host_histogram_outputs_ = (void*)queue_.enqueue_map_buffer_async(device_histogram_outputs_, boost::compute::command_queue::map_read, 
                                                                   0, size, histogram_wait_event, kernel_wait_obj_);
  // we will wait for this object in WaitAndGetHistograms
  histograms_wait_obj_ = boost::compute::wait_list(histogram_wait_event);
}

template <typename HistType>
void GPUTreeLearner::WaitAndGetHistograms(FeatureHistogram* histograms) {
  HistType* hist_outputs = (HistType*) host_histogram_outputs_;
  // when the output is ready, the compuataion is done
  histograms_wait_obj_.wait();
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < num_dense_features_; ++i) {
    int dense_index = dense_feature_map_[i];
    auto old_histogram_array = histograms[dense_index].GetData();
    int bin_size = histograms[dense_index].SizeOfHistgram() / sizeof(HistogramBinEntry);
    if (device_bin_mults_[i] == 1) {
      for (int j = 0; j < bin_size; ++j) {
        old_histogram_array[j].sum_gradients = hist_outputs[i * device_bin_size_+ j].sum_gradients;
        old_histogram_array[j].sum_hessians = hist_outputs[i * device_bin_size_ + j].sum_hessians;
        old_histogram_array[j].cnt = hist_outputs[i * device_bin_size_ + j].cnt;
      }
    }
    else {
      // values of this feature has been redistributed to multiple bins; need a reduction here
      int ind = 0;
      for (int j = 0; j < bin_size; ++j) {
        double sum_g = 0.0, sum_h = 0.0;
        size_t cnt = 0;
        for (int k = 0; k < device_bin_mults_[i]; ++k) {
          sum_g += hist_outputs[i * device_bin_size_+ ind].sum_gradients;
          sum_h += hist_outputs[i * device_bin_size_+ ind].sum_hessians;
          cnt += hist_outputs[i * device_bin_size_ + ind].cnt;
          ind++;
        }
        old_histogram_array[j].sum_gradients = sum_g;
        old_histogram_array[j].sum_hessians = sum_h;
        old_histogram_array[j].cnt = cnt;
      }
    }
  }
  queue_.enqueue_unmap_buffer(device_histogram_outputs_, host_histogram_outputs_);
}

void GPUTreeLearner::AllocateGPUMemory() {
  num_dense_features_ = 0;
  for (int i = 0; i < num_features_; ++i) {
    if (ordered_bins_[i] == nullptr) {
      num_dense_features_++;
    }
  }
  // how many 4-feature tuples we have
  num_dense_feature4_ = (num_dense_features_ + 3) / 4;
  // leave some safe margin for prefetching
  int allocated_num_data_ = num_data_ + 256 * (1 << max_exp_workgroups_per_feature_);
  // allocate memory for all features (FIXME: 4 GB barrier on some devices, need to split to multiple buffers)
  device_features_.reset();
  device_features_ = std::unique_ptr<boost::compute::vector<Feature4>>(new boost::compute::vector<Feature4>(num_dense_feature4_ * num_data_, ctx_));
  // make ordered_gradients and hessians larger (including extra room for prefetching), and pin them 
  ordered_gradients_.reserve(allocated_num_data_);
  ordered_hessians_.reserve(allocated_num_data_);
  pinned_gradients_ = boost::compute::buffer(); // deallocate
  pinned_gradients_ = boost::compute::buffer(ctx_, allocated_num_data_ * sizeof(score_t), 
                                             boost::compute::memory_object::read_write | boost::compute::memory_object::use_host_ptr, 
                                             ordered_gradients_.data());
  ptr_pinned_gradients_ = queue_.enqueue_map_buffer(pinned_gradients_, boost::compute::command_queue::map_write_invalidate_region, 
                                                    0, allocated_num_data_ * sizeof(score_t));
  pinned_hessians_ = boost::compute::buffer(); // deallocate
  pinned_hessians_  = boost::compute::buffer(ctx_, allocated_num_data_ * sizeof(score_t), 
                                             boost::compute::memory_object::read_write | boost::compute::memory_object::use_host_ptr, 
                                             ordered_hessians_.data());
  ptr_pinned_hessians_ = queue_.enqueue_map_buffer(pinned_hessians_, boost::compute::command_queue::map_write_invalidate_region, 
                                                   0, allocated_num_data_ * sizeof(score_t));
  // allocate space for gradients and hessians on device
  // we will copy gradients and hessians in after ordered_gradients_ and ordered_hessians_ are constructed
  device_gradients_ = boost::compute::buffer(); // deallocate
  device_gradients_ = boost::compute::buffer(ctx_, allocated_num_data_ * sizeof(score_t), 
                      boost::compute::memory_object::read_only, nullptr);
  device_hessians_ = boost::compute::buffer(); // deallocate
  device_hessians_  = boost::compute::buffer(ctx_, allocated_num_data_ * sizeof(score_t), 
                      boost::compute::memory_object::read_only, nullptr);
  // copy indices to the device
  device_data_indices_.reset();
  device_data_indices_ = std::unique_ptr<boost::compute::vector<data_size_t>>(new boost::compute::vector<data_size_t>(allocated_num_data_, ctx_));
  boost::compute::fill(device_data_indices_->begin(), device_data_indices_->end(), 0, queue_);
  // histogram bin entry size depends on the precision (single/double)
  size_t hist_bin_entry_sz = tree_config_->gpu_use_dp ? sizeof(HistogramBinEntry) : sizeof(GPUHistogramBinEntry);
  Log::Info("Size of histogram bin entry: %d", hist_bin_entry_sz);
  // create output buffer, each feature has a histogram with device_bin_size_ bins,
  // each work group generates a sub-histogram of 4 features.
  if (!device_subhistograms_) {
    // only initialize once
    device_subhistograms_ = std::unique_ptr<boost::compute::vector<char>>(new boost::compute::vector<char>(
                              max_num_workgroups_ * 4 * device_bin_size_ * hist_bin_entry_sz, ctx_));
  }
  // create atomic counters for inter-group coordination
  sync_counters_.reset();
  sync_counters_ = std::unique_ptr<boost::compute::vector<int>>(new boost::compute::vector<int>(
                    num_dense_feature4_, ctx_));
  boost::compute::fill(sync_counters_->begin(), sync_counters_->end(), 0, queue_);
  // The output buffer is allocated to host directly, to overlap compute and data transfer
  device_histogram_outputs_ = boost::compute::buffer(); // deallocate
  device_histogram_outputs_ = boost::compute::buffer(ctx_, num_dense_feature4_ * 4 * device_bin_size_ * hist_bin_entry_sz, 
                           boost::compute::memory_object::write_only | boost::compute::memory_object::alloc_host_ptr, nullptr);
  // find the dense features and group then into feature4
  int i, k, copied_feature4 = 0, dense_ind[4];
  dense_feature_map_.clear();
  device_bin_mults_.clear();
  sparse_feature_map_.clear();
  for (i = 0, k = 0; i < num_features_; ++i) {
    // looking for 4 non-sparse features
    if (ordered_bins_[i] == nullptr) {
      dense_ind[k] = i;
      // decide if we need to redistribute the bin
      double t = device_bin_size_ / (double)train_data_->FeatureAt(i)->num_bin();
      // multiplier must be a power of 2
      device_bin_mults_.push_back((int)round(pow(2, floor(log2(t)))));
      // device_bin_mults_.push_back(1);
      #if GPU_DEBUG >= 1
      printf("feature %d using multiplier %d\n", i, device_bin_mults_.back());
      #endif
      k++;
    }
    else {
      sparse_feature_map_.push_back(i);
    }
    // found 
    if (k == 4) {
      k = 0;
      dense_feature_map_.push_back(dense_ind[0]);
      dense_feature_map_.push_back(dense_ind[1]);
      dense_feature_map_.push_back(dense_ind[2]);
      dense_feature_map_.push_back(dense_ind[3]);
      copied_feature4++;
    }
  }
  // Now generate new data structure feature4, and copy data to the device
  #pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < dense_feature_map_.size() / 4; ++i) {
    std::vector<Feature4> host4(num_data_);
    auto dense_ind = dense_feature_map_.begin() + i * 4;
    auto dev_bin_mult = device_bin_mults_.begin() + i * 4;
    #if GPU_DEBUG >= 1
    printf("Copying feature %d, %d, %d, %d to device\n", dense_ind[0], dense_ind[1], dense_ind[2], dense_ind[3]);
    printf("feature size: %d, %d, %d, %d, %d\n", 
    static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[0])->bin_data())->num_data(),
    static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[1])->bin_data())->num_data(),
    static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[2])->bin_data())->num_data(),
    static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[3])->bin_data())->num_data(),
    num_data_);
    #endif
    for (int j = 0; j < num_data_; ++j) {
      host4[j].s0 = static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[0])->bin_data())->Get(j)
                    * dev_bin_mult[0] + ((j+0) & (dev_bin_mult[0] - 1));
      host4[j].s1 = static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[1])->bin_data())->Get(j)
                    * dev_bin_mult[1] + ((j+1) & (dev_bin_mult[1] - 1)); 
      host4[j].s2 = static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[2])->bin_data())->Get(j)
                    * dev_bin_mult[2] + ((j+2) & (dev_bin_mult[2] - 1));
      host4[j].s3 = static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[3])->bin_data())->Get(j)
                    * dev_bin_mult[3] + ((j+3) & (dev_bin_mult[3] - 1));
    }
    boost::compute::copy(host4.begin(), host4.end(), device_features_->begin() + i * num_data_, queue_);
    #if GPU_DEBUG >= 1
    printf("first example of features are: %d %d %d %d\n", host4[0].s0, host4[0].s1, host4[0].s2, host4[0].s3);
    printf("Feature %d, %d, %d, %d copied to device with multiplier %d %d %d %d\n", 
           dense_ind[0], dense_ind[1], dense_ind[2], dense_ind[3], dev_bin_mult[0], dev_bin_mult[1], dev_bin_mult[2], dev_bin_mult[3]);
    #endif
  }
  if (k != 0) {
    std::vector<Feature4> host4(num_data_);
    #if GPU_DEBUG >= 1
    printf("%d features left\n", k);
    #endif
    for (int j = 0; j < num_data_; ++j) {
      for (i = 0; i < k; ++i) {
        host4[j].s[i] = static_cast<const LightGBM::DenseBin<char>*>(train_data_->FeatureAt(dense_ind[i])->bin_data())->Get(j)
                        * device_bin_mults_[copied_feature4 * 4 + i] + ((j+i) & (device_bin_mults_[copied_feature4 * 4 + i] - 0));
      }
      for (i = k; i < 4; ++i) {
        // fill this empty feature to some "random" value
        host4[j].s[i] = j;
      }
    }
    // copying the last 1-3 features
    boost::compute::copy(host4.begin(), host4.end(), device_features_->begin() + (num_dense_feature4_ - 1) * num_data_, queue_);
    #if GPU_DEBUG >= 1
    printf("Last features copied to device\n");
    #endif
    for (i = 0; i < k; ++i) {
      dense_feature_map_.push_back(dense_ind[i]);
    }
  }
  // setup kernel arguments
  for (int i = 0; i <= max_exp_workgroups_per_feature_; ++i) {
    // The only argument that needs to be changed later is num_data_
    histogram_kernels_[i].set_args(*device_features_, num_data_,
                                       *device_data_indices_, num_data_, device_gradients_, device_hessians_,
                                       *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
    histogram_fulldata_kernels_[i].set_args(*device_features_, num_data_,
                                        *device_data_indices_, num_data_, device_gradients_, device_hessians_,
                                        *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
  }
  #if GPU_DEBUG >= 1
  printf("Dense feature list (size %lu): ", dense_feature_map_.size());
  for (i = 0; i < num_dense_features_; ++i) {
    printf("%d ", dense_feature_map_[i]);
  }
  printf("\n");
  printf("Sparse feature list (size %lu): ", sparse_feature_map_.size());
  for (i = 0; i < num_features_ - num_dense_features_; ++i) {
    printf("%d ", sparse_feature_map_[i]);
  }
  printf("\n");
  #endif
}

void GPUTreeLearner::InitGPU(int platform_id, int device_id) {
  // currently we don't use constant memory
  int use_constants = 0;

  // Get the max bin size, used for selecting best GPU kernel
  max_num_bin_ = 0;
  #if GPU_DEBUG >= 1
  printf("bin size: ");
  #endif
  for (int i = 0; i < train_data_->num_features(); ++i) {
    #if GPU_DEBUG >= 1
    printf("%d, ", train_data_->FeatureAt(i)->num_bin());
    #endif
    max_num_bin_ = std::max(max_num_bin_, train_data_->FeatureAt(i)->num_bin());
  }
  #if GPU_DEBUG >= 1
  printf("\n");
  #endif
  // initialize GPU
  dev_ = boost::compute::system::default_device();
  if (platform_id >= 0 && device_id >= 0) {
    const std::vector<boost::compute::platform> platforms = boost::compute::system::platforms();
    if ((int)platforms.size() > platform_id) {
      const std::vector<boost::compute::device> platform_devices = platforms[platform_id].devices();
      if ((int)platform_devices.size() > device_id) {
        Log::Info("Using requested OpenCL platform %d device %d", platform_id, device_id);
        dev_ = platform_devices[device_id];
      }   
    }   
  }   
  std::string kernel_source;
  std::string kernel_name;
  // determine which kernel to use based on the max number of bins
  if (max_num_bin_ <= 64) {
    kernel_source = kernel64_src_;
    kernel_name = "histogram64";
    device_bin_size_ = 64;
  }
  else if ( max_num_bin_ <= 255) {
    kernel_source = kernel256_src_;
    kernel_name = "histogram256";
    device_bin_size_ = 256;
  }
  else {
    Log::Fatal("bin size %d cannot run on GPU", max_num_bin_);
  }
  ctx_ = boost::compute::context(dev_);
  queue_ = boost::compute::command_queue(ctx_, dev_);
  Log::Info("Using GPU Device: %s, Vendor: %s", dev_.name().c_str(), dev_.vendor().c_str());
  Log::Info("Compiling OpenCL Kernel with %d bins...", device_bin_size_);
  // create OpenCL kernels for different number of workgroups per feature
  histogram_kernels_.resize(max_exp_workgroups_per_feature_+1);
  histogram_fulldata_kernels_.resize(max_exp_workgroups_per_feature_+1);
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i <= max_exp_workgroups_per_feature_; ++i) {
    auto program = boost::compute::program::create_with_source(kernel_source, ctx_);
    std::ostringstream opts;
    opts << " -D POWER_FEATURE_WORKGROUPS=" << i
         << " -D USE_CONSTANT_BUF=" << use_constants << " -D USE_DP_FLOAT=" << int(tree_config_->gpu_use_dp)
         << " -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math";
    #if GPU_DEBUG >= 1
    std::cout << "Building options: " << opts.str() << std::endl;
    #endif
    // kernel with indices in an array
    try {
      program.build(opts.str());
    }
    catch (boost::compute::opencl_error &e) {
      if (program.build_log().size() > 0) {
        Log::Fatal("GPU program built failure:\n %s", program.build_log().c_str());
      }
      else {
        Log::Fatal("GPU program built failure, log unavailable");
      }
    }
    histogram_kernels_[i] = program.create_kernel(kernel_name);
    opts << " -D IGNORE_INDICES=1";
    // kernel with all data indices (for root node)
    program = boost::compute::program::create_with_source(kernel_source, ctx_);
    try {
      program.build(opts.str());
    }
    catch (boost::compute::opencl_error &e) {
      if (program.build_log().size() > 0) {
        Log::Fatal("GPU program built failure:\n %s", program.build_log().c_str());
      }
      else {
        Log::Fatal("GPU program built failure, log unavailable");
      }
    }
    histogram_fulldata_kernels_[i] = program.create_kernel(kernel_name);
  }
  AllocateGPUMemory();
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
  // GPU memory has to been reallocated because data may have been changed
  AllocateGPUMemory();
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
      Log::Info("No further splits with positive gain, best gain: %f, leaves: %d, left count: %d, right count: %d",
                   best_leaf_SplitInfo.gain, split + 1, best_leaf_SplitInfo.left_count, best_leaf_SplitInfo.right_count);
      break;
    }
    // split tree with best leaf
    Split(tree.get(), best_leaf, &left_leaf, &right_leaf);
  }
  return tree.release();
}

void GPUTreeLearner::BeforeTrain() {

  // copy indices, gradients and hessians to device, start as early as possible
  #if GPU_DEBUG >= 2
  printf("Copying intial full gradients and hessians to device\n");
  #endif
  // Copy initial hessians and gradients to GPU.
  // We start copying as early as possible.
  if (!use_bagging_) {
    hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, num_data_ * sizeof(score_t), hessians_);
    gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, num_data_ * sizeof(score_t), gradients_);
  }


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
    #if GPU_DEBUG > 0
    printf("Not using bagging, examples count = %d\n", data_partition_->leaf_count(0));
    #endif
    // No need to copy the index for the root
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
    #if GPU_DEBUG > 0
    printf("Using bagging, examples count = %d\n", cnt);
    #endif
    // transfer the indices to GPU
    indices_future_ = boost::compute::copy_async(indices, indices + cnt, device_data_indices_->begin(), queue_);
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < cnt; ++i) {
      ordered_hessians_[i] = hessians_[indices[i]];
    }
    // transfer hessian to GPU
    hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, cnt * sizeof(score_t), ordered_hessians_.data());
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < cnt; ++i) {
      ordered_gradients_[i] = gradients_[indices[i]];
    }
    // transfer gradients to GPU
    gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, cnt * sizeof(score_t), ordered_gradients_.data());
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
    #if GPU_DEBUG >= 2
    Log::Info("Copying indices, gradients and hessians to GPU...");
    printf("indices size %d being copied (left = %d, right = %d)\n", end - begin,num_data_in_left_child,num_data_in_right_child);
    #endif
    indices_future_ = boost::compute::copy_async(indices + begin, indices + end, device_data_indices_->begin(), queue_);


    // This is about 7% of time, to re-order gradient and hessians
    #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      ordered_hessians_[i - begin] = hessians_[indices[i]];
    }
    // copy ordered hessians to the GPU:
    // hessians_future_ = boost::compute::copy_async(ordered_hessians_.begin(), ordered_hessians_.begin() + end - begin, device_hessians_->begin(), queue_);
    hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, (end - begin) * sizeof(score_t), ptr_pinned_hessians_);


    #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      ordered_gradients_[i - begin] = gradients_[indices[i]];
    }
    // copy ordered gradients to the GPU:
    // gradients_future_ = boost::compute::copy_async(ordered_gradients_.begin(), ordered_gradients_.begin() + end - begin, device_gradients_->begin(), queue_);
    gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, (end - begin) * sizeof(score_t), ptr_pinned_gradients_);

    // assign pointer
    ptr_to_ordered_gradients_smaller_leaf_ = ordered_gradients_.data();
    ptr_to_ordered_hessians_smaller_leaf_ = ordered_hessians_.data();
    

    #if GPU_DEBUG >= 2
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

  // #define DEBUG_COMPARE
  #ifdef DEBUG_COMPARE
  // GPUHistogramBinEntry needs to be changed to HistogramBinEntry for double precision debugging
  auto tmp_histogram_outputs_ = std::unique_ptr<GPUHistogramBinEntry[]>(new GPUHistogramBinEntry[device_bin_size_ * num_dense_feature4_ * 4]());
  // initialize the subhistgram buffer to some known values
  for(int i = 0; i < 1; ++i) {
    for (int j = 0; j < device_bin_size_; ++j) {
      // printf("%f\n", tmp_histogram_outputs_[i * device_bin_size_ + j].sum_gradients);
      tmp_histogram_outputs_[i * device_bin_size_+ j].sum_gradients = std::numeric_limits<float>::quiet_NaN();
      tmp_histogram_outputs_[i * device_bin_size_ + j].sum_hessians = std::numeric_limits<float>::quiet_NaN();
      tmp_histogram_outputs_[i * device_bin_size_ + j].cnt = 99999999;
    }
  }
  for(int i = 0; i < max_num_workgroups_ * 4; ++i) {
    boost::compute::copy((const char *)(tmp_histogram_outputs_.get()), 
                         (const char*)(tmp_histogram_outputs_.get()) + device_bin_size_ * sizeof(GPUHistogramBinEntry), 
                         device_subhistograms_->begin() + i * device_bin_size_ * sizeof(GPUHistogramBinEntry), queue_);
  }
  #endif
  // Find histograms using GPU
  bool use_gpu = smaller_leaf_splits_->num_data_in_leaf() > 0;
  if (use_gpu) {
    GPUHistogram(smaller_leaf_splits_->num_data_in_leaf(), smaller_leaf_histogram_array_);
  }
  else {
    printf("Not using GPU because data size <= 0\n");
    // size is 0, so directly fill histogram with all 0s
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_dense_features_; ++i) {
      int dense_index = dense_feature_map_[i];
      auto old_histogram_array = smaller_leaf_histogram_array_[dense_index].GetData();
      int bin_size = smaller_leaf_histogram_array_[dense_index].SizeOfHistgram() / sizeof(HistogramBinEntry);
      // printf("copying dense feature %d (index %d) size %d\n", i, dense_index, bin_size);
      // histogram size can be smaller than 255 (not a fixed number for each feature)
      // but the GPU code can only handle up to 256
      for (int j = 0; j < bin_size; ++j) {
        old_histogram_array[j].sum_gradients = 0.0;
        old_histogram_array[j].sum_hessians = 0.0;
        old_histogram_array[j].cnt = 0;
      }
      // PrintHistograms(old_histogram_array, bin_size);
    }
  }

  // When GPU is computing the dense bins, CPU works on the sparse bins
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_features_ - num_dense_features_; ++i) {
    int feature_index = sparse_feature_map_[i];
    // feature is not used
    if ((!is_feature_used_.empty() && is_feature_used_[feature_index] == false)) continue;
    // if parent(larger) leaf cannot split at current feature
    if (parent_leaf_histogram_array_ != nullptr && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }

    // construct histograms for smaller leaf
    // histograms for the larger leaf can thus be computed
    // used ordered bin (we know it is a sparse feature)
    ordered_bins_[feature_index]->ConstructHistogram(smaller_leaf_splits_->LeafIndex(),
      gradients_,
      hessians_,
      smaller_leaf_histogram_array_[feature_index].GetData());
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
      ordered_bins_[feature_index]->ConstructHistogram(larger_leaf_splits_->LeafIndex(),
        gradients_,
        hessians_,
        larger_leaf_histogram_array_[feature_index].GetData());
    }

    // Now we have constructed for the larger child as well, by substracting if possible
    // find best threshold for larger child, for each feature
    larger_leaf_histogram_array_[feature_index].FindBestThreshold(
      larger_leaf_splits_->sum_gradients(),
      larger_leaf_splits_->sum_hessians(),
      larger_leaf_splits_->num_data_in_leaf(),
      &larger_leaf_splits_->BestSplitPerFeature()[feature_index]);
  }
  
  // wait for GPU, and then we will process the dense bins (find thresholds)
  if (use_gpu) {
    if (tree_config_->gpu_use_dp) {
      // use double precision
      WaitAndGetHistograms<HistogramBinEntry>(smaller_leaf_histogram_array_);
    }
    else {
      // use single precision
      WaitAndGetHistograms<GPUHistogramBinEntry>(smaller_leaf_histogram_array_);
    }
  }

  // check GPU results
  #ifdef DEBUG_COMPARE
  for (int i = 0; i < num_dense_features_; ++i) {
    int feature_index = dense_feature_map_[i];
    // printf("Comparing histogram for feature %d\n", feature_index);
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
    CompareHistograms(gpu_histogram, current_histogram, size, feature_index);
    std::copy(gpu_histogram, gpu_histogram + size, current_histogram);
    delete [] gpu_histogram;
  }
  #endif

  #pragma omp parallel for schedule(guided)
  // go over all features, find the best split
  // feature parallel here
  for (int i = 0; i < num_dense_features_; ++i) {
    int feature_index = dense_feature_map_[i];
    // feature is not used
    if ((!is_feature_used_.empty() && is_feature_used_[feature_index] == false)) continue;
    // if parent(larger) leaf cannot split at current feature
    if (parent_leaf_histogram_array_ != nullptr && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
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
        // if not use ordered bin
        // call ConstructHistogram in dense_bin to construct histogram for this feature
      train_data_->FeatureAt(feature_index)->bin_data()->ConstructHistogram(
        larger_leaf_splits_->data_indices(),
        larger_leaf_splits_->num_data_in_leaf(),
        ptr_to_ordered_gradients_larger_leaf_,
        ptr_to_ordered_hessians_larger_leaf_,
        larger_leaf_histogram_array_[feature_index].GetData());
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
    if ((best_split_info.left_count != smaller_leaf_splits_->num_data_in_leaf()) ||
        (best_split_info.right_count!= larger_leaf_splits_->num_data_in_leaf())) {
          Log::Warning("Bug in GPU histogram!!");
          printf("split %d: %d, smaller_leaf: %d, larger_leaf: %d\n", best_split_info.left_count, best_split_info.right_count, smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
        }
  } else {
    smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(), best_split_info.right_sum_gradient, best_split_info.right_sum_hessian);
    larger_leaf_splits_->Init(*left_leaf, data_partition_.get(), best_split_info.left_sum_gradient, best_split_info.left_sum_hessian);
    if ((best_split_info.left_count != larger_leaf_splits_->num_data_in_leaf()) ||
        (best_split_info.right_count!= smaller_leaf_splits_->num_data_in_leaf())) {
          Log::Warning("Bug in GPU histogram!!");
          printf("split %d: %d, smaller_leaf: %d, larger_leaf: %d\n", best_split_info.left_count, best_split_info.right_count, smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
        }
  }
}

}  // namespace LightGBM

#endif // USE_GPU
