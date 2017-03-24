#ifdef USE_GPU
#include "gpu_tree_learner.h"
#include "../io/dense_bin.hpp"
#include "../io/dense_nbits_bin.hpp"

#include <LightGBM/utils/array_args.h>
#include <LightGBM/bin.h>

#include <algorithm>
#include <vector>


#define GPU_DEBUG 0

namespace LightGBM {

GPUTreeLearner::GPUTreeLearner(const TreeConfig* tree_config)
  :SerialTreeLearner(tree_config) {
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
  if (ptr_pinned_feature_masks_) {
    queue_.enqueue_unmap_buffer(pinned_feature_masks_, ptr_pinned_feature_masks_);
  }
}

void GPUTreeLearner::Init(const Dataset* train_data) {
  // initialize SerialTreeLearner
  SerialTreeLearner::Init(train_data);
  // Initialize GPU buffers and kernels
  InitGPU(tree_config_->gpu_platform_id, tree_config_->gpu_device_id);
}

// some functions used for debugging the GPU histogram construction
#if GPU_DEBUG > 0

void PrintHistograms(HistogramBinEntry* h, size_t size) {
  size_t total = 0;
  for (size_t i = 0; i < size; ++i) {
    printf("%03lu=%9.3g,%9.3g,%7d\t", i, h[i].sum_gradients, h[i].sum_hessians, h[i].cnt);
    total += h[i].cnt;
    if ((i & 3) == 3)
        printf("\n");
  }
  printf("\nTotal examples: %lu\n", total);
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
      // printf("grad %g != %g (%d ULPs)\n", h1[i].sum_gradients, h2[i].sum_gradients, ulps);
      // goto err;
    }
    a.f = h1[i].sum_hessians;
    b.f = h2[i].sum_hessians;
    ulps = Float_t::ulp_diff(a, b);
    if (ulps > 0) {
      // printf("hessian %g != %g (%d ULPs)\n", h1[i].sum_hessians, h2[i].sum_hessians, ulps);
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
  #if GPU_DEBUG >= 4
  printf("Computing histogram for %d examples and (%d * %d) features\n", leaf_num_data, dword_features_, num_dense_feature4_);
  printf("We can have at most %d workgroups per feature4 for efficiency reasons.\n"
  "Best workgroup size per feature for full utilization is %d\n", (int)ceil(t), (1 << exp_workgroups_per_feature));
  #endif
  exp_workgroups_per_feature = std::min(exp_workgroups_per_feature, (int)ceil(log((double)t)/log(2.0)));
  if (exp_workgroups_per_feature < 0)
      exp_workgroups_per_feature = 0;
  if (exp_workgroups_per_feature > kMaxLogWorkgroupsPerFeature)
      exp_workgroups_per_feature = kMaxLogWorkgroupsPerFeature;
  // return 0;
  return exp_workgroups_per_feature;
}

void GPUTreeLearner::GPUHistogram(data_size_t leaf_num_data, bool use_all_features) {
  // we have already copied ordered gradients, ordered hessians and indices to GPU
  // decide the best number of workgroups working on one feature4 tuple
  // set work group size based on feature size
  // each 2^exp_workgroups_per_feature workgroups work on a feature4 tuple
  int exp_workgroups_per_feature = GetNumWorkgroupsPerFeature(leaf_num_data);
  int num_workgroups = (1 << exp_workgroups_per_feature) * num_dense_feature4_;
  if (num_workgroups > preallocd_max_num_wg_) {
    preallocd_max_num_wg_ = num_workgroups;
    Log::Info("Increasing preallocd_max_num_wg_ to %d for launching more workgroups.", preallocd_max_num_wg_);
    device_subhistograms_.reset(new boost::compute::vector<char>(
                              preallocd_max_num_wg_ * dword_features_ * device_bin_size_ * hist_bin_entry_sz_, ctx_));
    // we need to refresh the kernel arguments after reallocating
    for (int i = 0; i <= kMaxLogWorkgroupsPerFeature; ++i) {
      // The only argument that needs to be changed later is num_data_
      histogram_kernels_[i].set_arg(7, *device_subhistograms_);
      histogram_allfeats_kernels_[i].set_arg(7, *device_subhistograms_);
      histogram_fulldata_kernels_[i].set_arg(7, *device_subhistograms_);
    }
  }
  #if GPU_DEBUG >= 4
  printf("setting exp_workgroups_per_feature to %d, using %u work groups\n", exp_workgroups_per_feature, num_workgroups);
  printf("Constructing histogram with %d examples\n", leaf_num_data);
  #endif
  
  // the GPU kernel will process all features in one call, and each
  // 2^exp_workgroups_per_feature (compile time constant) workgroup will
  // process one feature4 tuple

  if (use_all_features) {
    histogram_allfeats_kernels_[exp_workgroups_per_feature].set_arg(4, leaf_num_data);
  }
  else {
    histogram_kernels_[exp_workgroups_per_feature].set_arg(4, leaf_num_data);
  }
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
    if (use_all_features) {
      kernel_wait_obj_ = boost::compute::wait_list(
                         queue_.enqueue_1d_range_kernel(histogram_allfeats_kernels_[exp_workgroups_per_feature], 0, num_workgroups * 256, 256));
    }
    else {
      kernel_wait_obj_ = boost::compute::wait_list(
                         queue_.enqueue_1d_range_kernel(histogram_kernels_[exp_workgroups_per_feature], 0, num_workgroups * 256, 256));
    }
  }
  // copy the results asynchronously. Size depends on if double precision is used
  size_t output_size = num_dense_feature4_ * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;
  boost::compute::event histogram_wait_event;
  host_histogram_outputs_ = (void*)queue_.enqueue_map_buffer_async(device_histogram_outputs_, boost::compute::command_queue::map_read, 
                                                                   0, output_size, histogram_wait_event, kernel_wait_obj_);
  // we will wait for this object in WaitAndGetHistograms
  histograms_wait_obj_ = boost::compute::wait_list(histogram_wait_event);
}

template <typename HistType>
void GPUTreeLearner::WaitAndGetHistograms(FeatureHistogram* histograms, const std::vector<int8_t>& is_feature_used) {
  HistType* hist_outputs = (HistType*) host_histogram_outputs_;
  // when the output is ready, the compuataion is done
  histograms_wait_obj_.wait();
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < num_dense_features_; ++i) {
    int dense_index = dense_feature_map_[i];
    if (!is_feature_used[dense_index]) {
      continue;
    }
    auto old_histogram_array = histograms[dense_index].RawData() - 1;
    int bin_size = train_data_->FeatureNumTotalBin(dense_index); 
    // printf("Feature %d, Bin = %d, Total Bin = %d\n", dense_index, train_data_->FeatureNumBin(dense_index), train_data_->FeatureNumTotalBin(dense_index));
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
  // how many feature tuples we have
  num_dense_feature4_ = (num_dense_features_ + (dword_features_ - 1)) / dword_features_;
  // leave some safe margin for prefetching
  int allocated_num_data_ = num_data_ + 256 * (1 << kMaxLogWorkgroupsPerFeature);
  // allocate memory for all features (FIXME: 4 GB barrier on some devices, need to split to multiple buffers)
  device_features_.reset();
  device_features_ = std::unique_ptr<boost::compute::vector<Feature4>>(new boost::compute::vector<Feature4>(num_dense_feature4_ * num_data_, ctx_));
  // unpin old buffer if necessary before destructing them
  if (ptr_pinned_gradients_) {
    queue_.enqueue_unmap_buffer(pinned_gradients_, ptr_pinned_gradients_);
  }
  if (ptr_pinned_hessians_) {
    queue_.enqueue_unmap_buffer(pinned_hessians_, ptr_pinned_hessians_);
  }
  if (ptr_pinned_feature_masks_) {
    queue_.enqueue_unmap_buffer(pinned_feature_masks_, ptr_pinned_feature_masks_);
  }
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
  // allocate feature mask
  feature_masks_.resize(num_dense_feature4_ * dword_features_);
  device_feature_masks_ = boost::compute::buffer(); // deallocate
  device_feature_masks_ = boost::compute::buffer(ctx_, num_dense_feature4_ * dword_features_, 
                          boost::compute::memory_object::read_only, nullptr);
  pinned_feature_masks_ = boost::compute::buffer(ctx_, num_dense_feature4_ * dword_features_, 
                                             boost::compute::memory_object::read_write | boost::compute::memory_object::use_host_ptr, 
                                             feature_masks_.data());
  ptr_pinned_feature_masks_ = queue_.enqueue_map_buffer(pinned_feature_masks_, boost::compute::command_queue::map_write_invalidate_region,
                                                        0, num_dense_feature4_ * dword_features_);
  memset(ptr_pinned_feature_masks_, 0, num_dense_feature4_ * dword_features_);
  // copy indices to the device
  device_data_indices_.reset();
  device_data_indices_ = std::unique_ptr<boost::compute::vector<data_size_t>>(new boost::compute::vector<data_size_t>(allocated_num_data_, ctx_));
  boost::compute::fill(device_data_indices_->begin(), device_data_indices_->end(), 0, queue_);
  // histogram bin entry size depends on the precision (single/double)
  hist_bin_entry_sz_ = tree_config_->gpu_use_dp ? sizeof(HistogramBinEntry) : sizeof(GPUHistogramBinEntry);
  Log::Info("Size of histogram bin entry: %d", hist_bin_entry_sz_);
  // create output buffer, each feature has a histogram with device_bin_size_ bins,
  // each work group generates a sub-histogram of dword_features_ features.
  if (!device_subhistograms_) {
    // only initialize once
    device_subhistograms_ = std::unique_ptr<boost::compute::vector<char>>(new boost::compute::vector<char>(
                              preallocd_max_num_wg_ * dword_features_ * device_bin_size_ * hist_bin_entry_sz_, ctx_));
  }
  // create atomic counters for inter-group coordination
  sync_counters_.reset();
  sync_counters_ = std::unique_ptr<boost::compute::vector<int>>(new boost::compute::vector<int>(
                    num_dense_feature4_, ctx_));
  boost::compute::fill(sync_counters_->begin(), sync_counters_->end(), 0, queue_);
  // The output buffer is allocated to host directly, to overlap compute and data transfer
  device_histogram_outputs_ = boost::compute::buffer(); // deallocate
  device_histogram_outputs_ = boost::compute::buffer(ctx_, num_dense_feature4_ * dword_features_ * device_bin_size_ * hist_bin_entry_sz_, 
                           boost::compute::memory_object::write_only | boost::compute::memory_object::alloc_host_ptr, nullptr);
  // find the dense features and group then into feature4
  int i, k, copied_feature4 = 0, dense_ind[dword_features_];
  dense_feature_map_.clear();
  device_bin_mults_.clear();
  sparse_feature_map_.clear();
  for (i = 0, k = 0; i < num_features_; ++i) {
    // looking for dword_features_ non-sparse features
    if (ordered_bins_[i] == nullptr) {
      dense_ind[k] = i;
      // decide if we need to redistribute the bin
      double t = device_bin_size_ / (double)train_data_->FeatureNumTotalBin(i);
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
    if (k == dword_features_) {
      k = 0;
      for (int j = 0; j < dword_features_; ++j) {
        dense_feature_map_.push_back(dense_ind[j]);
      }
      copied_feature4++;
    }
  }
  // for data transfer time
  auto start_time = std::chrono::steady_clock::now();
  // Now generate new data structure feature4, and copy data to the device
  int nthreads = std::min(omp_get_max_threads(), (int)dense_feature_map_.size() / dword_features_);
  std::vector<Feature4*> host4_vecs(nthreads);
  std::vector<boost::compute::buffer> host4_bufs(nthreads);
  std::vector<Feature4*> host4_ptrs(nthreads);
  // preallocate arrays for all threads, and pin them
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nthreads; ++i) {
    host4_vecs[i] = (Feature4*)boost::alignment::aligned_alloc(4096, num_data_ * sizeof(Feature4));
    host4_bufs[i] = boost::compute::buffer(ctx_, num_data_ * sizeof(Feature4), 
                    boost::compute::memory_object::read_write | boost::compute::memory_object::use_host_ptr, 
                    host4_vecs[i]);
    host4_ptrs[i] = (Feature4*)queue_.enqueue_map_buffer(host4_bufs[i], boost::compute::command_queue::map_write_invalidate_region,
                    0, num_data_ * sizeof(Feature4));
  }
  #pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < dense_feature_map_.size() / dword_features_; ++i) {
    int tid = omp_get_thread_num();
    Feature4* host4 = host4_ptrs[tid];
    auto dense_ind = dense_feature_map_.begin() + i * dword_features_;
    auto dev_bin_mult = device_bin_mults_.begin() + i * dword_features_;
    #if GPU_DEBUG >= 1
    printf("Copying feature ");
    for (int l = 0; l < dword_features_; ++l) {
      printf("%d ", dense_ind[l]);
    }
    printf("to devices\n");
    #endif
    if (dword_features_ == 8) {
      // one feature datapoint is 4 bits
      BinIterator* bin_iters[8];
      for (int s_idx = 0; s_idx < 8; ++s_idx) {
        bin_iters[s_idx] = train_data_->FeatureIterator(dense_ind[s_idx]);
        if (dynamic_cast<Dense4bitsBinIterator*>(bin_iters[s_idx]) == 0) {
          Log::Fatal("GPU tree learner assumes that all bins are Dense4bitsBin when num_bin <= 16, but feature %d is not.", dense_ind[s_idx]);
        }
      }
      // this guarantees that the RawGet() function is inlined, rather than using virtual function dispatching
      Dense4bitsBinIterator iters[8] = {
        *static_cast<Dense4bitsBinIterator*>(bin_iters[0]),
        *static_cast<Dense4bitsBinIterator*>(bin_iters[1]),
        *static_cast<Dense4bitsBinIterator*>(bin_iters[2]),
        *static_cast<Dense4bitsBinIterator*>(bin_iters[3]),
        *static_cast<Dense4bitsBinIterator*>(bin_iters[4]),
        *static_cast<Dense4bitsBinIterator*>(bin_iters[5]),
        *static_cast<Dense4bitsBinIterator*>(bin_iters[6]),
        *static_cast<Dense4bitsBinIterator*>(bin_iters[7])};
      for (int j = 0; j < num_data_; ++j) {
        host4[j].s0 = (iters[0].RawGet(j) * dev_bin_mult[0] + ((j+0) & (dev_bin_mult[0] - 1))) 
                    |((iters[1].RawGet(j) * dev_bin_mult[1] + ((j+1) & (dev_bin_mult[1] - 1))) << 4);
        host4[j].s1 = (iters[2].RawGet(j) * dev_bin_mult[2] + ((j+2) & (dev_bin_mult[2] - 1))) 
                    |((iters[3].RawGet(j) * dev_bin_mult[3] + ((j+3) & (dev_bin_mult[3] - 1))) << 4);
        host4[j].s2 = (iters[4].RawGet(j) * dev_bin_mult[4] + ((j+4) & (dev_bin_mult[4] - 1))) 
                    |((iters[5].RawGet(j) * dev_bin_mult[5] + ((j+5) & (dev_bin_mult[5] - 1))) << 4);
        host4[j].s3 = (iters[6].RawGet(j) * dev_bin_mult[6] + ((j+6) & (dev_bin_mult[6] - 1))) 
                    |((iters[7].RawGet(j) * dev_bin_mult[7] + ((j+7) & (dev_bin_mult[7] - 1))) << 4);
      }
    }
    else if (dword_features_ == 4) {
      // one feature datapoint is one byte
      for (int s_idx = 0; s_idx < 4; ++s_idx) {
        BinIterator* bin_iter = train_data_->FeatureIterator(dense_ind[s_idx]);
        // this guarantees that the RawGet() function is inlined, rather than using virtual function dispatching
        if (dynamic_cast<DenseBinIterator<uint8_t>*>(bin_iter) != 0) {
          // Dense bin
          DenseBinIterator<uint8_t> iter = *static_cast<DenseBinIterator<uint8_t>*>(bin_iter);
          for (int j = 0; j < num_data_; ++j) {
            host4[j].s[s_idx] = iter.RawGet(j) * dev_bin_mult[s_idx] + ((j+s_idx) & (dev_bin_mult[s_idx] - 1));
          }
        }
        else if (dynamic_cast<Dense4bitsBinIterator*>(bin_iter) != 0) {
          // Dense 4-bit bin
          Dense4bitsBinIterator iter = *static_cast<Dense4bitsBinIterator*>(bin_iter);
          for (int j = 0; j < num_data_; ++j) {
            host4[j].s[s_idx] = iter.RawGet(j) * dev_bin_mult[s_idx] + ((j+s_idx) & (dev_bin_mult[s_idx] - 1));
          }
        }
        else {
          Log::Fatal("Bug in GPU tree builder: only DenseBin and Dense4bitsBin are supported!"); 
        }
      }
    }
    else {
      Log::Fatal("Bug in GPU tree builder: dword_features_ can only be 4 or 8!");
    }
    queue_.enqueue_write_buffer(device_features_->get_buffer(),
                        i * num_data_ * sizeof(Feature4), num_data_ * sizeof(Feature4), host4);
    #if GPU_DEBUG >= 1
    printf("first example of feature tuple is: %d %d %d %d\n", host4[0].s0, host4[0].s1, host4[0].s2, host4[0].s3);
    printf("Features copied to device with multiplier ");
    for (int l = 0; l < dword_features_; ++l) {
      printf("%d ", dev_bin_mult[l]);
    }
    printf("\n");
    #endif
  }
  if (k != 0) {
    Feature4* host4 = host4_ptrs[0];
    if (dword_features_ == 8) {
      memset(host4, 0, num_data_ * sizeof(Feature4));
    }
    #if GPU_DEBUG >= 1
    printf("%d features left\n", k);
    #endif
    for (i = 0; i < k; ++i) {
      if (dword_features_ == 8) {
        BinIterator* bin_iter = train_data_->FeatureIterator(dense_ind[i]);
        if (dynamic_cast<Dense4bitsBinIterator*>(bin_iter) != 0) {
          Dense4bitsBinIterator iter = *static_cast<Dense4bitsBinIterator*>(bin_iter);
          for (int j = 0; j < num_data_; ++j) {
            host4[j].s[i >> 1] |= ((iter.RawGet(j) * device_bin_mults_[copied_feature4 * dword_features_ + i]
                                + ((j+i) & (device_bin_mults_[copied_feature4 * dword_features_ + i] - 1)))
                               << ((i & 1) << 2));
          }
        }
        else {
          Log::Fatal("GPU tree learner assumes that all bins are Dense4bitsBin when num_bin <= 16, but feature %d is not.", dense_ind[i]);
        }
      }
      else if (dword_features_ == 4) {
        BinIterator* bin_iter = train_data_->FeatureIterator(dense_ind[i]);
        if (dynamic_cast<DenseBinIterator<uint8_t>*>(bin_iter) != 0) {
          DenseBinIterator<uint8_t> iter = *static_cast<DenseBinIterator<uint8_t>*>(bin_iter);
          for (int j = 0; j < num_data_; ++j) {
            host4[j].s[i] = iter.RawGet(j) * device_bin_mults_[copied_feature4 * dword_features_ + i] 
                          + ((j+i) & (device_bin_mults_[copied_feature4 * dword_features_ + i] - 1));
          }
        }
        else if (dynamic_cast<Dense4bitsBinIterator*>(bin_iter) != 0) {
          Dense4bitsBinIterator iter = *static_cast<Dense4bitsBinIterator*>(bin_iter);
          for (int j = 0; j < num_data_; ++j) {
            host4[j].s[i] = iter.RawGet(j) * device_bin_mults_[copied_feature4 * dword_features_ + i] 
                          + ((j+i) & (device_bin_mults_[copied_feature4 * dword_features_ + i] - 1));
          }
        }
        else {
          Log::Fatal("BUG in GPU tree builder: only DenseBin and Dense4bitsBin are supported!"); 
        }
      }
      else {
        Log::Fatal("Bug in GPU tree builder: dword_features_ can only be 4 or 8!");
      }
    }
    // fill the leftover features
    if (dword_features_ == 8) {
      for (int j = 0; j < num_data_; ++j) {
        for (i = k; i < dword_features_; ++i) {
          // fill this empty feature to some "random" value
          host4[j].s[i >> 1] |= ((j & 0xf) << ((i & 1) << 2));
        }
      }
    }
    else if (dword_features_ == 4) {
      for (int j = 0; j < num_data_; ++j) {
        for (i = k; i < dword_features_; ++i) {
          // fill this empty feature to some "random" value
          host4[j].s[i] = j;
        }
      }
    }
    // copying the last 1-3 features
    queue_.enqueue_write_buffer(device_features_->get_buffer(),
                        (num_dense_feature4_ - 1) * num_data_ * sizeof(Feature4), num_data_ * sizeof(Feature4), host4);
    #if GPU_DEBUG >= 1
    printf("Last features copied to device\n");
    #endif
    for (i = 0; i < k; ++i) {
      dense_feature_map_.push_back(dense_ind[i]);
    }
  }
  // deallocate pinned space for feature copying
  for (int i = 0; i < nthreads; ++i) {
      queue_.enqueue_unmap_buffer(host4_bufs[i], host4_ptrs[i]);
      host4_bufs[i] = boost::compute::buffer();
      boost::alignment::aligned_free(host4_vecs[i]);
  }
  // data transfer time
  std::chrono::duration<double, std::milli> end_time = std::chrono::steady_clock::now() - start_time;
  // setup kernel arguments
  for (int i = 0; i <= kMaxLogWorkgroupsPerFeature; ++i) {
    // The only argument that needs to be changed later is num_data_
    histogram_kernels_[i].set_args(*device_features_, device_feature_masks_, num_data_,
                                       *device_data_indices_, num_data_, device_gradients_, device_hessians_,
                                       *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
    histogram_allfeats_kernels_[i].set_args(*device_features_, device_feature_masks_, num_data_,
                                       *device_data_indices_, num_data_, device_gradients_, device_hessians_,
                                       *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
    histogram_fulldata_kernels_[i].set_args(*device_features_, device_feature_masks_, num_data_,
                                        *device_data_indices_, num_data_, device_gradients_, device_hessians_,
                                        *device_subhistograms_, *sync_counters_, device_histogram_outputs_);
  }
  Log::Info("%d dense features (%.2f MB) transfered to GPU in %f secs. %d sparse features.", 
            dense_feature_map_.size(), ((dense_feature_map_.size() + (dword_features_ - 1)) / dword_features_) * num_data_ * sizeof(Feature4) / (1024.0 * 1024.0), 
            end_time * 1e-3, sparse_feature_map_.size());
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
    printf("%d, ", train_data_->FeatureNumTotalBin(i));
    #endif
    max_num_bin_ = std::max(max_num_bin_, train_data_->FeatureNumTotalBin(i));
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
  if (max_num_bin_ <= 16) {
    kernel_source = kernel16_src_;
    kernel_name = "histogram16";
    device_bin_size_ = 16;
    dword_features_ = 8;
  }
  else if (max_num_bin_ <= 64) {
    kernel_source = kernel64_src_;
    kernel_name = "histogram64";
    device_bin_size_ = 64;
    dword_features_ = 4;
  }
  else if ( max_num_bin_ <= 256) {
    kernel_source = kernel256_src_;
    kernel_name = "histogram256";
    device_bin_size_ = 256;
    dword_features_ = 4;
  }
  else {
    Log::Fatal("bin size %d cannot run on GPU", max_num_bin_);
  }
  if(max_num_bin_ == 65) {
    Log::Warning("Setting max_bin to 63 is sugguested for best performance");
  }
  if(max_num_bin_ == 17) {
    Log::Warning("Setting max_bin to 15 is sugguested for best performance");
  }
  ctx_ = boost::compute::context(dev_);
  queue_ = boost::compute::command_queue(ctx_, dev_);
  Log::Info("Using GPU Device: %s, Vendor: %s", dev_.name().c_str(), dev_.vendor().c_str());
  Log::Info("Compiling OpenCL Kernel with %d bins...", device_bin_size_);
  // create OpenCL kernels for different number of workgroups per feature
  histogram_kernels_.resize(kMaxLogWorkgroupsPerFeature+1);
  histogram_allfeats_kernels_.resize(kMaxLogWorkgroupsPerFeature+1);
  histogram_fulldata_kernels_.resize(kMaxLogWorkgroupsPerFeature+1);
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i <= kMaxLogWorkgroupsPerFeature; ++i) {
    boost::compute::program program;
    std::ostringstream opts;
    opts << " -D POWER_FEATURE_WORKGROUPS=" << i
         << " -D USE_CONSTANT_BUF=" << use_constants << " -D USE_DP_FLOAT=" << int(tree_config_->gpu_use_dp)
         << " -cl-strict-aliasing -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math";
    #if GPU_DEBUG >= 1
    std::cout << "Building options: " << opts.str() << std::endl;
    #endif
    // kernel with indices in an array
    try {
      program = boost::compute::program::build_with_source(kernel_source, ctx_, opts.str());
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
    
    // kernel with all features enabled, with elimited branches
    opts << " -D ENABLE_ALL_FEATURES=1";
    try {
      program = boost::compute::program::build_with_source(kernel_source, ctx_, opts.str());
    }
    catch (boost::compute::opencl_error &e) {
      if (program.build_log().size() > 0) {
        Log::Fatal("GPU program built failure:\n %s", program.build_log().c_str());
      }
      else {
        Log::Fatal("GPU program built failure, log unavailable");
      }
    }
    histogram_allfeats_kernels_[i] = program.create_kernel(kernel_name);

    // kernel with all data indices (for root node, and assumes that root node always uses all features)
    opts << " -D IGNORE_INDICES=1";
    try {
      program = boost::compute::program::build_with_source(kernel_source, ctx_, opts.str());
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
  Log::Info("GPU programs have been built");
  AllocateGPUMemory();
}

void GPUTreeLearner::ResetTrainingData(const Dataset* train_data) {
  SerialTreeLearner::ResetTrainingData(train_data);
  // GPU memory has to been reallocated because data may have been changed
  AllocateGPUMemory();
}

void GPUTreeLearner::BeforeTrain() {

  #if GPU_DEBUG >= 2
  printf("Copying intial full gradients and hessians to device\n");
  #endif
  // Copy initial full hessians and gradients to GPU.
  // We start copying as early as possible, instead of at ConstructHistogram().
  if (!use_bagging_) {
    hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, num_data_ * sizeof(score_t), hessians_);
    gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, num_data_ * sizeof(score_t), gradients_);
  }

  // reset histogram pool
  histogram_pool_.ResetMap();

  if (tree_config_->feature_fraction < 1) {
    int used_feature_cnt = static_cast<int>(train_data_->num_total_features()*tree_config_->feature_fraction);
    // initialize used features
    std::memset(is_feature_used_.data(), 0, sizeof(int8_t) * num_features_);
    // Get used feature at current tree
    auto used_feature_indices = random_.Sample(train_data_->num_total_features(), used_feature_cnt);
  #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(used_feature_indices.size()); ++i) {
      int inner_feature_index = train_data_->InnerFeatureIndex(used_feature_indices[i]);
      if (inner_feature_index < 0) {  continue; }
      is_feature_used_[inner_feature_index] = 1;
    }
  } else {
  #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_features_; ++i) {
      is_feature_used_[i] = 1;
    }
  }

  // initialize data partition
  data_partition_->Init();

  // reset the splits for leaves
  for (int i = 0; i < tree_config_->num_leaves; ++i) {
    best_split_per_leaf_[i].Reset();
  }

  // Sumup for root
  if (data_partition_->leaf_count(0) == num_data_) {
    // use all data
    smaller_leaf_splits_->Init(gradients_, hessians_);

  } else {
    // use bagging, only use part of data
    smaller_leaf_splits_->Init(0, data_partition_.get(), gradients_, hessians_);
    // On GPU, we start copying indices, gradients and hessians now, instead at ConstructHistogram()
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
  }

  larger_leaf_splits_->Init();

  // if has ordered bin, need to initialize the ordered bin
  if (has_ordered_bin_) {
  #ifdef TIMETAG
    auto start_time = std::chrono::steady_clock::now();
  #endif
    if (data_partition_->leaf_count(0) == num_data_) {
      // use all data, pass nullptr
    #pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(ordered_bin_indices_.size()); ++i) {
        ordered_bins_[ordered_bin_indices_[i]]->Init(nullptr, tree_config_->num_leaves);
      }
    } else {
      // bagging, only use part of data

      // mark used data
      const data_size_t* indices = data_partition_->indices();
      data_size_t begin = data_partition_->leaf_begin(0);
      data_size_t end = begin + data_partition_->leaf_count(0);
    #pragma omp parallel for schedule(static)
      for (data_size_t i = begin; i < end; ++i) {
        is_data_in_leaf_[indices[i]] = 1;
      }
      // initialize ordered bin
    #pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(ordered_bin_indices_.size()); ++i) {
        ordered_bins_[ordered_bin_indices_[i]]->Init(is_data_in_leaf_.data(), tree_config_->num_leaves);
      }
    #pragma omp parallel for schedule(static)
      for (data_size_t i = begin; i < end; ++i) {
        is_data_in_leaf_[indices[i]] = 0;
      }
    }
  #ifdef TIMETAG
    ordered_bin_time += std::chrono::steady_clock::now() - start_time;
  #endif
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
  int smaller_leaf = -1;
  // only have root
  if (right_leaf < 0) {
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
    larger_leaf_histogram_array_ = nullptr;
  } else if (num_data_in_left_child < num_data_in_right_child) {
    smaller_leaf = left_leaf;
    // put parent(left) leaf's histograms into larger leaf's histograms
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { parent_leaf_histogram_array_ = larger_leaf_histogram_array_; }
    histogram_pool_.Move(left_leaf, right_leaf);
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
  } else {
    smaller_leaf = right_leaf;
    // put parent(left) leaf's histograms to larger leaf's histograms
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { parent_leaf_histogram_array_ = larger_leaf_histogram_array_; }
    histogram_pool_.Get(right_leaf, &smaller_leaf_histogram_array_);
  }

  // Copy indices, gradients and hessians as early as possible
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

    #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      ordered_hessians_[i - begin] = hessians_[indices[i]];
    }
    // copy ordered hessians to the GPU:
    hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, (end - begin) * sizeof(score_t), ptr_pinned_hessians_);

    #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      ordered_gradients_[i - begin] = gradients_[indices[i]];
    }
    // copy ordered gradients to the GPU:
    gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, (end - begin) * sizeof(score_t), ptr_pinned_gradients_);

    #if GPU_DEBUG >= 2
    Log::Info("gradients/hessians/indiex copied to device with size %d", end - begin);
    #endif
  }

  // split for the ordered bin
  if (has_ordered_bin_ && right_leaf >= 0) {
  #ifdef TIMETAG
    auto start_time = std::chrono::steady_clock::now();
  #endif
    // mark data that at left-leaf
    const data_size_t* indices = data_partition_->indices();
    const auto left_cnt = data_partition_->leaf_count(left_leaf);
    const auto right_cnt = data_partition_->leaf_count(right_leaf);
    char mark = 1;
    data_size_t begin = data_partition_->leaf_begin(left_leaf);
    data_size_t end = begin + left_cnt;
    if (left_cnt > right_cnt) {
      begin = data_partition_->leaf_begin(right_leaf);
      end = begin + right_cnt;
      mark = 0;
    }
  #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      is_data_in_leaf_[indices[i]] = 1;
    }
    // split the ordered bin
  #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(ordered_bin_indices_.size()); ++i) {
      ordered_bins_[ordered_bin_indices_[i]]->Split(left_leaf, right_leaf, is_data_in_leaf_.data(), mark);
    }
  #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      is_data_in_leaf_[indices[i]] = 0;
    }
  #ifdef TIMETAG
    ordered_bin_time += std::chrono::steady_clock::now() - start_time;
  #endif
  }
  return true;

}

bool GPUTreeLearner::ConstructGPUHistogramsAsync(
  const std::vector<int8_t>& is_feature_used,
  const data_size_t* data_indices, data_size_t num_data,
  const score_t* gradients, const score_t* hessians,
  score_t* ordered_gradients, score_t* ordered_hessians) {

  if (num_data <= 0) {
    return false;
  }
  
  // copy data indices if it is not null
  if (data_indices != nullptr && num_data != num_data_) {
    indices_future_ = boost::compute::copy_async(data_indices, data_indices + num_data, device_data_indices_->begin(), queue_);
  }
  // generate and copy ordered_gradients if gradients is not null
  if (gradients != nullptr) {
    if (num_data != num_data_) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        ordered_gradients[i] = gradients[data_indices[i]];
      }
      gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, num_data * sizeof(score_t), ptr_pinned_gradients_);
    }
    else {
      gradients_future_ = queue_.enqueue_write_buffer_async(device_gradients_, 0, num_data * sizeof(score_t), gradients);
    }
  }
  // generate and copy ordered_hessians if hessians is not null
  if (hessians != nullptr) {
    if (num_data != num_data_) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        ordered_hessians[i] = hessians[data_indices[i]];
      }
      hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, num_data * sizeof(score_t), ptr_pinned_hessians_);
    }
    else {
      hessians_future_ = queue_.enqueue_write_buffer_async(device_hessians_, 0, num_data * sizeof(score_t), hessians);
    }
  }
  // construct the feature masks
  int used_dense_features = 0;
  for (int i = 0; i < num_dense_features_; ++i) {
    if (is_feature_used[dense_feature_map_[i]]) {
      feature_masks_[i] = 1;
      ++used_dense_features;
    }
    else {
      feature_masks_[i] = 0;
    }
  }
  bool use_all_features = used_dense_features == num_dense_features_;
#if GPU_DEBUG >= 1
  printf("feature masks:\n");
  for (unsigned int i = 0; i < feature_masks_.size(); ++i) {
    printf("%d ", feature_masks_[i]);
  }
  printf("\n");
  printf("%d features, %d used, %d\n", num_dense_features_, used_dense_features, use_all_features);
#endif
  if (!use_all_features) {
    queue_.enqueue_write_buffer(device_feature_masks_, 0, num_dense_feature4_ * dword_features_, ptr_pinned_feature_masks_);
  }
  // All data have been prepared, now run the GPU kernel
  GPUHistogram(num_data, use_all_features);
  return true;
}

void GPUTreeLearner::FindBestThresholds() {
#ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
#endif
  std::vector<int8_t> is_feature_used(num_features_, 0);
  std::vector<int8_t> is_sparse_feature_used(num_features_, 0);
  std::vector<int8_t> is_dense_feature_used(num_features_, 0);
#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used_[feature_index]) continue;
    if (parent_leaf_histogram_array_ != nullptr 
        && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    is_feature_used[feature_index] = 1;
    if (std::find(sparse_feature_map_.begin(), sparse_feature_map_.end(), feature_index) != sparse_feature_map_.end()) {
      is_sparse_feature_used[feature_index] = 1;
    }
    else {
      is_dense_feature_used[feature_index] = 1;
    }
  }
  bool use_subtract = true;
  if (parent_leaf_histogram_array_ == nullptr) {
    use_subtract = false;
  }

  // construct smaller leaf
  HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
  // first construct dense features on GPU asynchronously
  bool use_gpu = false;
#if 1
  use_gpu = ConstructGPUHistogramsAsync(is_feature_used,
    nullptr, smaller_leaf_splits_->num_data_in_leaf(),
    nullptr, nullptr,
    nullptr, nullptr);
  // then construct sparse features on CPU
  train_data_->ConstructHistograms(is_sparse_feature_used,
    smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
    smaller_leaf_splits_->LeafIndex(),
    ordered_bins_, gradients_, hessians_,
    ordered_gradients_.data(), ordered_hessians_.data(),
    ptr_smaller_leaf_hist_data);
#else
  train_data_->ConstructHistograms(is_feature_used,
    smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
    smaller_leaf_splits_->LeafIndex(),
    ordered_bins_, gradients_, hessians_,
    ordered_gradients_.data(), ordered_hessians_.data(),
    ptr_smaller_leaf_hist_data);
#endif
  // wait for GPU to finish
  if (use_gpu) {
    if (tree_config_->gpu_use_dp) {
      // use double precision
      WaitAndGetHistograms<HistogramBinEntry>(smaller_leaf_histogram_array_, is_feature_used);
    }
    else {
      // use single precision
      WaitAndGetHistograms<GPUHistogramBinEntry>(smaller_leaf_histogram_array_, is_feature_used);
    }
  }

  // check GPU results
  // #define DEBUG_COMPARE
  #ifdef DEBUG_COMPARE
  for (int i = 0; i < num_dense_features_; ++i) {
    int feature_index = dense_feature_map_[i];
    if (!is_feature_used[i])
      continue;
    size_t size = train_data_->FeatureNumBin(feature_index); 
    HistogramBinEntry* current_histogram = smaller_leaf_histogram_array_[feature_index].RawData() - 1;
    HistogramBinEntry* gpu_histogram = new HistogramBinEntry[size];
    data_size_t num_data = smaller_leaf_splits_->num_data_in_leaf();
    printf("Comparing histogram for feature %d (%d) size %d, %lu bins\n", feature_index, is_feature_used[feature_index], num_data, size);
    std::copy(current_histogram, current_histogram + size, gpu_histogram);
    std::memset(current_histogram, 0, train_data_->FeatureNumBin(feature_index) * sizeof(HistogramBinEntry));
    train_data_->FeatureBin(feature_index)->ConstructHistogram(
      num_data != num_data_ ? smaller_leaf_splits_->data_indices() : nullptr,
      num_data,
      num_data != num_data_ ? ordered_gradients_.data() : gradients_,
      num_data != num_data_ ? ordered_hessians_.data() : hessians_,
      smaller_leaf_histogram_array_[feature_index].RawData() - 1); 
    CompareHistograms(gpu_histogram, current_histogram, size, feature_index);
    std::copy(gpu_histogram, gpu_histogram + size, current_histogram);
    delete [] gpu_histogram;
  }
  #endif

  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    #if GPU_DEBUG >= 1
    printf("Constructing larger leaf!\n");
    #endif
    // construct larger leaf
    HistogramBinEntry* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - 1;
    // first construct dense features on GPU asynchronously
    // indices, gradients and hessians are not on GPU yet, need to transfer
    use_gpu = ConstructGPUHistogramsAsync(is_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
      gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data());
    // then construct sparse features on CPU
    train_data_->ConstructHistograms(is_sparse_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
      larger_leaf_splits_->LeafIndex(),
      ordered_bins_, gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data(),
      ptr_larger_leaf_hist_data);
    // wait for GPU to finish
    if (use_gpu) {
      if (tree_config_->gpu_use_dp) {
        // use double precision
        WaitAndGetHistograms<HistogramBinEntry>(larger_leaf_histogram_array_, is_feature_used);
      }
      else {
        // use single precision
        WaitAndGetHistograms<GPUHistogramBinEntry>(larger_leaf_histogram_array_, is_feature_used);
      }
    }
  }
#ifdef TIMETAG
  hist_time += std::chrono::steady_clock::now() - start_time;
#endif
#ifdef TIMETAG
  start_time = std::chrono::steady_clock::now();
#endif
  std::vector<SplitInfo> smaller_best(num_threads_);
  std::vector<SplitInfo> larger_best(num_threads_);
  // find splits
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used[feature_index]) { continue; }
    const int tid = omp_get_thread_num();
    SplitInfo smaller_split;
#if GPU_DEBUG >= 3
    size_t bin_size = train_data_->FeatureNumBin(feature_index) + 1; 
    printf("feature %d smaller leaf (before fix):\n", feature_index);
    PrintHistograms(smaller_leaf_histogram_array_[feature_index].RawData() - 1, bin_size);
#endif

    train_data_->FixHistogram(feature_index,
                              smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
                              smaller_leaf_splits_->num_data_in_leaf(),
                              smaller_leaf_histogram_array_[feature_index].RawData());

#if GPU_DEBUG >= 3
    printf("feature %d smaller leaf:\n", feature_index);
    PrintHistograms(smaller_leaf_histogram_array_[feature_index].RawData() - 1, bin_size);
#endif
    smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
      smaller_leaf_splits_->sum_gradients(),
      smaller_leaf_splits_->sum_hessians(),
      smaller_leaf_splits_->num_data_in_leaf(),
      &smaller_split);
    if (smaller_split.gain > smaller_best[tid].gain) {
      smaller_best[tid] = smaller_split;
      smaller_best[tid].feature = train_data_->RealFeatureIndex(feature_index);
    }
    // only has root leaf
    if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->LeafIndex() < 0) { continue; }

    if (use_subtract) {
      larger_leaf_histogram_array_[feature_index].Subtract(smaller_leaf_histogram_array_[feature_index]);
    } else {
      train_data_->FixHistogram(feature_index, larger_leaf_splits_->sum_gradients(), larger_leaf_splits_->sum_hessians(),
                                larger_leaf_splits_->num_data_in_leaf(),
                                larger_leaf_histogram_array_[feature_index].RawData());
    }
#if GPU_DEBUG >= 4
    printf("feature %d larger leaf:\n", feature_index);
    PrintHistograms(larger_leaf_histogram_array_[feature_index].RawData() - 1, bin_size);
#endif
    SplitInfo larger_split;
    // find best threshold for larger child
    larger_leaf_histogram_array_[feature_index].FindBestThreshold(
      larger_leaf_splits_->sum_gradients(),
      larger_leaf_splits_->sum_hessians(),
      larger_leaf_splits_->num_data_in_leaf(),
      &larger_split);
    if (larger_split.gain > larger_best[tid].gain) {
      larger_best[tid] = larger_split;
      larger_best[tid].feature = train_data_->RealFeatureIndex(feature_index);
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
#ifdef TIMETAG
  find_split_time += std::chrono::steady_clock::now() - start_time;
#endif
}

void GPUTreeLearner::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  const SplitInfo& best_split_info = best_split_per_leaf_[best_Leaf];
  const int inner_feature_index = train_data_->InnerFeatureIndex(best_split_info.feature);
  // left = parent
  *left_leaf = best_Leaf;
#if GPU_DEBUG >= 2
  printf("spliting leaf %d with feature %d thresh %d gain %f stat %f %f %f %f\n", best_Leaf, best_split_info.feature, best_split_info.threshold, best_split_info.gain, best_split_info.left_sum_gradient, best_split_info.right_sum_gradient, best_split_info.left_sum_hessian, best_split_info.right_sum_hessian);
#endif
  // split tree, will return right leaf
  *right_leaf = tree->Split(best_Leaf,
                            inner_feature_index,
                            train_data_->FeatureBinMapper(inner_feature_index)->bin_type(),
                            best_split_info.threshold,
                            best_split_info.feature,
                            train_data_->RealThreshold(inner_feature_index, best_split_info.threshold),
                            static_cast<double>(best_split_info.left_output),
                            static_cast<double>(best_split_info.right_output),
                            static_cast<data_size_t>(best_split_info.left_count),
                            static_cast<data_size_t>(best_split_info.right_count),
                            static_cast<double>(best_split_info.gain));
  // split data partition
  data_partition_->Split(best_Leaf, train_data_, inner_feature_index,
                         best_split_info.threshold, *right_leaf);

  // init the leaves that used on next iteration
  if (best_split_info.left_count < best_split_info.right_count) {
    smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                               best_split_info.left_sum_gradient,
                               best_split_info.left_sum_hessian);
    larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                              best_split_info.right_sum_gradient,
                              best_split_info.right_sum_hessian);
    if ((best_split_info.left_count != smaller_leaf_splits_->num_data_in_leaf()) ||
        (best_split_info.right_count!= larger_leaf_splits_->num_data_in_leaf())) {
      printf("split %d: %d, smaller_leaf: %d, larger_leaf: %d\n", best_split_info.left_count, best_split_info.right_count, smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
      Log::Fatal("Bug in GPU histogram!!");
    }
  } else {
    smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(), best_split_info.right_sum_gradient, best_split_info.right_sum_hessian);
    larger_leaf_splits_->Init(*left_leaf, data_partition_.get(), best_split_info.left_sum_gradient, best_split_info.left_sum_hessian);
    if ((best_split_info.left_count != larger_leaf_splits_->num_data_in_leaf()) ||
        (best_split_info.right_count!= smaller_leaf_splits_->num_data_in_leaf())) {
      printf("split %d: %d, smaller_leaf: %d, larger_leaf: %d\n", best_split_info.left_count, best_split_info.right_count, smaller_leaf_splits_->num_data_in_leaf(), larger_leaf_splits_->num_data_in_leaf());
      Log::Fatal("Bug in GPU histogram!!");
    }
  }
}

}   // namespace LightGBM
#endif // USE_GPU

