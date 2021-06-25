/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_CUDA_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_CUDA_TREE_LEARNER_H_

#include <LightGBM/utils/random.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/dataset.h>
#include <LightGBM/feature_group.h>
#include <LightGBM/tree.h>

#include <string>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "feature_histogram.hpp"
#include "serial_tree_learner.h"
#include "data_partition.hpp"
#include "split_info.hpp"
#include "leaf_splits.hpp"

#ifdef USE_CUDA
#include <LightGBM/cuda/vector_cudahost.h>
#include "cuda_kernel_launcher.h"


using json11::Json;

namespace LightGBM {

/*!
* \brief CUDA-based parallel learning algorithm.
*/
class CUDATreeLearner: public SerialTreeLearner {
 public:
    explicit CUDATreeLearner(const Config* tree_config);
    ~CUDATreeLearner();
    void Init(const Dataset* train_data, bool is_constant_hessian) override;
    void ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian, bool reset_multi_val_bin) override;
    Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;
    void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override {
      SerialTreeLearner::SetBaggingData(subset, used_indices, num_data);
      if (subset == nullptr && used_indices != nullptr) {
        if (num_data != num_data_) {
          use_bagging_ = true;
          return;
        }
      }
      use_bagging_ = false;
    }

 protected:
    void BeforeTrain() override;
    bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) override;
    void FindBestSplits(const Tree* tree) override;
    void Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) override;
    void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;

 private:
    typedef float gpu_hist_t;

    /*!
     * \brief Find the best number of workgroups processing one feature for maximizing efficiency
     * \param leaf_num_data The number of data examples on the current leaf being processed
     * \return Log2 of the best number for workgroups per feature, in range 0...kMaxLogWorkgroupsPerFeature
     */
    int GetNumWorkgroupsPerFeature(data_size_t leaf_num_data);

    /*!
     * \brief Initialize GPU device
     * \param num_gpu: number of maximum gpus
     */
    void InitGPU(int num_gpu);

    /*!
     * \brief Allocate memory for GPU computation // alloc only
     */
    void CountDenseFeatureGroups();  // compute num_dense_feature_group
    void prevAllocateGPUMemory();  // compute CPU-side param calculation & Pin HostMemory
    void AllocateGPUMemory();

    /*!
     * \ ResetGPUMemory
     */
    void ResetGPUMemory();

    /*!
     * \ copy dense feature from CPU to GPU
     */
    void copyDenseFeature();

    /*! 
     * \brief Compute GPU feature histogram for the current leaf.
     *        Indices, gradients and Hessians have been copied to the device.
     * \param leaf_num_data Number of data on current leaf
     * \param use_all_features Set to true to not use feature masks, with a faster kernel
     */
    void GPUHistogram(data_size_t leaf_num_data, bool use_all_features);

    void SetThreadData(ThreadData* thread_data, int device_id, int histogram_size,
                int leaf_num_data, bool use_all_features,
                int num_workgroups, int exp_workgroups_per_feature) {
      ThreadData* td = &thread_data[device_id];
      td->device_id             = device_id;
      td->histogram_size        = histogram_size;
      td->leaf_num_data         = leaf_num_data;
      td->num_data              = num_data_;
      td->use_all_features      = use_all_features;
      td->is_constant_hessian   = share_state_->is_constant_hessian;
      td->num_workgroups        = num_workgroups;
      td->stream                = stream_[device_id];
      td->device_features       = device_features_[device_id];
      td->device_feature_masks  = reinterpret_cast<uint8_t *>(device_feature_masks_[device_id]);
      td->device_data_indices   = device_data_indices_[device_id];
      td->device_gradients      = device_gradients_[device_id];
      td->device_hessians       = device_hessians_[device_id];
      td->hessians_const        = hessians_[0];
      td->device_subhistograms  = device_subhistograms_[device_id];
      td->sync_counters         = sync_counters_[device_id];
      td->device_histogram_outputs   = device_histogram_outputs_[device_id];
      td->exp_workgroups_per_feature = exp_workgroups_per_feature;

      td->kernel_start           = &(kernel_start_[device_id]);
      td->kernel_wait_obj        = &(kernel_wait_obj_[device_id]);
      td->kernel_input_wait_time = &(kernel_input_wait_time_[device_id]);

      size_t output_size = num_gpu_feature_groups_[device_id] * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;
      size_t host_output_offset = offset_gpu_feature_groups_[device_id] * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;
      td->output_size           = output_size;
      td->host_histogram_output = reinterpret_cast<char*>(host_histogram_outputs_) + host_output_offset;
      td->histograms_wait_obj   = &(histograms_wait_obj_[device_id]);
    }

    /*!
     * \brief Wait for GPU kernel execution and read histogram
     * \param histograms Destination of histogram results from GPU.
     */
    template <typename HistType>
    void WaitAndGetHistograms(FeatureHistogram* leaf_histogram_array);

    /*!
     * \brief Construct GPU histogram asynchronously. 
     *        Interface is similar to Dataset::ConstructHistograms().
     * \param is_feature_used A predicate vector for enabling each feature
     * \param data_indices Array of data example IDs to be included in histogram, will be copied to GPU.
     *                     Set to nullptr to skip copy to GPU.
     * \param num_data Number of data examples to be included in histogram
     * \return true if GPU kernel is launched, false if GPU is not used
    */
    bool ConstructGPUHistogramsAsync(
      const std::vector<int8_t>& is_feature_used,
      const data_size_t* data_indices, data_size_t num_data);

    /*! brief Log2 of max number of workgroups per feature*/
    const int kMaxLogWorkgroupsPerFeature = 10;  // 2^10
    /*! brief Max total number of workgroups with preallocated workspace.
     *        If we use more than this number of workgroups, we have to reallocate subhistograms */
    std::vector<int> preallocd_max_num_wg_;

    /*! \brief True if bagging is used */
    bool use_bagging_;

    /*! \brief GPU command queue object */
    std::vector<cudaStream_t> stream_;

    /*! \brief total number of feature-groups */
    int num_feature_groups_;
    /*! \brief total number of dense feature-groups, which will be processed on GPU */
    int num_dense_feature_groups_;
    std::vector<int> num_gpu_feature_groups_;
    std::vector<int> offset_gpu_feature_groups_;
    /*! \brief On GPU we read one DWORD (4-byte) of features of one example once.
     *  With bin size > 16, there are 4 features per DWORD.
     *  With bin size <=16, there are 8 features per DWORD.
     */
    int dword_features_;
    /*! \brief Max number of bins of training data, used to determine 
     * which GPU kernel to use */
    int max_num_bin_;
    /*! \brief Used GPU kernel bin size (64, 256) */
    int histogram_size_;
    int device_bin_size_;
    /*! \brief Size of histogram bin entry, depending if single or double precision is used */
    size_t hist_bin_entry_sz_;
    /*! \brief Indices of all dense feature-groups */
    std::vector<int> dense_feature_group_map_;
    /*! \brief Indices of all sparse feature-groups */
    std::vector<int> sparse_feature_group_map_;
    /*! \brief GPU memory object holding the training data */
    std::vector<uint8_t*> device_features_;
    /*! \brief GPU memory object holding the ordered gradient */
    std::vector<score_t*> device_gradients_;
    /*! \brief GPU memory object holding the ordered hessian */
    std::vector<score_t*> device_hessians_;
    /*! \brief A vector of feature mask. 1 = feature used, 0 = feature not used */
    std::vector<char> feature_masks_;
    /*! \brief GPU memory object holding the feature masks */
    std::vector<char*> device_feature_masks_;
    /*! \brief Pointer to pinned memory of feature masks */
    char* ptr_pinned_feature_masks_ = nullptr;
    /*! \brief GPU memory object holding indices of the leaf being processed */
    std::vector<data_size_t*> device_data_indices_;
    /*! \brief GPU memory object holding counters for workgroup coordination */
    std::vector<int*> sync_counters_;
    /*! \brief GPU memory object holding temporary sub-histograms per workgroup */
    std::vector<char*> device_subhistograms_;
    /*! \brief Host memory object for histogram output (GPU will write to Host memory directly) */
    std::vector<void*> device_histogram_outputs_;
    /*! \brief Host memory pointer for histogram outputs */
    void *host_histogram_outputs_;
    /*! CUDA waitlist object for waiting for data transfer before kernel execution */
    std::vector<cudaEvent_t> kernel_wait_obj_;
    /*! CUDA waitlist object for reading output histograms after kernel execution */
    std::vector<cudaEvent_t> histograms_wait_obj_;
    /*! CUDA Asynchronous waiting object for copying indices */
    std::vector<cudaEvent_t> indices_future_;
    /*! Asynchronous waiting object for copying gradients */
    std::vector<cudaEvent_t> gradients_future_;
    /*! Asynchronous waiting object for copying Hessians */
    std::vector<cudaEvent_t> hessians_future_;
    /*! Asynchronous waiting object for copying dense features */
    std::vector<cudaEvent_t> features_future_;

    // host-side buffer for converting feature data into featre4 data
    int nthreads_;  // number of Feature4* vector on host4_vecs_
    std::vector<cudaEvent_t> kernel_start_;
    std::vector<float> kernel_time_;  // measure histogram kernel time
    std::vector<std::chrono::duration<double, std::milli>> kernel_input_wait_time_;
    int num_gpu_;
    int allocated_num_data_;  // allocated data instances
    pthread_t **cpu_threads_;  // pthread, 1 cpu thread / gpu
};

}  // namespace LightGBM
#else  // USE_CUDA

// When GPU support is not compiled in, quit with an error message

namespace LightGBM {

class CUDATreeLearner: public SerialTreeLearner {
 public:
    #pragma warning(disable : 4702)
    explicit CUDATreeLearner(const Config* tree_config) : SerialTreeLearner(tree_config) {
      Log::Fatal("CUDA Tree Learner was not enabled in this build.\n"
                 "Please recompile with CMake option -DUSE_CUDA=1");
    }
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_TREELEARNER_CUDA_TREE_LEARNER_H_
