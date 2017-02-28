#ifndef LIGHTGBM_TREELEARNER_GPU_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_GPU_TREE_LEARNER_H_

#include <LightGBM/utils/random.h>
#include <LightGBM/utils/array_args.h>

#include <LightGBM/tree_learner.h>
#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>
#include <LightGBM/feature.h>
#include "feature_histogram.hpp"
#include "data_partition.hpp"
#include "split_info.hpp"
#include "leaf_splits.hpp"

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
// #include <boost/timer/timer.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/memory/local_buffer.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>

namespace LightGBM {


/*!
* \brief Used for learning a tree by single machine
*/
class GPUTreeLearner: public TreeLearner {
public:
  explicit GPUTreeLearner(const TreeConfig* tree_config);

  ~GPUTreeLearner();

  void Init(const Dataset* train_data) override;

  void ResetTrainingData(const Dataset* train_data) override;

  void ResetConfig(const TreeConfig* tree_config) override;

  Tree* Train(const score_t* gradients, const score_t *hessians) override;

  void SetBaggingData(const data_size_t* used_indices, data_size_t num_data) override {
    data_partition_->SetUsedDataIndices(used_indices, num_data);
  }

  void AddPredictionToScore(double* out_score) const override {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < data_partition_->num_leaves(); ++i) {
      double output = static_cast<double>(last_trained_tree_->LeafOutput(i));
      data_size_t cnt_leaf_data = 0;
      auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
        out_score[tmp_idx[j]] += output;
      }
    }
  }

protected:
  struct Feature4 {
	  union {
		  unsigned char s[4];
		  struct {
			  unsigned char s0;
			  unsigned char s1;
			  unsigned char s2;
			  unsigned char s3;
		  };
	  };
  };
  struct GPUHistogramBinEntry {
    score_t sum_gradients;
    score_t sum_hessians;
    uint32_t cnt;
  };

  /*!
  * \brief Some initial works before training
  */
  virtual void BeforeTrain();

  /*!
  * \brief Some initial works before FindBestSplit
  */
  virtual bool BeforeFindBestSplit(int left_leaf, int right_leaf);


  /*!
  * \brief Find best thresholds for all features, using multi-threading.
  *  The result will be stored in smaller_leaf_splits_ and larger_leaf_splits_.
  *  This function will be called in FindBestSplit.
  */
  virtual void FindBestThresholds();

  /*!
  * \brief Find best features for leaves from smaller_leaf_splits_ and larger_leaf_splits_.
  *  This function will be called after FindBestThresholds.
  */
  inline virtual void FindBestSplitsForLeaves();

  /*!
  * \brief Partition tree and data according best split.
  * \param tree Current tree, will be splitted on this function.
  * \param best_leaf The index of leaf that will be splitted.
  * \param left_leaf The index of left leaf after splitted.
  * \param right_leaf The index of right leaf after splitted.
  */
  virtual void Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf);

  int GetNumWorkgroupsPerFeature(data_size_t leaf_num_data);
  
  void InitGPU(int platform_id, int device_id);

  void GPUHistogram(data_size_t leaf_num_data, FeatureHistogram* histograms);
  
  void WaitAndGetHistograms(FeatureHistogram* histograms); 

  /*!
  * \brief Get the number of data in a leaf
  * \param leaf_idx The index of leaf
  * \return The number of data in the leaf_idx leaf
  */
  inline virtual data_size_t GetGlobalDataCountInLeaf(int leaf_idx) const;

  /*!
  * \brief Find best features for leaf from leaf_splits
  * \param leaf_splits
  */
  inline void FindBestSplitForLeaf(LeafSplits* leaf_splits);

  /*! \brief Last trained decision tree */
  const Tree* last_trained_tree_;
  /*! \brief number of data */
  data_size_t num_data_;
  /*! \brief number of features */
  int num_features_;
  /*! \brief training data */
  const Dataset* train_data_;
  /*! \brief gradients of current iteration */
  const score_t* gradients_;
  /*! \brief hessians of current iteration */
  const score_t* hessians_;
  /*! \brief training data partition on leaves */
  std::unique_ptr<DataPartition> data_partition_;
  /*! \brief used for generate used features */
  Random random_;
  /*! \brief used for sub feature training, is_feature_used_[i] = false means don't used feature i */
  std::vector<bool> is_feature_used_;
  /*! \brief pointer to histograms array of parent of current leaves */
  FeatureHistogram* parent_leaf_histogram_array_;
  /*! \brief pointer to histograms array of smaller leaf */
  FeatureHistogram* smaller_leaf_histogram_array_;
  /*! \brief pointer to histograms array of larger leaf */
  FeatureHistogram* larger_leaf_histogram_array_;

  /*! \brief store best split points for all leaves */
  std::vector<SplitInfo> best_split_per_leaf_;

  /*! \brief stores best thresholds for all feature for smaller leaf */
  std::unique_ptr<LeafSplits> smaller_leaf_splits_;
  /*! \brief stores best thresholds for all feature for larger leaf */
  std::unique_ptr<LeafSplits> larger_leaf_splits_;

  /*! \brief gradients of current iteration, ordered for cache optimized */
  std::vector<score_t> ordered_gradients_;
  /*! \brief hessians of current iteration, ordered for cache optimized */
  std::vector<score_t> ordered_hessians_;

  /*! \brief Pointer to ordered_gradients_, use this to avoid copy at BeforeTrain */
  const score_t* ptr_to_ordered_gradients_smaller_leaf_;
  /*! \brief Pointer to ordered_hessians_, use this to avoid copy at BeforeTrain*/
  const score_t* ptr_to_ordered_hessians_smaller_leaf_;

  /*! \brief Pointer to ordered_gradients_, use this to avoid copy at BeforeTrain */
  const score_t* ptr_to_ordered_gradients_larger_leaf_;
  /*! \brief Pointer to ordered_hessians_, use this to avoid copy at BeforeTrain*/
  const score_t* ptr_to_ordered_hessians_larger_leaf_;
  /*! \brief Store ordered bin */
  std::vector<std::unique_ptr<OrderedBin>> ordered_bins_;
  /*! \brief True if has ordered bin */
  bool has_ordered_bin_ = false;
  /*! \brief  is_data_in_leaf_[i] != 0 means i-th data is marked */
  std::vector<char> is_data_in_leaf_;
  /*! \brief used to cache historical histogram to speed up*/
  HistogramPool histogram_pool_;
  /*! \brief config of tree learner*/
  const TreeConfig* tree_config_;

  /*! \brief GPU related members */
  boost::compute::device dev_;
  boost::compute::context ctx_;
  boost::compute::command_queue queue_;
  /*! \brief GPU kernel for 256 bins */
  const char *kernel256_src_ = 
  #include "ocl/histogram256.cl"
  ;
  /*! \brief GPU kernel for 64 bins */
  const char *kernel64_src_ = 
  #include "ocl/histogram64.cl"
  ;
  /*! \brief GPU kernel for 64 bins */

  /*! \brief a array of histogram kernels with different number
     of workgroups per feature */
  std::vector<boost::compute::kernel> histogram_kernels_;
  boost::compute::kernel histogram_fulldata_kernel_;
  boost::compute::kernel reduction_kernel_;
  int num_dense_features_;
  int num_dense_feature4_;
  const int max_exp_workgroups_per_feature_ = 10; // 2^10
  const int max_num_workgroups_ = 1024;
  int max_num_bin_;
  int device_bin_size_;
  std::vector<int> dense_feature_map_;
  std::vector<int> sparse_feature_map_;
  std::vector<int> device_bin_mults_;
  std::unique_ptr<boost::compute::vector<Feature4>> device_features_;
  std::unique_ptr<boost::compute::vector<score_t>> device_gradients_;
  std::unique_ptr<boost::compute::vector<score_t>> device_hessians_;
  std::unique_ptr<boost::compute::vector<data_size_t>> device_data_indices_;
  std::unique_ptr<boost::compute::vector<int>> sync_counters_;
  std::unique_ptr<boost::compute::vector<char>> device_subhistograms_;
  boost::compute::buffer device_histogram_outputs_;
  boost::compute::wait_list kernel_wait_obj_;
  boost::compute::wait_list histograms_wait_obj_;
  std::unique_ptr<GPUHistogramBinEntry[]> host_histogram_outputs_;
  boost::compute::future<void> indices_future_;
  boost::compute::future<void> gradients_future_;
  boost::compute::future<void> hessians_future_;
};



inline void GPUTreeLearner::FindBestSplitsForLeaves() {
  FindBestSplitForLeaf(smaller_leaf_splits_.get());
  FindBestSplitForLeaf(larger_leaf_splits_.get());
}

inline data_size_t GPUTreeLearner::GetGlobalDataCountInLeaf(int leafIdx) const {
  if (leafIdx >= 0) {
    return data_partition_->leaf_count(leafIdx);
  } else {
    return 0;
  }
}

inline void GPUTreeLearner::FindBestSplitForLeaf(LeafSplits* leaf_splits) {
  if (leaf_splits == nullptr || leaf_splits->LeafIndex() < 0) {
    return;
  }
  std::vector<double> gains;
  for (size_t i = 0; i < leaf_splits->BestSplitPerFeature().size(); ++i) {
    gains.push_back(leaf_splits->BestSplitPerFeature()[i].gain);
  }
  int best_feature = static_cast<int>(ArrayArgs<double>::ArgMax(gains));
  int leaf = leaf_splits->LeafIndex();
  best_split_per_leaf_[leaf] = leaf_splits->BestSplitPerFeature()[best_feature];
  best_split_per_leaf_[leaf].feature = best_feature;
}

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_GPU_TREE_LEARNER_H_
