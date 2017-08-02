#ifndef LIGHTGBM_TREELEARNER_PARALLEL_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_PARALLEL_TREE_LEARNER_H_

#include <LightGBM/utils/array_args.h>

#include <LightGBM/network.h>
#include "serial_tree_learner.h"
#include "gpu_tree_learner.h"

#include <cstring>

#include <vector>
#include <memory>

namespace LightGBM {

/*!
* \brief Feature parallel learning algorithm.
*        Different machine will find best split on different features, then sync global best split
*        It is recommonded used when #data is small or #feature is large
*/
template <typename TREELEARNER_T>
class FeatureParallelTreeLearner: public TREELEARNER_T {
public:
  explicit FeatureParallelTreeLearner(const TreeConfig* tree_config);
  ~FeatureParallelTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;

protected:
  void BeforeTrain() override;
  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;
private:
  /*! \brief rank of local machine */
  int rank_;
  /*! \brief Number of machines of this parallel task */
  int num_machines_;
  /*! \brief Buffer for network send */
  std::vector<char> input_buffer_;
  /*! \brief Buffer for network receive */
  std::vector<char> output_buffer_;
};

/*!
* \brief Data parallel learning algorithm.
*        Workers use local data to construct histograms locally, then sync up global histograms.
*        It is recommonded used when #data is large or #feature is small
*/
template <typename TREELEARNER_T>
class DataParallelTreeLearner: public TREELEARNER_T {
public:
  explicit DataParallelTreeLearner(const TreeConfig* tree_config);
  ~DataParallelTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetConfig(const TreeConfig* tree_config) override;
protected:
  void BeforeTrain() override;
  void FindBestSplits() override;
  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;
  void Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) override;

  inline data_size_t GetGlobalDataCountInLeaf(int leaf_idx) const override {
    if (leaf_idx >= 0) {
      return global_data_count_in_leaf_[leaf_idx];
    } else {
      return 0;
    }
  }

private:
  /*! \brief Rank of local machine */
  int rank_;
  /*! \brief Number of machines of this parallel task */
  int num_machines_;
  /*! \brief Buffer for network send */
  std::vector<char> input_buffer_;
  /*! \brief Buffer for network receive */
  std::vector<char> output_buffer_;
  /*! \brief different machines will aggregate histograms for different features,
       use this to mark local aggregate features*/
  std::vector<bool> is_feature_aggregated_;
  /*! \brief Block start index for reduce scatter */
  std::vector<int> block_start_;
  /*! \brief Block size for reduce scatter */
  std::vector<int> block_len_;
  /*! \brief Write positions for feature histograms */
  std::vector<int> buffer_write_start_pos_;
  /*! \brief Read positions for local feature histograms */
  std::vector<int> buffer_read_start_pos_;
  /*! \brief Size for reduce scatter */
  int reduce_scatter_size_;
  /*! \brief Store global number of data in leaves  */
  std::vector<data_size_t> global_data_count_in_leaf_;
};

/*!
* \brief Voting based data parallel learning algorithm.
* Like data parallel, but not aggregate histograms for all features.
* Here using voting to reduce features, and only aggregate histograms for selected features.
* When #data is large and #feature is large, you can use this to have better speed-up
*/
template <typename TREELEARNER_T>
class VotingParallelTreeLearner: public TREELEARNER_T {
public:
  explicit VotingParallelTreeLearner(const TreeConfig* tree_config);
  ~VotingParallelTreeLearner() { }
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetConfig(const TreeConfig* tree_config) override;
protected:
  void BeforeTrain() override;
  bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) override;
  void FindBestSplits() override;
  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;
  void Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) override;

  inline data_size_t GetGlobalDataCountInLeaf(int leaf_idx) const override {
    if (leaf_idx >= 0) {
      return global_data_count_in_leaf_[leaf_idx];
    } else {
      return 0;
    }
  }
  /*!
  * \brief Perform global voting
  * \param leaf_idx index of leaf
  * \param splits All splits from local voting
  * \param out Result of gobal voting, only store feature indices
  */
  void GlobalVoting(int leaf_idx, const std::vector<LightSplitInfo>& splits,
    std::vector<int>* out);
  /*!
  * \brief Copy local histgram to buffer
  * \param smaller_top_features Selected features for smaller leaf
  * \param larger_top_features Selected features for larger leaf
  */
  void CopyLocalHistogram(const std::vector<int>& smaller_top_features,
    const std::vector<int>& larger_top_features);

private:
  /*! \brief Tree config used in local mode */
  TreeConfig local_tree_config_;
  /*! \brief Voting size */
  int top_k_;
  /*! \brief Rank of local machine*/
  int rank_;
  /*! \brief Number of machines */
  int num_machines_;
  /*! \brief Buffer for network send */
  std::vector<char> input_buffer_;
  /*! \brief Buffer for network receive */
  std::vector<char> output_buffer_;
  /*! \brief different machines will aggregate histograms for different features,
  use this to mark local aggregate features*/
  std::vector<bool> smaller_is_feature_aggregated_;
  /*! \brief different machines will aggregate histograms for different features,
  use this to mark local aggregate features*/
  std::vector<bool> larger_is_feature_aggregated_;
  /*! \brief Block start index for reduce scatter */
  std::vector<int> block_start_;
  /*! \brief Block size for reduce scatter */
  std::vector<int> block_len_;
  /*! \brief Read positions for feature histgrams at smaller leaf */
  std::vector<int> smaller_buffer_read_start_pos_;
  /*! \brief Read positions for feature histgrams at larger leaf */
  std::vector<int> larger_buffer_read_start_pos_;
  /*! \brief Size for reduce scatter */
  int reduce_scatter_size_;
  /*! \brief Store global number of data in leaves  */
  std::vector<data_size_t> global_data_count_in_leaf_;
  /*! \brief Store global split information for smaller leaf  */
  std::unique_ptr<LeafSplits> smaller_leaf_splits_global_;
  /*! \brief Store global split information for larger leaf  */
  std::unique_ptr<LeafSplits> larger_leaf_splits_global_;
  /*! \brief Store global histogram for smaller leaf  */
  std::unique_ptr<FeatureHistogram[]> smaller_leaf_histogram_array_global_;
  /*! \brief Store global histogram for larger leaf  */
  std::unique_ptr<FeatureHistogram[]> larger_leaf_histogram_array_global_;

  std::vector<HistogramBinEntry> smaller_leaf_histogram_data_;
  std::vector<HistogramBinEntry> larger_leaf_histogram_data_;
  std::vector<FeatureMetainfo> feature_metas_;
};

// To-do: reduce the communication cost by using bitset to communicate.
inline void SyncUpGlobalBestSplit(char* input_buffer_, char* output_buffer_, SplitInfo* smaller_best_split, SplitInfo* larger_best_split, int max_cat_threshold) {
  // sync global best info
  int size = SplitInfo::Size(max_cat_threshold);
  smaller_best_split->CopyTo(input_buffer_);
  larger_best_split->CopyTo(input_buffer_ + size);
  Network::Allreduce(input_buffer_, size * 2, size, output_buffer_, 
                     [&size] (const char* src, char* dst, int len) {
    int used_size = 0;
    LightSplitInfo p1, p2;
    while (used_size < len) {
      p1.CopyFrom(src);
      p2.CopyFrom(dst);
      if (p1 > p2) {
        std::memcpy(dst, src, size);
      }
      src += size;
      dst += size;
      used_size += size;
    }
  });
  // copy back
  smaller_best_split->CopyFrom(output_buffer_);
  larger_best_split->CopyFrom(output_buffer_ + size);
}

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_PARALLEL_TREE_LEARNER_H_

