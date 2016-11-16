#ifndef LIGHTGBM_TREELEARNER_PARALLEL_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_PARALLEL_TREE_LEARNER_H_

#include <LightGBM/utils/array_args.h>

#include <LightGBM/network.h>
#include "serial_tree_learner.h"

#include <cstring>

#include <vector>
#include <memory>

namespace LightGBM {

/*!
* \brief Feature parallel learning algorithm.
*        Different machine will find best split on different features, then sync global best split
*        It is recommonded used when #data is small or #feature is large
*/
class FeatureParallelTreeLearner: public SerialTreeLearner {
public:
  explicit FeatureParallelTreeLearner(const TreeConfig& tree_config);
  ~FeatureParallelTreeLearner();
  virtual void Init(const Dataset* train_data);

protected:
  void BeforeTrain() override;
  void FindBestSplitsForLeaves() override;
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
class DataParallelTreeLearner: public SerialTreeLearner {
public:
  explicit DataParallelTreeLearner(const TreeConfig& tree_config);
  ~DataParallelTreeLearner();
  void Init(const Dataset* train_data) override;
protected:
  void BeforeTrain() override;
  void FindBestThresholds() override;
  void FindBestSplitsForLeaves() override;
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


}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_PARALLEL_TREE_LEARNER_H_

