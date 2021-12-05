#ifndef LIGHTGBM_SAMPLE_STRATEGY_H_
#define LIGHTGBM_SAMPLE_STRATEGY_H_

#include <LightGBM/utils/random.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/tree_learner.h>
#include <LightGBM/objective_function.h>

namespace LightGBM {

class SampleStrategy {
 public:

  SampleStrategy() : balanced_bagging_(false), bagging_runner_(0, bagging_rand_block_) {};
 
  virtual ~SampleStrategy() {};
 
  static SampleStrategy* CreateSampleStrategy(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function, int num_tree_per_iteration);
 
  virtual void Bagging(int iter, TreeLearner* tree_learner, score_t* gradients, score_t* hessians) = 0;
 
  virtual void ResetGOSS() = 0;
 
  virtual void ResetBaggingConfig(const Config* config, bool is_change_dataset, 
          std::vector<score_t, Common::AlignmentAllocator<score_t, kAlignedSize>>& gradients, 
          std::vector<score_t, Common::AlignmentAllocator<score_t, kAlignedSize>>& hessians) = 0;
 
  bool is_use_subset() const { return is_use_subset_; }
 
  data_size_t bag_data_cnt() const { return bag_data_cnt_; }
 
  std::vector<data_size_t, Common::AlignmentAllocator<data_size_t, kAlignedSize>>& bag_data_indices() {return bag_data_indices_;}

 protected:
  const Config* config_;
  const Dataset* train_data_;
  const ObjectiveFunction* objective_function_;
  std::vector<data_size_t, Common::AlignmentAllocator<data_size_t, kAlignedSize>> bag_data_indices_;
  data_size_t bag_data_cnt_;
  data_size_t num_data_;
  int num_tree_per_iteration_;
  std::unique_ptr<Dataset> tmp_subset_;
  bool is_use_subset_;
  bool balanced_bagging_;
  const int bagging_rand_block_ = 1024;
  std::vector<Random> bagging_rands_;
  ParallelPartitionRunner<data_size_t, false> bagging_runner_;
};

} // namespace LightGBM
#endif // LIGHTGBM_SAMPLE_STRATEGY_H_
