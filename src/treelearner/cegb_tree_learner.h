#ifndef LIGHTGBM_TREELEARNER_CEGB_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_CEGB_TREE_LEARNER_H_

#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/random.h>

#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>
#include <LightGBM/tree_learner.h>

#include "data_partition.hpp"
#include "feature_histogram.hpp"
#include "leaf_splits.hpp"
#include "serial_tree_learner.h"
#include "split_info.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

namespace LightGBM {

class CEGBTreeLearner : public SerialTreeLearner {
public:
  CEGBTreeLearner(const Config *config) : SerialTreeLearner(config)
  {
  }

  ~CEGBTreeLearner() {}

protected:
  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf) override;

private:
  std::vector<bool> coupled_features_used;
};

} // namespace LightGBM

#endif // LIGHTGBM_TREELEARNER_CEGB_TREE_LEARNER_H_
