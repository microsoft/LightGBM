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

};

} // namespace LightGBM

#endif // LIGHTGBM_TREELEARNER_CEGB_TREE_LEARNER_H_
