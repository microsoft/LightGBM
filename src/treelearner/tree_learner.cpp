#include <LightGBM/tree_learner.h>

#include "serial_tree_learner.h"
#include "gpu_tree_learner.h"
#include "parallel_tree_learner.h"

namespace LightGBM {

TreeLearner* TreeLearner::CreateTreeLearner(const std::string& type, const TreeConfig* tree_config) {
  if (type == std::string("serial")) {
    return new SerialTreeLearner(tree_config);
  } else if (type == std::string("gpu")) {
    return new GPUTreeLearner(tree_config);
  } else if (type == std::string("feature")) {
    return new FeatureParallelTreeLearner(tree_config);
  } else if (type == std::string("data")) {
    return new DataParallelTreeLearner(tree_config);
  } else if (type == std::string("voting")) {
    return new VotingParallelTreeLearner(tree_config);
  }
  return nullptr;
}

}  // namespace LightGBM
