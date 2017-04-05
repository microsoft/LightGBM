#include <LightGBM/tree_learner.h>

#include "serial_tree_learner.h"
#include "gpu_tree_learner.h"
#include "parallel_tree_learner.h"

namespace LightGBM {

TreeLearner* TreeLearner::CreateTreeLearner(const std::string& learner_type, const std::string& device_type, const TreeConfig* tree_config) {
  if (device_type == std::string("cpu")) {
    if (learner_type == std::string("serial")) {
      return new SerialTreeLearner(tree_config);
    } else if (learner_type == std::string("feature")) {
      return new FeatureParallelTreeLearner<SerialTreeLearner>(tree_config);
    } else if (learner_type == std::string("data")) {
      return new DataParallelTreeLearner<SerialTreeLearner>(tree_config);
    } else if (learner_type == std::string("voting")) {
      return new VotingParallelTreeLearner<SerialTreeLearner>(tree_config);
    }
  }
  else if (device_type == std::string("gpu")) {
    if (learner_type == std::string("serial")) {
      return new GPUTreeLearner(tree_config);
    } else if (learner_type == std::string("feature")) {
      return new FeatureParallelTreeLearner<GPUTreeLearner>(tree_config);
    } else if (learner_type == std::string("data")) {
      return new DataParallelTreeLearner<GPUTreeLearner>(tree_config);
    } else if (learner_type == std::string("voting")) {
      return new VotingParallelTreeLearner<GPUTreeLearner>(tree_config);
    }
  }
  return nullptr;
}

}  // namespace LightGBM
