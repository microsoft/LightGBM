#include <LightGBM/tree_learner.h>

#include "serial_tree_learner.h"
#ifdef USE_GPU
#include "gpu_tree_learner.h"
#endif
#include "parallel_tree_learner.h"

namespace LightGBM {

TreeLearner* TreeLearner::CreateTreeLearner(const std::string& type, const TreeConfig* tree_config) {
  if (type == std::string("serial")) {
    return new SerialTreeLearner(tree_config);
  } else if (type == std::string("gpu")) {
  #ifdef USE_GPU
    return new GPUTreeLearner(tree_config);
  #else
    Log::Fatal("GPU Tree Learner was not enabled in this build. Recompile with -DUSE_GPU=1");
  #endif
  } else if (type == std::string("feature-gpu")) {
    return new FeatureParallelTreeLearner<GPUTreeLearner>(tree_config);
  } else if (type == std::string("feature")) {
    return new FeatureParallelTreeLearner<SerialTreeLearner>(tree_config);
  } else if (type == std::string("data-gpu")) {
    return new DataParallelTreeLearner<GPUTreeLearner>(tree_config);
  } else if (type == std::string("data")) {
    return new DataParallelTreeLearner<SerialTreeLearner>(tree_config);
  } else if (type == std::string("voting-gpu")) {
    return new VotingParallelTreeLearner<GPUTreeLearner>(tree_config);
  } else if (type == std::string("voting")) {
    return new VotingParallelTreeLearner<SerialTreeLearner>(tree_config);
  }
  return nullptr;
}

}  // namespace LightGBM
