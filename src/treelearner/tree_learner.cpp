#include <LightGBM/tree_learner.h>

#include "serial_tree_learner.h"
#include "parallel_tree_learner.h"

namespace LightGBM {

TreeLearner* TreeLearner::CreateTreeLearner(TreeLearnerType type, const TreeConfig* tree_config) {
  if (type == TreeLearnerType::kSerialTreeLearner) {
    return new SerialTreeLearner(tree_config);
  } else if (type == TreeLearnerType::kFeatureParallelTreelearner) {
    return new FeatureParallelTreeLearner(tree_config);
  } else if (type == TreeLearnerType::kDataParallelTreeLearner) {
    return new DataParallelTreeLearner(tree_config);
  } else if (type == TreeLearnerType::KVotingParallelTreeLearner) {
    return new VotingParallelTreeLearner(tree_config);
  }
  return nullptr;
}

}  // namespace LightGBM
