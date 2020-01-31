/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/tree_learner.h>

#include "gpu_tree_learner.h"
#include "parallel_tree_learner.h"
#include "serial_tree_learner.h"

namespace LightGBM {

TreeLearner* TreeLearner::CreateTreeLearner(const std::string& learner_type, const std::string& device_type, const Config* config) {
  if (device_type == std::string("cpu")) {
    if (learner_type == std::string("serial")) {
      return new SerialTreeLearner(config);
    } else if (learner_type == std::string("feature")) {
      return new FeatureParallelTreeLearner<SerialTreeLearner>(config);
    } else if (learner_type == std::string("data")) {
      return new DataParallelTreeLearner<SerialTreeLearner>(config);
    } else if (learner_type == std::string("voting")) {
      return new VotingParallelTreeLearner<SerialTreeLearner>(config);
    }
  } else if (device_type == std::string("gpu")) {
    if (learner_type == std::string("serial")) {
      return new GPUTreeLearner(config);
    } else if (learner_type == std::string("feature")) {
      return new FeatureParallelTreeLearner<GPUTreeLearner>(config);
    } else if (learner_type == std::string("data")) {
      return new DataParallelTreeLearner<GPUTreeLearner>(config);
    } else if (learner_type == std::string("voting")) {
      return new VotingParallelTreeLearner<GPUTreeLearner>(config);
    }
  }
  return nullptr;
}

}  // namespace LightGBM
