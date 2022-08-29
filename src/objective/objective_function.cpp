/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/objective_function.h>

#include "binary_objective.hpp"
#include "multiclass_objective.hpp"
#include "rank_objective.hpp"
#include "regression_objective.hpp"
#include "xentropy_objective.hpp"

#include "cuda/cuda_binary_objective.hpp"

namespace LightGBM {

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& type, const Config& config) {
  #ifdef USE_CUDA_EXP
  if (config.device_type == std::string("cuda_exp") && config.boosting == std::string("gbdt")) {
    if (type == std::string("regression")) {
      Log::Warning("Objective regression is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new RegressionL2loss(config);
    } else if (type == std::string("regression_l1")) {
      Log::Warning("Objective regression_l1 is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new RegressionL1loss(config);
    } else if (type == std::string("quantile")) {
      Log::Warning("Objective quantile is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new RegressionQuantileloss(config);
    } else if (type == std::string("huber")) {
      Log::Warning("Objective huber is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new RegressionHuberLoss(config);
    } else if (type == std::string("fair")) {
      Log::Warning("Objective fair is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new RegressionFairLoss(config);
    } else if (type == std::string("poisson")) {
      Log::Warning("Objective poisson is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new RegressionPoissonLoss(config);
    } else if (type == std::string("binary")) {
      return new CUDABinaryLogloss(config);
    } else if (type == std::string("lambdarank")) {
      Log::Warning("Objective lambdarank is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new LambdarankNDCG(config);
    } else if (type == std::string("rank_xendcg")) {
      Log::Warning("Objective rank_xendcg is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new RankXENDCG(config);
    } else if (type == std::string("multiclass")) {
      Log::Warning("Objective multiclass is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new MulticlassSoftmax(config);
    } else if (type == std::string("multiclassova")) {
      Log::Warning("Objective multiclassova is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new MulticlassOVA(config);
    } else if (type == std::string("cross_entropy")) {
      Log::Warning("Objective cross_entropy is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new CrossEntropy(config);
    } else if (type == std::string("cross_entropy_lambda")) {
      Log::Warning("Objective cross_entropy_lambda is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new CrossEntropyLambda(config);
    } else if (type == std::string("mape")) {
      Log::Warning("Objective mape is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new RegressionMAPELOSS(config);
    } else if (type == std::string("gamma")) {
      Log::Warning("Objective gamma is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new RegressionGammaLoss(config);
    } else if (type == std::string("tweedie")) {
      Log::Warning("Objective tweedie is not implemented in cuda_exp version. Fall back to boosting on CPU.");
      return new RegressionTweedieLoss(config);
    } else if (type == std::string("custom")) {
      Log::Warning("Using customized objective with cuda_exp. This requires copying gradients from CPU to GPU, which can be slow.");
      return nullptr;
    }
  } else {
  #endif  // USE_CUDA_EXP
    if (type == std::string("regression")) {
      return new RegressionL2loss(config);
    } else if (type == std::string("regression_l1")) {
      return new RegressionL1loss(config);
    } else if (type == std::string("quantile")) {
      return new RegressionQuantileloss(config);
    } else if (type == std::string("huber")) {
      return new RegressionHuberLoss(config);
    } else if (type == std::string("fair")) {
      return new RegressionFairLoss(config);
    } else if (type == std::string("poisson")) {
      return new RegressionPoissonLoss(config);
    } else if (type == std::string("binary")) {
      return new BinaryLogloss(config);
    } else if (type == std::string("lambdarank")) {
      return new LambdarankNDCG(config);
    } else if (type == std::string("rank_xendcg")) {
      return new RankXENDCG(config);
    } else if (type == std::string("multiclass")) {
      return new MulticlassSoftmax(config);
    } else if (type == std::string("multiclassova")) {
      return new MulticlassOVA(config);
    } else if (type == std::string("cross_entropy")) {
      return new CrossEntropy(config);
    } else if (type == std::string("cross_entropy_lambda")) {
      return new CrossEntropyLambda(config);
    } else if (type == std::string("mape")) {
      return new RegressionMAPELOSS(config);
    } else if (type == std::string("gamma")) {
      return new RegressionGammaLoss(config);
    } else if (type == std::string("tweedie")) {
      return new RegressionTweedieLoss(config);
    } else if (type == std::string("custom")) {
      return nullptr;
    }
  #ifdef USE_CUDA_EXP
  }
  #endif  // USE_CUDA_EXP
  Log::Fatal("Unknown objective type name: %s", type.c_str());
  return nullptr;
}

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& str) {
  auto strs = Common::Split(str.c_str(), ' ');
  auto type = strs[0];
  if (type == std::string("regression")) {
    return new RegressionL2loss(strs);
  } else if (type == std::string("regression_l1")) {
    return new RegressionL1loss(strs);
  } else if (type == std::string("quantile")) {
    return new RegressionQuantileloss(strs);
  } else if (type == std::string("huber")) {
    return new RegressionHuberLoss(strs);
  } else if (type == std::string("fair")) {
    return new RegressionFairLoss(strs);
  } else if (type == std::string("poisson")) {
    return new RegressionPoissonLoss(strs);
  } else if (type == std::string("binary")) {
    return new BinaryLogloss(strs);
  } else if (type == std::string("lambdarank")) {
    return new LambdarankNDCG(strs);
  } else if (type == std::string("rank_xendcg")) {
    return new RankXENDCG(strs);
  } else if (type == std::string("multiclass")) {
    return new MulticlassSoftmax(strs);
  } else if (type == std::string("multiclassova")) {
    return new MulticlassOVA(strs);
  } else if (type == std::string("cross_entropy")) {
    return new CrossEntropy(strs);
  } else if (type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambda(strs);
  } else if (type == std::string("mape")) {
    return new RegressionMAPELOSS(strs);
  } else if (type == std::string("gamma")) {
    return new RegressionGammaLoss(strs);
  } else if (type == std::string("tweedie")) {
    return new RegressionTweedieLoss(strs);
  } else if (type == std::string("custom")) {
    return nullptr;
  }
  Log::Fatal("Unknown objective type name: %s", type.c_str());
  return nullptr;
}

}  // namespace LightGBM
