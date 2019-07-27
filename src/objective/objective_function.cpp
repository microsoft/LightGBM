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

namespace LightGBM {

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& type, const Config& config) {
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
  Log::Fatal("Unknown objective type name: %s", type.c_str());
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
}

}  // namespace LightGBM
