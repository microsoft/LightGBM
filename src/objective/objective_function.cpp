#include <LightGBM/objective_function.h>
#include "regression_objective.hpp"
#include "binary_objective.hpp"
#include "rank_objective.hpp"
#include "multiclass_objective.hpp"
#include "xentropy_objective.hpp"

namespace LightGBM {

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& type, const Config& config) {
  if (type == std::string("regression") || type == std::string("regression_l2")
      || type == std::string("mean_squared_error") || type == std::string("mse") 
      || type == std::string("l2_root") || type == std::string("root_mean_squared_error") || type == std::string("rmse")) {
    return new RegressionL2loss(config);
  } else if (type == std::string("regression_l1") || type == std::string("mean_absolute_error")  || type == std::string("mae")) {
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
  } else if (type == std::string("multiclass") || type == std::string("softmax")) {
    return new MulticlassSoftmax(config);
  } else if (type == std::string("multiclassova") || type == std::string("multiclass_ova") || type == std::string("ova") || type == std::string("ovr")) {
    return new MulticlassOVA(config);
  } else if (type == std::string("xentropy") || type == std::string("cross_entropy")) {
    return new CrossEntropy(config);
  } else if (type == std::string("xentlambda") || type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambda(config);
  } else if (type == std::string("mean_absolute_percentage_error") || type == std::string("mape")) {
    return new RegressionMAPELOSS(config);
  } else if (type == std::string("gamma")) {
    return new RegressionGammaLoss(config);
  } else if (type == std::string("tweedie")) {
    return new RegressionTweedieLoss(config);
  } else if (type == std::string("none") || type == std::string("null") || type == std::string("custom") || type == std::string("na")) {
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
  } else if (type == std::string("xentropy") || type == std::string("cross_entropy")) {
    return new CrossEntropy(strs);
  } else if (type == std::string("xentlambda") || type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambda(strs);
  } else if (type == std::string("mean_absolute_percentage_error") || type == std::string("mape")) {
    return new RegressionMAPELOSS(strs);
  } else if (type == std::string("gamma")) {
    return new RegressionGammaLoss(strs);
  } else if (type == std::string("tweedie")) {
    return new RegressionTweedieLoss(strs);
  } else if (type == std::string("none") || type == std::string("null") || type == std::string("custom") || type == std::string("na")) {
    return nullptr;
  }
  Log::Fatal("Unknown objective type name: %s", type.c_str());
}

}  // namespace LightGBM
