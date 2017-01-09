#include <LightGBM/objective_function.h>
#include "regression_objective.hpp"
#include "binary_objective.hpp"
#include "rank_objective.hpp"
#include "multiclass_objective.hpp"

namespace LightGBM {

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& type, const ObjectiveConfig& config) {
  if (type == std::string("regression") || type == std::string("regression_l2")
      || type == std::string("mean_squared_error") || type == std::string("mse")) {
    return new RegressionL2loss(config);
  } else if (type == std::string("regression_l1") || type == std::string("mean_absolute_error")  || type == std::string("mae")) {
    return new RegressionL1loss(config);
  } else if (type == std::string("huber")) {
    return new RegressionHuberLoss(config);
  } else if (type == std::string("binary")) {
    return new BinaryLogloss(config);
  } else if (type == std::string("lambdarank")) {
    return new LambdarankNDCG(config);
  } else if (type == std::string("multiclass")) {
    return new MulticlassLogloss(config);
  }
  return nullptr;
}
}  // namespace LightGBM
