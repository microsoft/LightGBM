#include <LightGBM/objective_function.h>
#include "regression_objective.hpp"
#include "binary_objective.hpp"
#include "rank_objective.hpp"
#include "multiclass_objective.hpp"

namespace LightGBM {

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& type, const ObjectiveConfig& config) {
  if (type == "regression") {
    return new RegressionL2loss(config);
  } else if (type == "binary") {
    return new BinaryLogloss(config);
  } else if (type == "lambdarank") {
    return new LambdarankNDCG(config);
  } else if (type == "multiclass") {
    return new MulticlassLogloss(config);
  }
  return nullptr;
}
}  // namespace LightGBM
