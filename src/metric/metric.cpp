#include <LightGBM/metric.h>
#include "regression_metric.hpp"
#include "binary_metric.hpp"
#include "rank_metric.hpp"
#include "map_metric.hpp"
#include "multiclass_metric.hpp"
#include "xentropy_metric.hpp"

namespace LightGBM {

Metric* Metric::CreateMetric(const std::string& type, const MetricConfig& config) {
  if (type == std::string("regression") || type == std::string("regression_l2") || type == std::string("l2") || type == std::string("mean_squared_error") || type == std::string("mse")) {
    return new L2Metric(config);
  } else if (type == std::string("l2_root") || type == std::string("root_mean_squared_error") || type == std::string("rmse")) {
    return new RMSEMetric(config);
  } else if (type == std::string("regression_l1") || type == std::string("l1") || type == std::string("mean_absolute_error") || type == std::string("mae")) {
    return new L1Metric(config);
  } else if (type == std::string("quantile")) {
    return new QuantileMetric(config);
  } else if (type == std::string("huber")) {
    return new HuberLossMetric(config);
  } else if (type == std::string("fair")) {
    return new FairLossMetric(config);
  } else if (type == std::string("poisson")) {
    return new PoissonMetric(config);
  } else if (type == std::string("binary_logloss") || type == std::string("binary")) {
    return new BinaryLoglossMetric(config);
  } else if (type == std::string("binary_error")) {
    return new BinaryErrorMetric(config);
  } else if (type == std::string("auc")) {
    return new AUCMetric(config);
  } else if (type == std::string("ndcg")) {
    return new NDCGMetric(config);
  } else if (type == std::string("map") || type == std::string("mean_average_precision")) {
    return new MapMetric(config);
  } else if (type == std::string("multi_logloss") || type == std::string("multiclass") || type == std::string("softmax") || type == std::string("multiclassova") || type == std::string("multiclass_ova") || type == std::string("ova") || type == std::string("ovr")) {
    return new MultiSoftmaxLoglossMetric(config);
  } else if (type == std::string("multi_error")) {
    return new MultiErrorMetric(config);
  } else if (type == std::string("xentropy") || type == std::string("cross_entropy")) {
    return new CrossEntropyMetric(config);
  } else if (type == std::string("xentlambda") || type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambdaMetric(config);
  } else if (type == std::string("kldiv") || type == std::string("kullback_leibler")) {
    return new KullbackLeiblerDivergence(config);
  } else if (type == std::string("mean_absolute_percentage_error") || type == std::string("mape")) {
    return new MAPEMetric(config);
  } else if (type == std::string("gamma")) {
    return new GammaMetric(config);
  } else if (type == std::string("gamma_deviance")) {
    return new GammaDevianceMetric(config);
  } else if (type == std::string("tweedie")) {
    return new TweedieMetric(config);
  }
  return nullptr;
}

}  // namespace LightGBM
