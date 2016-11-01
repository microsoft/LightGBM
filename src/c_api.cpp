
#include <LightGBM/c_api.h>
#include <LightGBM/dataset.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/config.h>

#include <cstdio>
#include <vector>
#include <string>
#include <cstring>

namespace LightGBM {

class Booster {
public:
  explicit Booster(const char* filename):
    boosting_(Boosting::CreateBoosting(filename)) {
  }

  Booster(const Dataset* train_data, 
    std::vector<const Dataset*> valid_data, 
    std::vector<std::string> valid_names,
    const char* parameters)
    :train_data_(train_data), valid_datas_(valid_data) {
    config_.LoadFromString(parameters);
    // create boosting
    if (config_.io_config.input_model.size() > 0) {
      Log::Error("continued train from model is not support for c_api, \
        please use continued train with input score");
    }
    boosting_ = Boosting::CreateBoosting(config_.boosting_type, "");
    // create objective function
    objective_fun_ =
      ObjectiveFunction::CreateObjectiveFunction(config_.objective_type,
        config_.objective_config);
    // create training metric
    if (config_.boosting_config->is_provide_training_metric) {
      for (auto metric_type : config_.metric_types) {
        Metric* metric =
          Metric::CreateMetric(metric_type, config_.metric_config);
        if (metric == nullptr) { continue; }
        metric->Init("training", train_data_->metadata(),
          train_data_->num_data());
        train_metric_.push_back(metric);
      }
    }
    // add metric for validation data
    for (size_t i = 0; i < valid_datas_.size(); ++i) {
      valid_metrics_.emplace_back();
      for (auto metric_type : config_.metric_types) {
        Metric* metric = Metric::CreateMetric(metric_type, config_.metric_config);
        if (metric == nullptr) { continue; }
        metric->Init(valid_names[i].c_str(),
          valid_datas_[i]->metadata(),
          valid_datas_[i]->num_data());
        valid_metrics_.back().push_back(metric);
      }
    }
    // initialize the objective function
    objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
    // initialize the boosting
    boosting_->Init(config_.boosting_config, train_data_, objective_fun_,
      ConstPtrInVectorWarpper<Metric>(train_metric_));
    // add validation data into boosting
    for (size_t i = 0; i < valid_datas_.size(); ++i) {
      boosting_->AddDataset(valid_datas_[i],
        ConstPtrInVectorWarpper<Metric>(valid_metrics_[i]));
    }
  }

  ~Booster() {
    for (auto& metric : train_metric_) {
      if (metric != nullptr) { delete metric; }
    }
    for (auto& metric : valid_metrics_) {
      for (auto& sub_metric : metric) {
        if (sub_metric != nullptr) { delete sub_metric; }
      }
    }
    valid_metrics_.clear();
    if (boosting_ != nullptr) { delete boosting_; }
    if (objective_fun_ != nullptr) { delete objective_fun_; }
  }
private:
  Boosting* boosting_;
  /*! \brief All configs */
  OverallConfig config_;
  /*! \brief Training data */
  const Dataset* train_data_;
  /*! \brief Validation data */
  std::vector<const Dataset*> valid_datas_;
  /*! \brief Metric for training data */
  std::vector<Metric*> train_metric_;
  /*! \brief Metrics for validation data */
  std::vector<std::vector<Metric*>> valid_metrics_;
  /*! \brief Training objective function */
  ObjectiveFunction* objective_fun_;
};

}
