#include <omp.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/random.h>
#include <LightGBM/c_api.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/dataset.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/config.h>

#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <memory>

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
      Log::Warning("continued train from model is not support for c_api, \
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

using namespace LightGBM;


DllExport const char* LGBM_GetLastError() {
  return "Not error msg now, will support soon";
}



DllExport int LGBM_CreateDatasetFromFile(const char* filename,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {

  OverallConfig config;
  config.LoadFromString(parameters);
  DatasetLoader loader(config.io_config, nullptr);
  if (reference == nullptr) {
    *out = loader.LoadFromFile(filename);
  } else {
    *out = loader.LoadFromFileLikeOthers(filename, reinterpret_cast<const Dataset*>(*reference));
  }
  return 0;
}


DllExport int LGBM_CreateDatasetFromBinaryFile(const char* filename,
  DatesetHandle* out) {

  OverallConfig config;
  DatasetLoader loader(config.io_config, nullptr);
  *out = loader.LoadFromBinFile(filename, 0, 1);
  return 0;
}

DllExport int LGBM_CreateDatasetFromMat(const void* data,
  int float_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {

  OverallConfig config;
  config.LoadFromString(parameters);
  DatasetLoader loader(config.io_config, nullptr);
  Dataset* ret = nullptr;
  auto get_row_fun = Common::GetRowFunctionFromMat(data, nrow, ncol, float_type, is_row_major);
  if (reference == nullptr) {
    // sample data first
    Random rand(config.io_config.data_random_seed);
    const size_t sample_cnt = static_cast<size_t>(nrow < config.io_config.bin_construct_sample_cnt ? nrow : config.io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    std::vector<std::vector<double>> sample_values(ncol);
    for (size_t i = 0; i < sample_indices.size(); i++) {
      auto idx = sample_indices[i];
      auto row = get_row_fun(static_cast<int>(idx));
      for (size_t j = 0; j < row.size(); j++) {
        sample_values[j].push_back(row[j]);
      }
    }
    ret = loader.CostructFromSampleData(sample_values, nrow);
  } else {
    ret = new Dataset();
    // need to set num_data first
    ret->SetNumData(nrow);
    reinterpret_cast<const Dataset*>(*reference)->CopyFeatureMetadataTo(ret, config.io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nrow; ++i) {
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
  }
  ret->FinishLoad();
  *out = ret;
  return 0;
}

DllExport int LGBM_CreateDatasetFromCSR(const int32_t* indptr,
  const int32_t* indices,
  const void* data,
  int float_type,
  uint64_t nindptr,
  uint64_t nelem,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {

  OverallConfig config;
  config.LoadFromString(parameters);
  DatasetLoader loader(config.io_config, nullptr);
  Dataset* ret = nullptr;
  auto get_row_fun = Common::GetRowFunctionFromCSR(indptr, indices, data, float_type, nindptr, nelem);
  int32_t nrow = static_cast<int32_t>(nindptr - 1);
  if (reference == nullptr) {
    // sample data first
    Random rand(config.io_config.data_random_seed);
    const size_t sample_cnt = static_cast<size_t>(nrow < config.io_config.bin_construct_sample_cnt ? nrow : config.io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    std::vector<std::vector<double>> sample_values;
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      auto idx = sample_indices[i];
      auto row = get_row_fun(static_cast<int>(idx));
      // push 0 first, then edit the value according existing feature values
      for (auto& feature_values : sample_values) {
        feature_values.push_back(0.0);
      }
      for (std::pair<int, double>& inner_data : row) {
        if (static_cast<size_t>(inner_data.first) >= sample_values.size()) {
          // if need expand feature set
          size_t need_size = inner_data.first - sample_values.size() + 1;
          for (size_t j = 0; j < need_size; ++j) {
            // push i+1 0
            sample_values.emplace_back(i + 1, 0.0f);
          }
        }
        // edit the feature value
        sample_values[inner_data.first][i] = inner_data.second;
      }
    }
    ret = loader.CostructFromSampleData(sample_values, nrow);
  } else {
    ret = new Dataset();
    // need to set num_data first
    ret->SetNumData(nrow);
    reinterpret_cast<const Dataset*>(*reference)->CopyFeatureMetadataTo(ret, config.io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nindptr - 1; ++i) {
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
  }
  ret->FinishLoad();
  *out = ret;

  return 0;
}

DllExport int LGBM_DatasetFree(DatesetHandle* handle) {
  auto dataset = reinterpret_cast<Dataset*>(*handle);
  delete dataset;
  return 0;
}

DllExport int LGBM_DatasetSaveBinary(DatesetHandle handle,
  const char* filename) {
  auto dataset = reinterpret_cast<Dataset*>(handle);
  dataset->SaveBinaryFile(filename);
  return 0;
}

DllExport int LGBM_DatasetSetField(DatesetHandle handle,
  const char* field_name,
  const void* field_data,
  uint64_t num_element,
  int type) {
  auto dataset = reinterpret_cast<Dataset*>(handle);
  dataset->SetField(field_name, field_data, num_element, type);
  return 0;
}


DllExport int LGBM_DatasetGetField(DatesetHandle handle,
  const char* field_name,
  uint64_t* out_len,
  const void** out_ptr,
  int* out_type) {
  auto dataset = reinterpret_cast<Dataset*>(handle);
  dataset->GetField(field_name, out_len, out_ptr, out_type);
  return 0;
}


DllExport int LGBM_DatasetGetNumData(DatesetHandle handle,
  uint64_t* out) {
  auto dataset = reinterpret_cast<Dataset*>(handle);
  *out = dataset->num_data();
  return 0;
}

DllExport int LGBM_DatasetGetNumFeature(DatesetHandle handle,
  uint64_t* out) {
  auto dataset = reinterpret_cast<Dataset*>(handle);
  *out = dataset->num_total_features();
  return 0;
}
