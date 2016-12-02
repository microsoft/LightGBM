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
#include <stdexcept>
#include <mutex>

#include "./application/predictor.hpp"

namespace LightGBM {

class Booster {
public:
  explicit Booster(const char* filename) {
    boosting_.reset(Boosting::CreateBoosting(filename));
  }

  Booster(const Dataset* train_data, 
    const char* parameters) {
    auto param = ConfigBase::Str2Map(parameters);
    config_.Set(param);
    // create boosting
    if (config_.io_config.input_model.size() > 0) {
      Log::Warning("continued train from model is not support for c_api, \
        please use continued train with input score");
    }
    boosting_.reset(Boosting::CreateBoosting(config_.boosting_type, nullptr));
    ConstructObjectAndTrainingMetrics(train_data);
    // initialize the boosting
    boosting_->Init(&config_.boosting_config, train_data, objective_fun_.get(),
      Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
  }

  void MergeFrom(const Booster* other) {
    std::lock_guard<std::mutex> lock(mutex_);
    boosting_->MergeFrom(other->boosting_.get());
  }

  ~Booster() {

  }

  void ResetTrainingData(const Dataset* train_data) {
    std::lock_guard<std::mutex> lock(mutex_);
    train_data_ = train_data;
    ConstructObjectAndTrainingMetrics(train_data_);
    // initialize the boosting
    boosting_->ResetTrainingData(&config_.boosting_config, train_data_, 
      objective_fun_.get(), Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
  }

  void ResetConfig(const char* parameters) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto param = ConfigBase::Str2Map(parameters);
    if (param.count("num_class")) {
      Log::Fatal("cannot change num class during training");
    }
    if (param.count("boosting_type")) {
      Log::Fatal("cannot change boosting_type during training");
    }
    config_.Set(param);
    if (param.size() == 1 && (param.count("learning_rate") || param.count("shrinkage_rate"))) {
      // only need to set learning rate
      boosting_->ResetShrinkageRate(config_.boosting_config.learning_rate);
    } else {
      ResetTrainingData(train_data_);
    }
  }

  void AddValidData(const Dataset* valid_data) {
    std::lock_guard<std::mutex> lock(mutex_);
    valid_metrics_.emplace_back();
    for (auto metric_type : config_.metric_types) {
      auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_.metric_config));
      if (metric == nullptr) { continue; }
      metric->Init(valid_data->metadata(), valid_data->num_data());
      valid_metrics_.back().push_back(std::move(metric));
    }
    valid_metrics_.back().shrink_to_fit();
    boosting_->AddValidDataset(valid_data,
      Common::ConstPtrInVectorWrapper<Metric>(valid_metrics_.back()));
  }
  bool TrainOneIter() {
    std::lock_guard<std::mutex> lock(mutex_);
    return boosting_->TrainOneIter(nullptr, nullptr, false);
  }

  bool TrainOneIter(const float* gradients, const float* hessians) {
    std::lock_guard<std::mutex> lock(mutex_);
    return boosting_->TrainOneIter(gradients, hessians, false);
  }

  void RollbackOneIter() {
    std::lock_guard<std::mutex> lock(mutex_);
    boosting_->RollbackOneIter();
  }

  void PrepareForPrediction(int num_iteration, int predict_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    boosting_->SetNumIterationForPred(num_iteration);
    bool is_predict_leaf = false;
    bool is_raw_score = false;
    if (predict_type == C_API_PREDICT_LEAF_INDEX) {
      is_predict_leaf = true;
    } else if (predict_type == C_API_PREDICT_RAW_SCORE) {
      is_raw_score = true;
    } else {
      is_raw_score = false;
    }
    predictor_.reset(new Predictor(boosting_.get(), is_raw_score, is_predict_leaf));
  }

  void GetPredictAt(int data_idx, score_t* out_result, data_size_t* out_len) {
    boosting_->GetPredictAt(data_idx, out_result, out_len);
  }

  std::vector<double> Predict(const std::vector<std::pair<int, double>>& features) {
    return predictor_->GetPredictFunction()(features);
  }

  void PredictForFile(const char* data_filename, const char* result_filename, bool data_has_header) {
    predictor_->Predict(data_filename, result_filename, data_has_header);
  }

  void SaveModelToFile(int num_iteration, const char* filename) {
    boosting_->SaveModelToFile(num_iteration, filename);
  }

  std::string DumpModel() {
    return boosting_->DumpModel();
  }

  int GetEvalCounts() const {
    int ret = 0;
    for (const auto& metric : train_metric_) {
      ret += static_cast<int>(metric->GetName().size());
    }
    return ret;
  }

  int GetEvalNames(char** out_strs) const {
    int idx = 0;
    for (const auto& metric : train_metric_) {
      for (const auto& name : metric->GetName()) {
        std::strcpy(out_strs[idx], name.c_str());
        ++idx;
      }
    }
    return idx;
  }

  const Boosting* GetBoosting() const { return boosting_.get(); }
  
private:

  void ConstructObjectAndTrainingMetrics(const Dataset* train_data) {
    // create objective function
    objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction(config_.objective_type,
      config_.objective_config));
    if (objective_fun_ == nullptr) {
      Log::Warning("Using self-defined objective functions");
    }
    // create training metric
    train_metric_.clear();
    for (auto metric_type : config_.metric_types) {
      auto metric = std::unique_ptr<Metric>(
        Metric::CreateMetric(metric_type, config_.metric_config));
      if (metric == nullptr) { continue; }
      metric->Init(train_data->metadata(), train_data->num_data());
      train_metric_.push_back(std::move(metric));
    }
    train_metric_.shrink_to_fit();
    // initialize the objective function
    if (objective_fun_ != nullptr) {
      objective_fun_->Init(train_data->metadata(), train_data->num_data());
    }
  }

  const Dataset* train_data_;
  std::unique_ptr<Boosting> boosting_;
  /*! \brief All configs */
  OverallConfig config_;
  /*! \brief Metric for training data */
  std::vector<std::unique_ptr<Metric>> train_metric_;
  /*! \brief Metrics for validation data */
  std::vector<std::vector<std::unique_ptr<Metric>>> valid_metrics_;
  /*! \brief Training objective function */
  std::unique_ptr<ObjectiveFunction> objective_fun_;
  /*! \brief Using predictor for prediction task */
  std::unique_ptr<Predictor> predictor_;
  /*! \brief mutex for threading safe call */
  std::mutex mutex_;
};

}

using namespace LightGBM;

DllExport const char* LGBM_GetLastError() {
  return LastErrorMsg();
}

DllExport int LGBM_DatasetCreateFromFile(const char* filename,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  IOConfig io_config;
  io_config.Set(param);
  DatasetLoader loader(io_config, nullptr);
  loader.SetHeader(filename);
  if (reference == nullptr) {
    *out = loader.LoadFromFile(filename);
  } else {
    *out = loader.LoadFromFileAlignWithOtherDataset(filename,
      reinterpret_cast<const Dataset*>(*reference));
  }
  API_END();
}

DllExport int LGBM_DatasetCreateFromMat(const void* data,
  int data_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  IOConfig io_config;
  io_config.Set(param);
  DatasetLoader loader(io_config, nullptr);
  std::unique_ptr<Dataset> ret;
  auto get_row_fun = RowFunctionFromDenseMatric(data, nrow, ncol, data_type, is_row_major);
  if (reference == nullptr) {
    // sample data first
    Random rand(io_config.data_random_seed);
    const int sample_cnt = static_cast<int>(nrow < io_config.bin_construct_sample_cnt ? nrow : io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    std::vector<std::vector<double>> sample_values(ncol);
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      auto idx = sample_indices[i];
      auto row = get_row_fun(static_cast<int>(idx));
      for (size_t j = 0; j < row.size(); ++j) {
        if (std::fabs(row[j]) > 1e-15) {
          sample_values[j].push_back(row[j]);
        }
      }
    }
    ret.reset(loader.CostructFromSampleData(sample_values, sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow, io_config.num_class));
    ret->CopyFeatureMapperFrom(
      reinterpret_cast<const Dataset*>(*reference),
      io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nrow; ++i) {
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
  }
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

DllExport int LGBM_DatasetCreateFromCSR(const void* indptr,
  int indptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t nindptr,
  int64_t nelem,
  int64_t num_col,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  IOConfig io_config;
  io_config.Set(param);
  DatasetLoader loader(io_config, nullptr);
  std::unique_ptr<Dataset> ret;
  auto get_row_fun = RowFunctionFromCSR(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  int32_t nrow = static_cast<int32_t>(nindptr - 1);
  if (reference == nullptr) {
    // sample data first
    Random rand(io_config.data_random_seed);
    const int sample_cnt = static_cast<int>(nrow < io_config.bin_construct_sample_cnt ? nrow : io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    std::vector<std::vector<double>> sample_values;
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      auto idx = sample_indices[i];
      auto row = get_row_fun(static_cast<int>(idx));
      for (std::pair<int, double>& inner_data : row) {
        if (std::fabs(inner_data.second) > 1e-15) {
          if (static_cast<size_t>(inner_data.first) >= sample_values.size()) {
            // if need expand feature set
            size_t need_size = inner_data.first - sample_values.size() + 1;
            for (size_t j = 0; j < need_size; ++j) {
              sample_values.emplace_back();
            }
          }
          // edit the feature value
          sample_values[inner_data.first].push_back(inner_data.second);
        }
      }
    }
    CHECK(num_col >= static_cast<int>(sample_values.size()));
    ret.reset(loader.CostructFromSampleData(sample_values, sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow, io_config.num_class));
    ret->CopyFeatureMapperFrom(
      reinterpret_cast<const Dataset*>(*reference),
      io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nindptr - 1; ++i) {
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
  }
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

DllExport int LGBM_DatasetCreateFromCSC(const void* col_ptr,
  int col_ptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t ncol_ptr,
  int64_t nelem,
  int64_t num_row,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  IOConfig io_config;
  io_config.Set(param);
  DatasetLoader loader(io_config, nullptr);
  std::unique_ptr<Dataset> ret;
  auto get_col_fun = ColumnFunctionFromCSC(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem);
  int32_t nrow = static_cast<int32_t>(num_row);
  if (reference == nullptr) {
    Log::Warning("Construct from CSC format is not efficient");
    // sample data first
    Random rand(io_config.data_random_seed);
    const int sample_cnt = static_cast<int>(nrow < io_config.bin_construct_sample_cnt ? nrow : io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    std::vector<std::vector<double>> sample_values(ncol_ptr - 1);
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < static_cast<int>(sample_values.size()); ++i) {
      auto cur_col = get_col_fun(i);
      sample_values[i] = SampleFromOneColumn(cur_col, sample_indices);
    }
    ret.reset(loader.CostructFromSampleData(sample_values, sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow, io_config.num_class));
    ret->CopyFeatureMapperFrom(
      reinterpret_cast<const Dataset*>(*reference),
      io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < ncol_ptr - 1; ++i) {
    const int tid = omp_get_thread_num();
    auto one_col = get_col_fun(i);
    ret->PushOneColumn(tid, i, one_col);
  }
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

DllExport int LGBM_DatasetGetSubset(
  const DatesetHandle* handle,
  const int32_t* used_row_indices,
  int32_t num_used_row_indices,
  const char* parameters,
  DatesetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  IOConfig io_config;
  io_config.Set(param);
  auto full_dataset = reinterpret_cast<const Dataset*>(*handle);
  auto ret = std::unique_ptr<Dataset>(
    full_dataset->Subset(used_row_indices,
      num_used_row_indices, 
      io_config.is_enable_sparse));
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

DllExport int LGBM_DatasetFree(DatesetHandle handle) {
  API_BEGIN();
  delete reinterpret_cast<Dataset*>(handle);
  API_END();
}

DllExport int LGBM_DatasetSaveBinary(DatesetHandle handle,
  const char* filename) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  dataset->SaveBinaryFile(filename);
  API_END();
}

DllExport int LGBM_DatasetSetField(DatesetHandle handle,
  const char* field_name,
  const void* field_data,
  int64_t num_element,
  int type) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  bool is_success = false;
  if (type == C_API_DTYPE_FLOAT32) {
    is_success = dataset->SetFloatField(field_name, reinterpret_cast<const float*>(field_data), static_cast<int32_t>(num_element));
  } else if (type == C_API_DTYPE_INT32) {
    is_success = dataset->SetIntField(field_name, reinterpret_cast<const int*>(field_data), static_cast<int32_t>(num_element));
  }
  if (!is_success) { throw std::runtime_error("Input data type erorr or field not found"); }
  API_END();
}

DllExport int LGBM_DatasetGetField(DatesetHandle handle,
  const char* field_name,
  int64_t* out_len,
  const void** out_ptr,
  int* out_type) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  bool is_success = false;
  if (dataset->GetFloatField(field_name, out_len, reinterpret_cast<const float**>(out_ptr))) {
    *out_type = C_API_DTYPE_FLOAT32;
    is_success = true;
  } else if (dataset->GetIntField(field_name, out_len, reinterpret_cast<const int**>(out_ptr))) {
    *out_type = C_API_DTYPE_INT32;
    is_success = true;
  }
  if (!is_success) { throw std::runtime_error("Field not found"); }
  if (*out_ptr == nullptr) { *out_len = 0; }
  API_END();
}

DllExport int LGBM_DatasetGetNumData(DatesetHandle handle,
  int64_t* out) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  *out = dataset->num_data();
  API_END();
}

DllExport int LGBM_DatasetGetNumFeature(DatesetHandle handle,
  int64_t* out) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  *out = dataset->num_total_features();
  API_END();
}


// ---- start of booster

DllExport int LGBM_BoosterCreate(const DatesetHandle train_data,
  const char* parameters,
  BoosterHandle* out) {
  API_BEGIN();
  const Dataset* p_train_data = reinterpret_cast<const Dataset*>(train_data);
  auto ret = std::unique_ptr<Booster>(new Booster(p_train_data, parameters));
  *out = ret.release();
  API_END();
}

DllExport int LGBM_BoosterCreateFromModelfile(
  const char* filename,
  int64_t* out_num_iterations,
  BoosterHandle* out) {
  API_BEGIN();
  auto ret = std::unique_ptr<Booster>(new Booster(filename));
  *out_num_iterations = static_cast<int64_t>(ret->GetBoosting()->NumberOfTotalModel()
    / ret->GetBoosting()->NumberOfClasses());
  *out = ret.release();
  API_END();
}

DllExport int LGBM_BoosterFree(BoosterHandle handle) {
  API_BEGIN();
  delete reinterpret_cast<Booster*>(handle);
  API_END();
}

DllExport int LGBM_BoosterMerge(BoosterHandle handle,
  BoosterHandle other_handle) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  Booster* ref_other_booster = reinterpret_cast<Booster*>(other_handle);
  ref_booster->MergeFrom(ref_other_booster);
  API_END();
}

DllExport int LGBM_BoosterAddValidData(BoosterHandle handle,
  const DatesetHandle valid_data) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  const Dataset* p_dataset = reinterpret_cast<const Dataset*>(valid_data);
  ref_booster->AddValidData(p_dataset);
  API_END();
}

DllExport int LGBM_BoosterResetTrainingData(BoosterHandle handle,
  const DatesetHandle train_data) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  const Dataset* p_dataset = reinterpret_cast<const Dataset*>(train_data);
  ref_booster->ResetTrainingData(p_dataset);
  API_END();
}

DllExport int LGBM_BoosterResetParameter(BoosterHandle handle, const char* parameters) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->ResetConfig(parameters);
  API_END();
}

DllExport int LGBM_BoosterGetNumClasses(BoosterHandle handle, int64_t* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetBoosting()->NumberOfClasses();
  API_END();
}

DllExport int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  if (ref_booster->TrainOneIter()) {
    *is_finished = 1;
  } else {
    *is_finished = 0;
  }
  API_END();
}

DllExport int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
  const float* grad,
  const float* hess,
  int* is_finished) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  if (ref_booster->TrainOneIter(grad, hess)) {
    *is_finished = 1;
  } else {
    *is_finished = 0;
  }
  API_END();
}

DllExport int LGBM_BoosterRollbackOneIter(BoosterHandle handle) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->RollbackOneIter();
  API_END();
}

DllExport int LGBM_BoosterGetCurrentIteration(BoosterHandle handle, int64_t* out_iteration) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_iteration = ref_booster->GetBoosting()->GetCurrentIteration();
  API_END();
}
/*!
* \brief Get number of eval
* \return total number of eval result
*/
DllExport int LGBM_BoosterGetEvalCounts(BoosterHandle handle, int64_t* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetEvalCounts();
  API_END();
}

/*!
* \brief Get number of eval
* \return total number of eval result
*/
DllExport int LGBM_BoosterGetEvalNames(BoosterHandle handle, int64_t* out_len, char** out_strs) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetEvalNames(out_strs);
  API_END();
}


DllExport int LGBM_BoosterGetEval(BoosterHandle handle,
  int data_idx,
  int64_t* out_len,
  float* out_results) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto boosting = ref_booster->GetBoosting();
  auto result_buf = boosting->GetEvalAt(data_idx);
  *out_len = static_cast<int64_t>(result_buf.size());
  for (size_t i = 0; i < result_buf.size(); ++i) {
    (out_results)[i] = static_cast<float>(result_buf[i]);
  }
  API_END();
}

DllExport int LGBM_BoosterGetPredict(BoosterHandle handle,
  int data_idx,
  int64_t* out_len,
  float* out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  int len = 0;
  ref_booster->GetPredictAt(data_idx, out_result, &len);
  *out_len = static_cast<int64_t>(len);
  API_END();
}

DllExport int LGBM_BoosterPredictForFile(BoosterHandle handle,
  const char* data_filename,
  int data_has_header,
  int predict_type,
  int64_t num_iteration,
  const char* result_filename) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->PrepareForPrediction(static_cast<int>(num_iteration), predict_type);
  bool bool_data_has_header = data_has_header > 0 ? true : false;
  ref_booster->PredictForFile(data_filename, result_filename, bool_data_has_header);
  API_END();
}

DllExport int LGBM_BoosterPredictForCSR(BoosterHandle handle,
  const void* indptr,
  int indptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t nindptr,
  int64_t nelem,
  int64_t,
  int predict_type,
  int64_t num_iteration,
  int64_t* out_len,
  float* out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->PrepareForPrediction(static_cast<int>(num_iteration), predict_type);

  auto get_row_fun = RowFunctionFromCSR(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  int num_preb_in_one_row = ref_booster->GetBoosting()->NumberOfClasses();
  if (predict_type == C_API_PREDICT_LEAF_INDEX) {
    if (num_iteration > 0) {
      num_preb_in_one_row *= static_cast<int>(num_iteration);
    } else {
      num_preb_in_one_row *= ref_booster->GetBoosting()->NumberOfTotalModel() / num_preb_in_one_row;
    }
  }
  int nrow = static_cast<int>(nindptr - 1);
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nrow; ++i) {
    auto one_row = get_row_fun(i);
    auto predicton_result = ref_booster->Predict(one_row);
    for (int j = 0; j < static_cast<int>(predicton_result.size()); ++j) {
      out_result[i * num_preb_in_one_row + j] = static_cast<float>(predicton_result[j]);
    }
  }
  *out_len = nrow * num_preb_in_one_row;
  API_END();
}

DllExport int LGBM_BoosterPredictForMat(BoosterHandle handle,
  const void* data,
  int data_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  int predict_type,
  int64_t num_iteration,
  int64_t* out_len,
  float* out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->PrepareForPrediction(static_cast<int>(num_iteration), predict_type);

  auto get_row_fun = RowPairFunctionFromDenseMatric(data, nrow, ncol, data_type, is_row_major);
  int num_preb_in_one_row = ref_booster->GetBoosting()->NumberOfClasses();
  if (predict_type == C_API_PREDICT_LEAF_INDEX) {
    if (num_iteration > 0) {
      num_preb_in_one_row *= static_cast<int>(num_iteration);
    } else {
      num_preb_in_one_row *= ref_booster->GetBoosting()->NumberOfTotalModel() / num_preb_in_one_row;
    }
  }
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nrow; ++i) {
    auto one_row = get_row_fun(i);
    auto predicton_result = ref_booster->Predict(one_row);
    for (int j = 0; j < static_cast<int>(predicton_result.size()); ++j) {
      out_result[i * num_preb_in_one_row + j] = static_cast<float>(predicton_result[j]);
    }
  }
  *out_len = nrow * num_preb_in_one_row;
  API_END();
}

DllExport int LGBM_BoosterSaveModel(BoosterHandle handle,
  int num_iteration,
  const char* filename) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->SaveModelToFile(num_iteration, filename);
  API_END();
}

DllExport int LGBM_BoosterDumpModel(BoosterHandle handle,
  int buffer_len,
  int64_t* out_len,
  char** out_str) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  std::string model = ref_booster->DumpModel();
  *out_len = static_cast<int64_t>(model.size());
  if (*out_len <= buffer_len) {
    std::strcpy(*out_str, model.c_str());
  }
  API_END();
}

// ---- start of some help functions

std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major) {
  if (data_type == C_API_DTYPE_FLOAT32) {
    const float* data_ptr = reinterpret_cast<const float*>(data);
    if (is_row_major) {
      return [data_ptr, num_col, num_row](int row_idx) {
        std::vector<double> ret(num_col);
        auto tmp_ptr = data_ptr + num_col * row_idx;
        for (int i = 0; i < num_col; ++i) {
          ret[i] = static_cast<double>(*(tmp_ptr + i));
        }
        return ret;
      };
    } else {
      return [data_ptr, num_col, num_row](int row_idx) {
        std::vector<double> ret(num_col);
        for (int i = 0; i < num_col; ++i) {
          ret[i] = static_cast<double>(*(data_ptr + num_row * i + row_idx));
        }
        return ret;
      };
    }
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    const double* data_ptr = reinterpret_cast<const double*>(data);
    if (is_row_major) {
      return [data_ptr, num_col, num_row](int row_idx) {
        std::vector<double> ret(num_col);
        auto tmp_ptr = data_ptr + num_col * row_idx;
        for (int i = 0; i < num_col; ++i) {
          ret[i] = static_cast<double>(*(tmp_ptr + i));
        }
        return ret;
      };
    } else {
      return [data_ptr, num_col, num_row](int row_idx) {
        std::vector<double> ret(num_col);
        for (int i = 0; i < num_col; ++i) {
          ret[i] = static_cast<double>(*(data_ptr + num_row * i + row_idx));
        }
        return ret;
      };
    }
  }
  throw std::runtime_error("unknown data type in RowFunctionFromDenseMatric");
}

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major) {
  auto inner_function = RowFunctionFromDenseMatric(data, num_row, num_col, data_type, is_row_major);
  if (inner_function != nullptr) {
    return [inner_function](int row_idx) {
      auto raw_values = inner_function(row_idx);
      std::vector<std::pair<int, double>> ret;
      for (int i = 0; i < static_cast<int>(raw_values.size()); ++i) {
        if (std::fabs(raw_values[i]) > 1e-15) {
          ret.emplace_back(i, raw_values[i]);
        }
      }
      return ret;
    };
  }
  return nullptr;
}

std::function<std::vector<std::pair<int, double>>(int idx)>
RowFunctionFromCSR(const void* indptr, int indptr_type, const int32_t* indices, const void* data, int data_type, int64_t nindptr, int64_t nelem) {
  if (data_type == C_API_DTYPE_FLOAT32) {
    const float* data_ptr = reinterpret_cast<const float*>(data);
    if (indptr_type == C_API_DTYPE_INT32) {
      const int32_t* ptr_indptr = reinterpret_cast<const int32_t*>(indptr);
      return [ptr_indptr, indices, data_ptr, nindptr, nelem](int idx) {
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_indptr[idx];
        int64_t end = ptr_indptr[idx + 1];
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else if (indptr_type == C_API_DTYPE_INT64) {
      const int64_t* ptr_indptr = reinterpret_cast<const int64_t*>(indptr);
      return [ptr_indptr, indices, data_ptr, nindptr, nelem](int idx) {
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_indptr[idx];
        int64_t end = ptr_indptr[idx + 1];
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    }
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    const double* data_ptr = reinterpret_cast<const double*>(data);
    if (indptr_type == C_API_DTYPE_INT32) {
      const int32_t* ptr_indptr = reinterpret_cast<const int32_t*>(indptr);
      return [ptr_indptr, indices, data_ptr, nindptr, nelem](int idx) {
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_indptr[idx];
        int64_t end = ptr_indptr[idx + 1];
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else if (indptr_type == C_API_DTYPE_INT64) {
      const int64_t* ptr_indptr = reinterpret_cast<const int64_t*>(indptr);
      return [ptr_indptr, indices, data_ptr, nindptr, nelem](int idx) {
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_indptr[idx];
        int64_t end = ptr_indptr[idx + 1];
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } 
  } 
  throw std::runtime_error("unknown data type in RowFunctionFromCSR");
}

std::function<std::vector<std::pair<int, double>>(int idx)>
ColumnFunctionFromCSC(const void* col_ptr, int col_ptr_type, const int32_t* indices, const void* data, int data_type, int64_t ncol_ptr, int64_t nelem) {
  if (data_type == C_API_DTYPE_FLOAT32) {
    const float* data_ptr = reinterpret_cast<const float*>(data);
    if (col_ptr_type == C_API_DTYPE_INT32) {
      const int32_t* ptr_col_ptr = reinterpret_cast<const int32_t*>(col_ptr);
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem](int idx) {
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_col_ptr[idx];
        int64_t end = ptr_col_ptr[idx + 1];
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else if (col_ptr_type == C_API_DTYPE_INT64) {
      const int64_t* ptr_col_ptr = reinterpret_cast<const int64_t*>(col_ptr);
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem](int idx) {
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_col_ptr[idx];
        int64_t end = ptr_col_ptr[idx + 1];
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } 
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    const double* data_ptr = reinterpret_cast<const double*>(data);
    if (col_ptr_type == C_API_DTYPE_INT32) {
      const int32_t* ptr_col_ptr = reinterpret_cast<const int32_t*>(col_ptr);
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem](int idx) {
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_col_ptr[idx];
        int64_t end = ptr_col_ptr[idx + 1];
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else if (col_ptr_type == C_API_DTYPE_INT64) {
      const int64_t* ptr_col_ptr = reinterpret_cast<const int64_t*>(col_ptr);
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem](int idx) {
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_col_ptr[idx];
        int64_t end = ptr_col_ptr[idx + 1];
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } 
  } 
  throw std::runtime_error("unknown data type in ColumnFunctionFromCSC");
}

std::vector<double> SampleFromOneColumn(const std::vector<std::pair<int, double>>& data, const std::vector<int>& indices) {
  size_t j = 0;
  std::vector<double> ret;
  for (auto row_idx : indices) {
    while (j < data.size() && data[j].first < static_cast<int>(row_idx)) {
      ++j;
    }
    if (j < data.size() && data[j].first == static_cast<int>(row_idx)) {
      ret.push_back(data[j].second);
    }
  }
  return ret;
}
