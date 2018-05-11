#include <LightGBM/utils/openmp_wrapper.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/random.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/c_api.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/dataset.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/config.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/network.h>

#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <functional>

#include "./application/predictor.hpp"

namespace LightGBM {

class Booster {
public:
  explicit Booster(const char* filename) {
    boosting_.reset(Boosting::CreateBoosting("gbdt", filename));
  }

  Booster(const Dataset* train_data,
          const char* parameters) {
    CHECK(train_data->num_features() > 0);
    auto param = ConfigBase::Str2Map(parameters);
    config_.Set(param);
    if (config_.num_threads > 0) {
      omp_set_num_threads(config_.num_threads);
    }
    // create boosting
    if (config_.io_config.input_model.size() > 0) {
      Log::Warning("Continued train from model is not supported for c_api,\n"
                   "please use continued train with input score");
    }

    boosting_.reset(Boosting::CreateBoosting(config_.boosting_type, nullptr));

    train_data_ = train_data;
    CreateObjectiveAndMetrics();
    // initialize the boosting
    if (config_.boosting_config.tree_learner_type == std::string("feature")) {
      Log::Fatal("Do not support feature parallel in c api");
    }
    if (Network::num_machines() == 1 && config_.boosting_config.tree_learner_type != std::string("serial")) {
      Log::Warning("Only find one worker, will switch to serial tree learner");
      config_.boosting_config.tree_learner_type = "serial";
    }
    boosting_->Init(&config_.boosting_config, train_data_, objective_fun_.get(),
                    Common::ConstPtrInVectorWrapper<Metric>(train_metric_));

  }

  void MergeFrom(const Booster* other) {
    std::lock_guard<std::mutex> lock(mutex_);
    boosting_->MergeFrom(other->boosting_.get());
  }

  ~Booster() {

  }

  void CreateObjectiveAndMetrics() {
    // create objective function
    objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction(config_.objective_type,
                                                                    config_.objective_config));
    if (objective_fun_ == nullptr) {
      Log::Warning("Using self-defined objective function");
    }
    // initialize the objective function
    if (objective_fun_ != nullptr) {
      objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
    }

    // create training metric
    train_metric_.clear();
    for (auto metric_type : config_.metric_types) {
      auto metric = std::unique_ptr<Metric>(
        Metric::CreateMetric(metric_type, config_.metric_config));
      if (metric == nullptr) { continue; }
      metric->Init(train_data_->metadata(), train_data_->num_data());
      train_metric_.push_back(std::move(metric));
    }
    train_metric_.shrink_to_fit();
  }

  void ResetTrainingData(const Dataset* train_data) {
    if (train_data != train_data_) {
      CHECK(train_data->num_features() > 0);
      std::lock_guard<std::mutex> lock(mutex_);
      train_data_ = train_data;
      CreateObjectiveAndMetrics();
      // reset the boosting
      boosting_->ResetTrainingData(train_data_,
                                   objective_fun_.get(), Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
    }
  }

  void ResetConfig(const char* parameters) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto param = ConfigBase::Str2Map(parameters);
    if (param.count("num_class")) {
      Log::Fatal("Cannot change num_class during training");
    }
    if (param.count("boosting_type")) {
      Log::Fatal("Cannot change boosting_type during training");
    }
    if (param.count("metric")) {
      Log::Fatal("Cannot change metric during training");
    }

    config_.Set(param);
    if (config_.num_threads > 0) {
      omp_set_num_threads(config_.num_threads);
    }

    if (param.count("objective")) {
      // create objective function
      objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction(config_.objective_type,
                                                                      config_.objective_config));
      if (objective_fun_ == nullptr) {
        Log::Warning("Using self-defined objective function");
      }
      // initialize the objective function
      if (objective_fun_ != nullptr) {
        objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
      }
      boosting_->ResetTrainingData(train_data_,
                                   objective_fun_.get(), Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
    }

    boosting_->ResetConfig(&config_.boosting_config);

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
    return boosting_->TrainOneIter(nullptr, nullptr);
  }

  bool TrainOneIter(const score_t* gradients, const score_t* hessians) {
    std::lock_guard<std::mutex> lock(mutex_);
    return boosting_->TrainOneIter(gradients, hessians);
  }

  void RollbackOneIter() {
    std::lock_guard<std::mutex> lock(mutex_);
    boosting_->RollbackOneIter();
  }

  void Predict(int num_iteration, int predict_type, int nrow,
               std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun,
               const IOConfig& config,
               double* out_result, int64_t* out_len) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool is_predict_leaf = false;
    bool is_raw_score = false;
    bool is_predict_contrib = false;
    if (predict_type == C_API_PREDICT_LEAF_INDEX) {
      is_predict_leaf = true;
    } else if (predict_type == C_API_PREDICT_RAW_SCORE) {
      is_raw_score = true;
    } else if (predict_type == C_API_PREDICT_CONTRIB) {
      is_predict_contrib = true;
    } else {
      is_raw_score = false;
    }

    Predictor predictor(boosting_.get(), num_iteration, is_raw_score, is_predict_leaf, is_predict_contrib,
                        config.pred_early_stop, config.pred_early_stop_freq, config.pred_early_stop_margin);
    int64_t num_pred_in_one_row = boosting_->NumPredictOneRow(num_iteration, is_predict_leaf, is_predict_contrib);
    auto pred_fun = predictor.GetPredictFunction();
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nrow; ++i) {
      OMP_LOOP_EX_BEGIN();
      auto one_row = get_row_fun(i);
      auto pred_wrt_ptr = out_result + static_cast<size_t>(num_pred_in_one_row) * i;
      pred_fun(one_row, pred_wrt_ptr);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    *out_len = nrow * num_pred_in_one_row;
  }

  void Predict(int num_iteration, int predict_type, const char* data_filename,
               int data_has_header, const IOConfig& config,
               const char* result_filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool is_predict_leaf = false;
    bool is_raw_score = false;
    bool is_predict_contrib = false;
    if (predict_type == C_API_PREDICT_LEAF_INDEX) {
      is_predict_leaf = true;
    } else if (predict_type == C_API_PREDICT_RAW_SCORE) {
      is_raw_score = true;
    } else if (predict_type == C_API_PREDICT_CONTRIB) {
      is_predict_contrib = true;
    } else {
      is_raw_score = false;
    }
    Predictor predictor(boosting_.get(), num_iteration, is_raw_score, is_predict_leaf, is_predict_contrib,
                        config.pred_early_stop, config.pred_early_stop_freq, config.pred_early_stop_margin);
    bool bool_data_has_header = data_has_header > 0 ? true : false;
    predictor.Predict(data_filename, result_filename, bool_data_has_header);
  }

  void GetPredictAt(int data_idx, double* out_result, int64_t* out_len) {
    boosting_->GetPredictAt(data_idx, out_result, out_len);
  }

  void SaveModelToFile(int num_iteration, const char* filename) {
    boosting_->SaveModelToFile(num_iteration, filename);
  }

  void LoadModelFromString(const char* model_str) {
    size_t len = std::strlen(model_str);
    boosting_->LoadModelFromString(model_str, len);
  }

  std::string SaveModelToString(int num_iteration) {
    return boosting_->SaveModelToString(num_iteration);
  }

  std::string DumpModel(int num_iteration) {
    return boosting_->DumpModel(num_iteration);
  }

  std::vector<double> FeatureImportance(int num_iteration, int importance_type) {
    return boosting_->FeatureImportance(num_iteration, importance_type);
  }

  double GetLeafValue(int tree_idx, int leaf_idx) const {
    return dynamic_cast<GBDTBase*>(boosting_.get())->GetLeafValue(tree_idx, leaf_idx);
  }

  void SetLeafValue(int tree_idx, int leaf_idx, double val) {
    std::lock_guard<std::mutex> lock(mutex_);
    dynamic_cast<GBDTBase*>(boosting_.get())->SetLeafValue(tree_idx, leaf_idx, val);
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
        std::memcpy(out_strs[idx], name.c_str(), name.size() + 1);
        ++idx;
      }
    }
    return idx;
  }

  int GetFeatureNames(char** out_strs) const {
    int idx = 0;
    for (const auto& name : boosting_->FeatureNames()) {
      std::memcpy(out_strs[idx], name.c_str(), name.size() + 1);
      ++idx;
    }
    return idx;
  }

  const Boosting* GetBoosting() const { return boosting_.get(); }

private:

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
  /*! \brief mutex for threading safe call */
  std::mutex mutex_;
};

}

using namespace LightGBM;

// some help functions used to convert data

std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int idx)>
RowFunctionFromCSR(const void* indptr, int indptr_type, const int32_t* indices,
                   const void* data, int data_type, int64_t nindptr, int64_t nelem);

// Row iterator of on column for CSC matrix
class CSC_RowIterator {
public:
  CSC_RowIterator(const void* col_ptr, int col_ptr_type, const int32_t* indices,
                  const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx);
  ~CSC_RowIterator() {}
  // return value at idx, only can access by ascent order
  double Get(int idx);
  // return next non-zero pair, if index < 0, means no more data
  std::pair<int, double> NextNonZero();
private:
  int nonzero_idx_ = 0;
  int cur_idx_ = -1;
  double cur_val_ = 0.0f;
  bool is_end_ = false;
  std::function<std::pair<int, double>(int idx)> iter_fun_;
};

// start of c_api functions

const char* LGBM_GetLastError() {
  return LastErrorMsg();
}

int LGBM_DatasetCreateFromFile(const char* filename,
                               const char* parameters,
                               const DatasetHandle reference,
                               DatasetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  OverallConfig config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  DatasetLoader loader(config.io_config,nullptr, 1, filename);
  if (reference == nullptr) {
    if (Network::num_machines() == 1) {
      *out = loader.LoadFromFile(filename, "");
    } else {
      *out = loader.LoadFromFile(filename, "", Network::rank(), Network::num_machines());
    }
  } else {
    *out = loader.LoadFromFileAlignWithOtherDataset(filename, "",
                                                    reinterpret_cast<const Dataset*>(reference));
  }
  API_END();
}


int LGBM_DatasetCreateFromSampledColumn(double** sample_data,
                                        int** sample_indices,
                                        int32_t ncol,
                                        const int* num_per_col,
                                        int32_t num_sample_row,
                                        int32_t num_total_row,
                                        const char* parameters,
                                        DatasetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  OverallConfig config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  DatasetLoader loader(config.io_config, nullptr, 1, nullptr);
  *out = loader.CostructFromSampleData(sample_data, sample_indices, ncol, num_per_col,
                                       num_sample_row,
                                       static_cast<data_size_t>(num_total_row));
  API_END();
}


int LGBM_DatasetCreateByReference(const DatasetHandle reference,
                                  int64_t num_total_row,
                                  DatasetHandle* out) {
  API_BEGIN();
  std::unique_ptr<Dataset> ret;
  ret.reset(new Dataset(static_cast<data_size_t>(num_total_row)));
  ret->CreateValid(reinterpret_cast<const Dataset*>(reference));
  *out = ret.release();
  API_END();
}

int LGBM_DatasetPushRows(DatasetHandle dataset,
                         const void* data,
                         int data_type,
                         int32_t nrow,
                         int32_t ncol,
                         int32_t start_row) {
  API_BEGIN();
  auto p_dataset = reinterpret_cast<Dataset*>(dataset);
  auto get_row_fun = RowFunctionFromDenseMatric(data, nrow, ncol, data_type, 1);
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nrow; ++i) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    p_dataset->PushOneRow(tid, start_row + i, one_row);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  if (start_row + nrow == p_dataset->num_data()) {
    p_dataset->FinishLoad();
  }
  API_END();
}

int LGBM_DatasetPushRowsByCSR(DatasetHandle dataset,
                              const void* indptr,
                              int indptr_type,
                              const int32_t* indices,
                              const void* data,
                              int data_type,
                              int64_t nindptr,
                              int64_t nelem,
                              int64_t,
                              int64_t start_row) {
  API_BEGIN();
  auto p_dataset = reinterpret_cast<Dataset*>(dataset);
  auto get_row_fun = RowFunctionFromCSR(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  int32_t nrow = static_cast<int32_t>(nindptr - 1);
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nrow; ++i) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    p_dataset->PushOneRow(tid,
                          static_cast<data_size_t>(start_row + i), one_row);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  if (start_row + nrow == static_cast<int64_t>(p_dataset->num_data())) {
    p_dataset->FinishLoad();
  }
  API_END();
}

int LGBM_DatasetCreateFromMat(const void* data,
                              int data_type,
                              int32_t nrow,
                              int32_t ncol,
                              int is_row_major,
                              const char* parameters,
                              const DatasetHandle reference,
                              DatasetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  OverallConfig config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  std::unique_ptr<Dataset> ret;
  auto get_row_fun = RowFunctionFromDenseMatric(data, nrow, ncol, data_type, is_row_major);
  if (reference == nullptr) {
    // sample data first
    Random rand(config.io_config.data_random_seed);
    int sample_cnt = static_cast<int>(nrow < config.io_config.bin_construct_sample_cnt ? nrow : config.io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    sample_cnt = static_cast<int>(sample_indices.size());
    std::vector<std::vector<double>> sample_values(ncol);
    std::vector<std::vector<int>> sample_idx(ncol);
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      auto idx = sample_indices[i];
      auto row = get_row_fun(static_cast<int>(idx));
      for (size_t j = 0; j < row.size(); ++j) {
        if (std::fabs(row[j]) > kZeroThreshold || std::isnan(row[j])) {
          sample_values[j].emplace_back(row[j]);
          sample_idx[j].emplace_back(static_cast<int>(i));
        }
      }
    }
    DatasetLoader loader(config.io_config, nullptr, 1, nullptr);
    ret.reset(loader.CostructFromSampleData(Common::Vector2Ptr<double>(sample_values).data(),
                                            Common::Vector2Ptr<int>(sample_idx).data(),
                                            static_cast<int>(sample_values.size()),
                                            Common::VectorSize<double>(sample_values).data(),
                                            sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow));
    ret->CreateValid(
      reinterpret_cast<const Dataset*>(reference));
  }
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nrow; ++i) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

int LGBM_DatasetCreateFromCSR(const void* indptr,
                              int indptr_type,
                              const int32_t* indices,
                              const void* data,
                              int data_type,
                              int64_t nindptr,
                              int64_t nelem,
                              int64_t num_col,
                              const char* parameters,
                              const DatasetHandle reference,
                              DatasetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  OverallConfig config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  std::unique_ptr<Dataset> ret;
  auto get_row_fun = RowFunctionFromCSR(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  int32_t nrow = static_cast<int32_t>(nindptr - 1);
  if (reference == nullptr) {
    // sample data first
    Random rand(config.io_config.data_random_seed);
    int sample_cnt = static_cast<int>(nrow < config.io_config.bin_construct_sample_cnt ? nrow : config.io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    sample_cnt = static_cast<int>(sample_indices.size());
    std::vector<std::vector<double>> sample_values(num_col);
    std::vector<std::vector<int>> sample_idx(num_col);
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      auto idx = sample_indices[i];
      auto row = get_row_fun(static_cast<int>(idx));
      for (std::pair<int, double>& inner_data : row) {
        CHECK(inner_data.first < num_col);
        if (std::fabs(inner_data.second) > kZeroThreshold || std::isnan(inner_data.second)) {
          sample_values[inner_data.first].emplace_back(inner_data.second);
          sample_idx[inner_data.first].emplace_back(static_cast<int>(i));
        }
      }
    }
    DatasetLoader loader(config.io_config, nullptr, 1, nullptr);
    ret.reset(loader.CostructFromSampleData(Common::Vector2Ptr<double>(sample_values).data(),
                                            Common::Vector2Ptr<int>(sample_idx).data(),
                                            static_cast<int>(sample_values.size()),
                                            Common::VectorSize<double>(sample_values).data(),
                                            sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow));
    ret->CreateValid(
      reinterpret_cast<const Dataset*>(reference));
  }
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nindptr - 1; ++i) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

int LGBM_DatasetCreateFromCSC(const void* col_ptr,
                              int col_ptr_type,
                              const int32_t* indices,
                              const void* data,
                              int data_type,
                              int64_t ncol_ptr,
                              int64_t nelem,
                              int64_t num_row,
                              const char* parameters,
                              const DatasetHandle reference,
                              DatasetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  OverallConfig config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  std::unique_ptr<Dataset> ret;
  int32_t nrow = static_cast<int32_t>(num_row);
  if (reference == nullptr) {
    // sample data first
    Random rand(config.io_config.data_random_seed);
    int sample_cnt = static_cast<int>(nrow < config.io_config.bin_construct_sample_cnt ? nrow : config.io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    sample_cnt = static_cast<int>(sample_indices.size());
    std::vector<std::vector<double>> sample_values(ncol_ptr - 1);
    std::vector<std::vector<int>> sample_idx(ncol_ptr - 1);
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(sample_values.size()); ++i) {
      OMP_LOOP_EX_BEGIN();
      CSC_RowIterator col_it(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, i);
      for (int j = 0; j < sample_cnt; j++) {
        auto val = col_it.Get(sample_indices[j]);
        if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
          sample_values[i].emplace_back(val);
          sample_idx[i].emplace_back(j);
        }
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    DatasetLoader loader(config.io_config, nullptr, 1, nullptr);
    ret.reset(loader.CostructFromSampleData(Common::Vector2Ptr<double>(sample_values).data(),
                                            Common::Vector2Ptr<int>(sample_idx).data(),
                                            static_cast<int>(sample_values.size()),
                                            Common::VectorSize<double>(sample_values).data(),
                                            sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow));
    ret->CreateValid(
      reinterpret_cast<const Dataset*>(reference));
  }
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < ncol_ptr - 1; ++i) {
    OMP_LOOP_EX_BEGIN();
    const int tid = omp_get_thread_num();
    int feature_idx = ret->InnerFeatureIndex(i);
    if (feature_idx < 0) { continue; }
    int group = ret->Feature2Group(feature_idx);
    int sub_feature = ret->Feture2SubFeature(feature_idx);
    CSC_RowIterator col_it(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, i);
    int row_idx = 0;
    while (row_idx < nrow) {
      auto pair = col_it.NextNonZero();
      row_idx = pair.first;
      // no more data
      if (row_idx < 0) { break; }
      ret->PushOneData(tid, row_idx, group, sub_feature, pair.second);
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

int LGBM_DatasetGetSubset(
  const DatasetHandle handle,
  const int32_t* used_row_indices,
  int32_t num_used_row_indices,
  const char* parameters,
  DatasetHandle* out) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameters);
  OverallConfig config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  auto full_dataset = reinterpret_cast<const Dataset*>(handle);
  CHECK(num_used_row_indices > 0);
  const int32_t lower = 0;
  const int32_t upper = full_dataset->num_data() - 1;
  Common::CheckElementsIntervalClosed(used_row_indices, lower, upper, num_used_row_indices, "Used indices of subset");
  auto ret = std::unique_ptr<Dataset>(new Dataset(num_used_row_indices));
  ret->CopyFeatureMapperFrom(full_dataset);
  ret->CopySubset(full_dataset, used_row_indices, num_used_row_indices, true);
  *out = ret.release();
  API_END();
}

int LGBM_DatasetSetFeatureNames(
  DatasetHandle handle,
  const char** feature_names,
  int num_feature_names) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  std::vector<std::string> feature_names_str;
  for (int i = 0; i < num_feature_names; ++i) {
    feature_names_str.emplace_back(feature_names[i]);
  }
  dataset->set_feature_names(feature_names_str);
  API_END();
}

int LGBM_DatasetGetFeatureNames(
  DatasetHandle handle,
  char** feature_names,
  int* num_feature_names) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  auto inside_feature_name = dataset->feature_names();
  *num_feature_names = static_cast<int>(inside_feature_name.size());
  for (int i = 0; i < *num_feature_names; ++i) {
    std::memcpy(feature_names[i], inside_feature_name[i].c_str(), inside_feature_name[i].size() + 1);
  }
  API_END();
}

#pragma warning(disable : 4702)
int LGBM_DatasetFree(DatasetHandle handle) {
  API_BEGIN();
  delete reinterpret_cast<Dataset*>(handle);
  API_END();
}

int LGBM_DatasetSaveBinary(DatasetHandle handle,
                           const char* filename) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  dataset->SaveBinaryFile(filename);
  API_END();
}

int LGBM_DatasetSetField(DatasetHandle handle,
                         const char* field_name,
                         const void* field_data,
                         int num_element,
                         int type) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  bool is_success = false;
  if (type == C_API_DTYPE_FLOAT32) {
    is_success = dataset->SetFloatField(field_name, reinterpret_cast<const float*>(field_data), static_cast<int32_t>(num_element));
  } else if (type == C_API_DTYPE_INT32) {
    is_success = dataset->SetIntField(field_name, reinterpret_cast<const int*>(field_data), static_cast<int32_t>(num_element));
  } else if (type == C_API_DTYPE_FLOAT64) {
    is_success = dataset->SetDoubleField(field_name, reinterpret_cast<const double*>(field_data), static_cast<int32_t>(num_element));
  }
  if (!is_success) { throw std::runtime_error("Input data type error or field not found"); }
  API_END();
}

int LGBM_DatasetGetField(DatasetHandle handle,
                         const char* field_name,
                         int* out_len,
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
  } else if (dataset->GetDoubleField(field_name, out_len, reinterpret_cast<const double**>(out_ptr))) {
    *out_type = C_API_DTYPE_FLOAT64;
    is_success = true;
  }
  if (!is_success) { throw std::runtime_error("Field not found"); }
  if (*out_ptr == nullptr) { *out_len = 0; }
  API_END();
}

int LGBM_DatasetGetNumData(DatasetHandle handle,
                           int* out) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  *out = dataset->num_data();
  API_END();
}

int LGBM_DatasetGetNumFeature(DatasetHandle handle,
                              int* out) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  *out = dataset->num_total_features();
  API_END();
}

// ---- start of booster

int LGBM_BoosterCreate(const DatasetHandle train_data,
                       const char* parameters,
                       BoosterHandle* out) {
  API_BEGIN();
  const Dataset* p_train_data = reinterpret_cast<const Dataset*>(train_data);
  auto ret = std::unique_ptr<Booster>(new Booster(p_train_data, parameters));
  *out = ret.release();
  API_END();
}

int LGBM_BoosterCreateFromModelfile(
  const char* filename,
  int* out_num_iterations,
  BoosterHandle* out) {
  API_BEGIN();
  auto ret = std::unique_ptr<Booster>(new Booster(filename));
  *out_num_iterations = ret->GetBoosting()->GetCurrentIteration();
  *out = ret.release();
  API_END();
}

int LGBM_BoosterLoadModelFromString(
  const char* model_str,
  int* out_num_iterations,
  BoosterHandle* out) {
  API_BEGIN();
  auto ret = std::unique_ptr<Booster>(new Booster(nullptr));
  ret->LoadModelFromString(model_str);
  *out_num_iterations = ret->GetBoosting()->GetCurrentIteration();
  *out = ret.release();
  API_END();
}

#pragma warning(disable : 4702)
int LGBM_BoosterFree(BoosterHandle handle) {
  API_BEGIN();
  delete reinterpret_cast<Booster*>(handle);
  API_END();
}

int LGBM_BoosterMerge(BoosterHandle handle,
                      BoosterHandle other_handle) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  Booster* ref_other_booster = reinterpret_cast<Booster*>(other_handle);
  ref_booster->MergeFrom(ref_other_booster);
  API_END();
}

int LGBM_BoosterAddValidData(BoosterHandle handle,
                             const DatasetHandle valid_data) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  const Dataset* p_dataset = reinterpret_cast<const Dataset*>(valid_data);
  ref_booster->AddValidData(p_dataset);
  API_END();
}

int LGBM_BoosterResetTrainingData(BoosterHandle handle,
                                  const DatasetHandle train_data) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  const Dataset* p_dataset = reinterpret_cast<const Dataset*>(train_data);
  ref_booster->ResetTrainingData(p_dataset);
  API_END();
}

int LGBM_BoosterResetParameter(BoosterHandle handle, const char* parameters) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->ResetConfig(parameters);
  API_END();
}

int LGBM_BoosterGetNumClasses(BoosterHandle handle, int* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetBoosting()->NumberOfClasses();
  API_END();
}

int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  if (ref_booster->TrainOneIter()) {
    *is_finished = 1;
  } else {
    *is_finished = 0;
  }
  API_END();
}

int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
                                    const float* grad,
                                    const float* hess,
                                    int* is_finished) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  #ifdef SCORE_T_USE_DOUBLE
  Log::Fatal("Don't support custom loss function when SCORE_T_USE_DOUBLE is enabled");
  #else
  if (ref_booster->TrainOneIter(grad, hess)) {
    *is_finished = 1;
  } else {
    *is_finished = 0;
  }
  #endif
  API_END();
}

int LGBM_BoosterRollbackOneIter(BoosterHandle handle) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->RollbackOneIter();
  API_END();
}

int LGBM_BoosterGetCurrentIteration(BoosterHandle handle, int* out_iteration) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_iteration = ref_booster->GetBoosting()->GetCurrentIteration();
  API_END();
}

int LGBM_BoosterGetEvalCounts(BoosterHandle handle, int* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetEvalCounts();
  API_END();
}

int LGBM_BoosterGetEvalNames(BoosterHandle handle, int* out_len, char** out_strs) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetEvalNames(out_strs);
  API_END();
}

int LGBM_BoosterGetFeatureNames(BoosterHandle handle, int* out_len, char** out_strs) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetFeatureNames(out_strs);
  API_END();
}

int LGBM_BoosterGetNumFeature(BoosterHandle handle, int* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = ref_booster->GetBoosting()->MaxFeatureIdx() + 1;
  API_END();
}

int LGBM_BoosterGetEval(BoosterHandle handle,
                        int data_idx,
                        int* out_len,
                        double* out_results) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto boosting = ref_booster->GetBoosting();
  auto result_buf = boosting->GetEvalAt(data_idx);
  *out_len = static_cast<int>(result_buf.size());
  for (size_t i = 0; i < result_buf.size(); ++i) {
    (out_results)[i] = static_cast<double>(result_buf[i]);
  }
  API_END();
}

int LGBM_BoosterGetNumPredict(BoosterHandle handle,
                              int data_idx,
                              int64_t* out_len) {
  API_BEGIN();
  auto boosting = reinterpret_cast<Booster*>(handle)->GetBoosting();
  *out_len = boosting->GetNumPredictAt(data_idx);
  API_END();
}

int LGBM_BoosterGetPredict(BoosterHandle handle,
                           int data_idx,
                           int64_t* out_len,
                           double* out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->GetPredictAt(data_idx, out_result, out_len);
  API_END();
}

int LGBM_BoosterPredictForFile(BoosterHandle handle,
                               const char* data_filename,
                               int data_has_header,
                               int predict_type,
                               int num_iteration,
                               const char* parameter,
                               const char* result_filename) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameter);
  OverallConfig config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->Predict(num_iteration, predict_type, data_filename, data_has_header,
                       config.io_config, result_filename);
  API_END();
}

int LGBM_BoosterCalcNumPredict(BoosterHandle handle,
                               int num_row,
                               int predict_type,
                               int num_iteration,
                               int64_t* out_len) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_len = static_cast<int64_t>(num_row * ref_booster->GetBoosting()->NumPredictOneRow(
    num_iteration, predict_type == C_API_PREDICT_LEAF_INDEX, predict_type == C_API_PREDICT_CONTRIB));
  API_END();
}

int LGBM_BoosterPredictForCSR(BoosterHandle handle,
                              const void* indptr,
                              int indptr_type,
                              const int32_t* indices,
                              const void* data,
                              int data_type,
                              int64_t nindptr,
                              int64_t nelem,
                              int64_t,
                              int predict_type,
                              int num_iteration,
                              const char* parameter,
                              int64_t* out_len,
                              double* out_result) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameter);
  OverallConfig config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto get_row_fun = RowFunctionFromCSR(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  int nrow = static_cast<int>(nindptr - 1);
  ref_booster->Predict(num_iteration, predict_type, nrow, get_row_fun,
                       config.io_config, out_result, out_len);
  API_END();
}

int LGBM_BoosterPredictForCSC(BoosterHandle handle,
                              const void* col_ptr,
                              int col_ptr_type,
                              const int32_t* indices,
                              const void* data,
                              int data_type,
                              int64_t ncol_ptr,
                              int64_t nelem,
                              int64_t num_row,
                              int predict_type,
                              int num_iteration,
                              const char* parameter,
                              int64_t* out_len,
                              double* out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto param = ConfigBase::Str2Map(parameter);
  OverallConfig config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  int num_threads = 1;
  #pragma omp parallel
  #pragma omp master
  {
    num_threads = omp_get_num_threads();
  }
  int ncol = static_cast<int>(ncol_ptr - 1);
  std::vector<std::vector<CSC_RowIterator>> iterators(num_threads, std::vector<CSC_RowIterator>());
  for (int i = 0; i < num_threads; ++i) {
    for (int j = 0; j < ncol; ++j) {
      iterators[i].emplace_back(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, j);
    }
  }
  std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun =
    [&iterators, ncol] (int i) {
    std::vector<std::pair<int, double>> one_row;
    const int tid = omp_get_thread_num();
    for (int j = 0; j < ncol; ++j) {
      auto val = iterators[tid][j].Get(i);
      if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
        one_row.emplace_back(j, val);
      }
    }
    return one_row;
  };
  ref_booster->Predict(num_iteration, predict_type, static_cast<int>(num_row), get_row_fun, config.io_config,
                       out_result, out_len);
  API_END();
}

int LGBM_BoosterPredictForMat(BoosterHandle handle,
                              const void* data,
                              int data_type,
                              int32_t nrow,
                              int32_t ncol,
                              int is_row_major,
                              int predict_type,
                              int num_iteration,
                              const char* parameter,
                              int64_t* out_len,
                              double* out_result) {
  API_BEGIN();
  auto param = ConfigBase::Str2Map(parameter);
  OverallConfig config;
  config.Set(param);
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto get_row_fun = RowPairFunctionFromDenseMatric(data, nrow, ncol, data_type, is_row_major);
  ref_booster->Predict(num_iteration, predict_type, nrow, get_row_fun,
                       config.io_config, out_result, out_len);
  API_END();
}

int LGBM_BoosterSaveModel(BoosterHandle handle,
                          int num_iteration,
                          const char* filename) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->SaveModelToFile(num_iteration, filename);
  API_END();
}

int LGBM_BoosterSaveModelToString(BoosterHandle handle,
                                  int num_iteration,
                                  int64_t buffer_len, 
                                  int64_t* out_len,
                                  char* out_str) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  std::string model = ref_booster->SaveModelToString(num_iteration);
  *out_len = static_cast<int64_t>(model.size()) + 1;
  if (*out_len <= buffer_len) {
    std::memcpy(out_str, model.c_str(), *out_len);
  }
  API_END();
}

int LGBM_BoosterDumpModel(BoosterHandle handle,
                          int num_iteration,
                          int64_t buffer_len,
                          int64_t* out_len,
                          char* out_str) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  std::string model = ref_booster->DumpModel(num_iteration);
  *out_len = static_cast<int64_t>(model.size()) + 1;
  if (*out_len <= buffer_len) {
    std::memcpy(out_str, model.c_str(), *out_len);
  }
  API_END();
}

int LGBM_BoosterGetLeafValue(BoosterHandle handle,
                             int tree_idx,
                             int leaf_idx,
                             double* out_val) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  *out_val = static_cast<double>(ref_booster->GetLeafValue(tree_idx, leaf_idx));
  API_END();
}

int LGBM_BoosterSetLeafValue(BoosterHandle handle,
                             int tree_idx,
                             int leaf_idx,
                             double val) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->SetLeafValue(tree_idx, leaf_idx, val);
  API_END();
}

int LGBM_BoosterFeatureImportance(BoosterHandle handle,
                                  int num_iteration,
                                  int importance_type,
                                  double* out_results) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  std::vector<double> feature_importances = ref_booster->FeatureImportance(num_iteration, importance_type);
  for (size_t i = 0; i < feature_importances.size(); ++i) {
    (out_results)[i] = feature_importances[i];
  }
  API_END();
}

int LGBM_NetworkInit(const char* machines,
                     int local_listen_port,
                     int listen_time_out,
                     int num_machines) {
  API_BEGIN();
  NetworkConfig config;
  config.machines = Common::RemoveQuotationSymbol(std::string(machines));
  config.local_listen_port = local_listen_port;
  config.num_machines = num_machines;
  config.time_out = listen_time_out;
  if (num_machines > 1) {
    Network::Init(config);
  }
  API_END();
}

int LGBM_NetworkFree() {
  API_BEGIN();
  Network::Dispose();
  API_END();
}

int LGBM_NetworkInitWithFunctions(int num_machines, int rank,
                                  void* reduce_scatter_ext_fun,
                                  void* allgather_ext_fun) {
  API_BEGIN();
  if (num_machines > 1) {
    Network::Init(num_machines, rank, (ReduceScatterFunction)reduce_scatter_ext_fun, (AllgatherFunction)allgather_ext_fun);
  }
  API_END();
}

// ---- start of some help functions

std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major) {
  if (data_type == C_API_DTYPE_FLOAT32) {
    const float* data_ptr = reinterpret_cast<const float*>(data);
    if (is_row_major) {
      return [data_ptr, num_col, num_row] (int row_idx) {
        std::vector<double> ret(num_col);
        auto tmp_ptr = data_ptr + static_cast<size_t>(num_col) * row_idx;
        for (int i = 0; i < num_col; ++i) {
          ret[i] = static_cast<double>(*(tmp_ptr + i));
        }
        return ret;
      };
    } else {
      return [data_ptr, num_col, num_row] (int row_idx) {
        std::vector<double> ret(num_col);
        for (int i = 0; i < num_col; ++i) {
          ret[i] = static_cast<double>(*(data_ptr + static_cast<size_t>(num_row) * i + row_idx));
        }
        return ret;
      };
    }
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    const double* data_ptr = reinterpret_cast<const double*>(data);
    if (is_row_major) {
      return [data_ptr, num_col, num_row] (int row_idx) {
        std::vector<double> ret(num_col);
        auto tmp_ptr = data_ptr + static_cast<size_t>(num_col) * row_idx;
        for (int i = 0; i < num_col; ++i) {
          ret[i] = static_cast<double>(*(tmp_ptr + i));
        }
        return ret;
      };
    } else {
      return [data_ptr, num_col, num_row] (int row_idx) {
        std::vector<double> ret(num_col);
        for (int i = 0; i < num_col; ++i) {
          ret[i] = static_cast<double>(*(data_ptr + static_cast<size_t>(num_row) * i + row_idx));
        }
        return ret;
      };
    }
  }
  throw std::runtime_error("Unknown data type in RowFunctionFromDenseMatric");
}

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major) {
  auto inner_function = RowFunctionFromDenseMatric(data, num_row, num_col, data_type, is_row_major);
  if (inner_function != nullptr) {
    return [inner_function] (int row_idx) {
      auto raw_values = inner_function(row_idx);
      std::vector<std::pair<int, double>> ret;
      for (int i = 0; i < static_cast<int>(raw_values.size()); ++i) {
        if (std::fabs(raw_values[i]) > kZeroThreshold || std::isnan(raw_values[i])) {
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
      return [ptr_indptr, indices, data_ptr, nindptr, nelem] (int idx) {
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
      return [ptr_indptr, indices, data_ptr, nindptr, nelem] (int idx) {
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
      return [ptr_indptr, indices, data_ptr, nindptr, nelem] (int idx) {
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
      return [ptr_indptr, indices, data_ptr, nindptr, nelem] (int idx) {
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
  throw std::runtime_error("Unknown data type in RowFunctionFromCSR");
}

std::function<std::pair<int, double>(int idx)>
IterateFunctionFromCSC(const void* col_ptr, int col_ptr_type, const int32_t* indices, const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx) {
  CHECK(col_idx < ncol_ptr && col_idx >= 0);
  if (data_type == C_API_DTYPE_FLOAT32) {
    const float* data_ptr = reinterpret_cast<const float*>(data);
    if (col_ptr_type == C_API_DTYPE_INT32) {
      const int32_t* ptr_col_ptr = reinterpret_cast<const int32_t*>(col_ptr);
      int64_t start = ptr_col_ptr[col_idx];
      int64_t end = ptr_col_ptr[col_idx + 1];
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem, start, end] (int bias) {
        int64_t i = static_cast<int64_t>(start + bias);
        if (i >= end) {
          return std::make_pair(-1, 0.0);
        }
        int idx = static_cast<int>(indices[i]);
        double val = static_cast<double>(data_ptr[i]);
        return std::make_pair(idx, val);
      };
    } else if (col_ptr_type == C_API_DTYPE_INT64) {
      const int64_t* ptr_col_ptr = reinterpret_cast<const int64_t*>(col_ptr);
      int64_t start = ptr_col_ptr[col_idx];
      int64_t end = ptr_col_ptr[col_idx + 1];
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem, start, end] (int bias) {
        int64_t i = static_cast<int64_t>(start + bias);
        if (i >= end) {
          return std::make_pair(-1, 0.0);
        }
        int idx = static_cast<int>(indices[i]);
        double val = static_cast<double>(data_ptr[i]);
        return std::make_pair(idx, val);
      };
    }
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    const double* data_ptr = reinterpret_cast<const double*>(data);
    if (col_ptr_type == C_API_DTYPE_INT32) {
      const int32_t* ptr_col_ptr = reinterpret_cast<const int32_t*>(col_ptr);
      int64_t start = ptr_col_ptr[col_idx];
      int64_t end = ptr_col_ptr[col_idx + 1];
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem, start, end] (int bias) {
        int64_t i = static_cast<int64_t>(start + bias);
        if (i >= end) {
          return std::make_pair(-1, 0.0);
        }
        int idx = static_cast<int>(indices[i]);
        double val = static_cast<double>(data_ptr[i]);
        return std::make_pair(idx, val);
      };
    } else if (col_ptr_type == C_API_DTYPE_INT64) {
      const int64_t* ptr_col_ptr = reinterpret_cast<const int64_t*>(col_ptr);
      int64_t start = ptr_col_ptr[col_idx];
      int64_t end = ptr_col_ptr[col_idx + 1];
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem, start, end] (int bias) {
        int64_t i = static_cast<int64_t>(start + bias);
        if (i >= end) {
          return std::make_pair(-1, 0.0);
        }
        int idx = static_cast<int>(indices[i]);
        double val = static_cast<double>(data_ptr[i]);
        return std::make_pair(idx, val);
      };
    }
  }
  throw std::runtime_error("Unknown data type in CSC matrix");
}

CSC_RowIterator::CSC_RowIterator(const void* col_ptr, int col_ptr_type, const int32_t* indices,
                                 const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx) {
  iter_fun_ = IterateFunctionFromCSC(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, col_idx);
}

double CSC_RowIterator::Get(int idx) {
  while (idx > cur_idx_ && !is_end_) {
    auto ret = iter_fun_(nonzero_idx_);
    if (ret.first < 0) {
      is_end_ = true;
      break;
    }
    cur_idx_ = ret.first;
    cur_val_ = ret.second;
    ++nonzero_idx_;
  }
  if (idx == cur_idx_) {
    return cur_val_;
  } else {
    return 0.0f;
  }
}

std::pair<int, double> CSC_RowIterator::NextNonZero() {
  if (!is_end_) {
    auto ret = iter_fun_(nonzero_idx_);
    ++nonzero_idx_;
    if (ret.first < 0) {
      is_end_ = true;
    }
    return ret;
  } else {
    return std::make_pair(-1, 0.0);
  }
}
