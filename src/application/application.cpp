#include <LightGBM/application.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/text_reader.h>

#include <LightGBM/network.h>
#include <LightGBM/dataset.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/metric.h>

#include "predictor.hpp"

#include <LightGBM/utils/openmp_wrapper.h>

#include <cstdio>
#include <ctime>

#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace LightGBM {

Application::Application(int argc, char** argv) {
  LoadParameters(argc, argv);
  // set number of threads for openmp
  if (config_.num_threads > 0) {
    omp_set_num_threads(config_.num_threads);
  }
  if (config_.io_config.data_filename.size() == 0 && config_.task_type != TaskType::kConvertModel) {
    Log::Fatal("No training/prediction data, application quit");
  }
  omp_set_nested(0);
}

Application::~Application() {
  if (config_.is_parallel) {
    Network::Dispose();
  }
}

void Application::LoadParameters(int argc, char** argv) {
  std::unordered_map<std::string, std::string> params;
  for (int i = 1; i < argc; ++i) {
    std::vector<std::string> tmp_strs = Common::Split(argv[i], '=');
    if (tmp_strs.size() == 2) {
      std::string key = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[0]));
      std::string value = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[1]));
      if (key.size() <= 0) {
        continue;
      }
      params[key] = value;
    } else {
      Log::Warning("Unknown parameter in command line: %s", argv[i]);
    }
  }
  // check for alias
  ParameterAlias::KeyAliasTransform(&params);
  // read parameters from config file
  if (params.count("config_file") > 0) {
    TextReader<size_t> config_reader(params["config_file"].c_str(), false);
    config_reader.ReadAllLines();
    if (!config_reader.Lines().empty()) {
      for (auto& line : config_reader.Lines()) {
        // remove str after "#"
        if (line.size() > 0 && std::string::npos != line.find_first_of("#")) {
          line.erase(line.find_first_of("#"));
        }
        line = Common::Trim(line);
        if (line.size() == 0) {
          continue;
        }
        std::vector<std::string> tmp_strs = Common::Split(line.c_str(), '=');
        if (tmp_strs.size() == 2) {
          std::string key = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[0]));
          std::string value = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[1]));
          if (key.size() <= 0) {
            continue;
          }
          // Command-line has higher priority
          if (params.count(key) == 0) {
            params[key] = value;
          }
        } else {
          Log::Warning("Unknown parameter in config file: %s", line.c_str());
        }
      }
    } else {
      Log::Warning("Config file %s doesn't exist, will ignore",
                   params["config_file"].c_str());
    }
  }
  // check for alias again
  ParameterAlias::KeyAliasTransform(&params);
  // load configs
  config_.Set(params);
  Log::Info("Finished loading parameters");
}

void Application::LoadData() {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::unique_ptr<Predictor> predictor;
  // prediction is needed if using input initial model(continued train)
  PredictFunction predict_fun = nullptr;
  PredictionEarlyStopInstance pred_early_stop = CreatePredictionEarlyStopInstance("none", LightGBM::PredictionEarlyStopConfig());
  // need to continue training
  if (boosting_->NumberOfTotalModel() > 0) {
    predictor.reset(new Predictor(boosting_.get(), -1, true, false, false, -1, -1));
    predict_fun = predictor->GetPredictFunction();
  }

  // sync up random seed for data partition
  if (config_.is_parallel_find_bin) {
    config_.io_config.data_random_seed =
      GlobalSyncUpByMin<int>(config_.io_config.data_random_seed);
  }

  DatasetLoader dataset_loader(config_.io_config, predict_fun,
                               config_.boosting_config.num_class, config_.io_config.data_filename.c_str());
  // load Training data
  if (config_.is_parallel_find_bin) {
    // load data for parallel training
    train_data_.reset(dataset_loader.LoadFromFile(config_.io_config.data_filename.c_str(),
                                                  Network::rank(), Network::num_machines()));
  } else {
    // load data for single machine
    train_data_.reset(dataset_loader.LoadFromFile(config_.io_config.data_filename.c_str(), 0, 1));
  }
  // need save binary file
  if (config_.io_config.is_save_binary_file) {
    train_data_->SaveBinaryFile(nullptr);
  }
  // create training metric
  if (config_.boosting_config.is_provide_training_metric) {
    for (auto metric_type : config_.metric_types) {
      auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_.metric_config));
      if (metric == nullptr) { continue; }
      metric->Init(train_data_->metadata(), train_data_->num_data());
      train_metric_.push_back(std::move(metric));
    }
  }
  train_metric_.shrink_to_fit();


  if (!config_.metric_types.empty()) {
    // only when have metrics then need to construct validation data

    // Add validation data, if it exists
    for (size_t i = 0; i < config_.io_config.valid_data_filenames.size(); ++i) {
      // add
      auto new_dataset = std::unique_ptr<Dataset>(
        dataset_loader.LoadFromFileAlignWithOtherDataset(
          config_.io_config.valid_data_filenames[i].c_str(),
          train_data_.get())
        );
      valid_datas_.push_back(std::move(new_dataset));
      // need save binary file
      if (config_.io_config.is_save_binary_file) {
        valid_datas_.back()->SaveBinaryFile(nullptr);
      }

      // add metric for validation data
      valid_metrics_.emplace_back();
      for (auto metric_type : config_.metric_types) {
        auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_.metric_config));
        if (metric == nullptr) { continue; }
        metric->Init(valid_datas_.back()->metadata(),
                     valid_datas_.back()->num_data());
        valid_metrics_.back().push_back(std::move(metric));
      }
      valid_metrics_.back().shrink_to_fit();
    }
    valid_datas_.shrink_to_fit();
    valid_metrics_.shrink_to_fit();
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  // output used time on each iteration
  Log::Info("Finished loading data in %f seconds",
            std::chrono::duration<double, std::milli>(end_time - start_time) * 1e-3);
}

void Application::InitTrain() {
  if (config_.is_parallel) {
    // need init network
    Network::Init(config_.network_config);
    Log::Info("Finished initializing network");
    // sync global random seed for feature patition
    config_.boosting_config.tree_config.feature_fraction_seed =
      GlobalSyncUpByMin<int>(config_.boosting_config.tree_config.feature_fraction_seed);
    config_.boosting_config.tree_config.feature_fraction =
      GlobalSyncUpByMin<double>(config_.boosting_config.tree_config.feature_fraction);
    config_.boosting_config.drop_seed =
      GlobalSyncUpByMin<int>(config_.boosting_config.drop_seed);
  }

  // create boosting
  boosting_.reset(
    Boosting::CreateBoosting(config_.boosting_type,
                             config_.io_config.input_model.c_str()));
  // create objective function
  objective_fun_.reset(
    ObjectiveFunction::CreateObjectiveFunction(config_.objective_type,
                                               config_.objective_config));
  // load training data
  LoadData();
  // initialize the objective function
  objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
  // initialize the boosting
  boosting_->Init(&config_.boosting_config, train_data_.get(), objective_fun_.get(),
                  Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
  // add validation data into boosting
  for (size_t i = 0; i < valid_datas_.size(); ++i) {
    boosting_->AddValidDataset(valid_datas_[i].get(),
                               Common::ConstPtrInVectorWrapper<Metric>(valid_metrics_[i]));
  }
  Log::Info("Finished initializing training");
}

void Application::Train() {
  Log::Info("Started training...");
  int total_iter = config_.boosting_config.num_iterations;
  bool is_finished = false;
  bool need_eval = true;
  auto start_time = std::chrono::steady_clock::now();
  for (int iter = 0; iter < total_iter && !is_finished; ++iter) {
    is_finished = boosting_->TrainOneIter(nullptr, nullptr, need_eval);
    auto end_time = std::chrono::steady_clock::now();
    // output used time per iteration
    Log::Info("%f seconds elapsed, finished iteration %d", std::chrono::duration<double,
              std::milli>(end_time - start_time) * 1e-3, iter + 1);
    if (config_.io_config.snapshot_freq > 0 
        && (iter+1) % config_.io_config.snapshot_freq == 0) {
      std::string snapshot_out = config_.io_config.output_model + ".snapshot_iter_" + std::to_string(iter + 1);
      boosting_->SaveModelToFile(-1, snapshot_out.c_str());
    }
  }
  // save model to file
  boosting_->SaveModelToFile(-1, config_.io_config.output_model.c_str());
  // convert model to if-else statement code
  if (config_.convert_model_language == std::string("cpp")) {
    boosting_->SaveModelToIfElse(-1, config_.io_config.convert_model.c_str());
  }
  Log::Info("Finished training");
}

void Application::Predict() {
  // create predictor
  Predictor predictor(boosting_.get(), config_.io_config.num_iteration_predict, config_.io_config.is_predict_raw_score,
                      config_.io_config.is_predict_leaf_index, config_.io_config.pred_early_stop, 
                      config_.io_config.pred_early_stop_freq, config_.io_config.pred_early_stop_margin);
  predictor.Predict(config_.io_config.data_filename.c_str(),
                    config_.io_config.output_result.c_str(), config_.io_config.has_header);
  Log::Info("Finished prediction");
}

void Application::InitPredict() {
  boosting_.reset(
    Boosting::CreateBoosting(config_.io_config.input_model.c_str()));
  Log::Info("Finished initializing prediction");
}

void Application::ConvertModel() {
  boosting_.reset(
    Boosting::CreateBoosting(config_.boosting_type,
                             config_.io_config.input_model.c_str()));
  boosting_->SaveModelToIfElse(-1, config_.io_config.convert_model.c_str());
}

template<typename T>
T Application::GlobalSyncUpByMin(T& local) {
  T global = local;
  if (!config_.is_parallel) {
    // no need to sync if not parallel learning
    return global;
  }
  Network::Allreduce(reinterpret_cast<char*>(&local),
                     sizeof(local), sizeof(local),
                     reinterpret_cast<char*>(&global),
                     [](const char* src, char* dst, int len) {
    int used_size = 0;
    const int type_size = sizeof(T);
    const T *p1;
    T *p2;
    while (used_size < len) {
      p1 = reinterpret_cast<const T *>(src);
      p2 = reinterpret_cast<T *>(dst);
      if (*p1 < *p2) {
        std::memcpy(dst, src, type_size);
      }
      src += type_size;
      dst += type_size;
      used_size += type_size;
    }
  });
  return global;
}

}  // namespace LightGBM
